# search_v2.py — Member Feedback Search V2
#
# Architecture:
#   Layer 1  — Fuzzy match using healthcare-equivalent terms
#              (true synonyms only — same concept, different expression)
#   Layer 2  — Semantic match using direct comment embeddings
#              + multi-vector query-time expansion
#   Merge    — Reciprocal Rank Fusion across both layers
#   Rerank   — Strict LLM binary classification (default NOT_RELEVANT)
#
# Embedding cache:
#   Comments are embedded once and cached to ./embed_cache/*.npy
#   so subsequent searches load instantly.
#
# Usage:
#   python search_v2.py --csv feedback.csv --text-col comment
#   python search_v2.py --csv feedback.csv --text-col comment --id-col comment_id
#
# Dependencies:
#   pip install pandas numpy rapidfuzz openai google-generativeai

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

# Provider imports are optional — a missing SDK (e.g. openai on Streamlit Cloud)
# only disables that provider instead of breaking the whole module.
try:
    from provider_databricks import DatabricksProvider
except ImportError:
    DatabricksProvider = None
try:
    from provider_gemini import GeminiProvider, GEMINI_LLM_MODELS, GEMINI_EMBED_MODEL
except ImportError:
    GeminiProvider, GEMINI_LLM_MODELS, GEMINI_EMBED_MODEL = None, [], ""
try:
    from provider_local import LocalProvider
except ImportError:
    LocalProvider = None
try:
    from provider_ollama_gemini import OllamaGeminiProvider
except ImportError:
    OllamaGeminiProvider = None

_available = tuple(c for c in (DatabricksProvider, GeminiProvider,
                               LocalProvider, OllamaGeminiProvider) if c is not None)
Provider = Union[_available] if _available else object

# ── Constants ─────────────────────────────────────────────────────────────────

FUZZY_THRESHOLD       = 70    # token_set_ratio gate (0-100) to become a candidate

# ── Cosine thresholds — MEAN-CENTERED scale ──────────────────────────────────
# These apply AFTER center_normalize (see embedding_mean). Centering recentres
# random comment pairs near 0, so on this corpus relevant docs sit ~0.20-0.48 and
# noise sits near 0. (Raw gemini-embedding-2 cosines were ~0.6-0.9 — the old
# 0.55/0.70 values were inside the noise band and are NOT comparable.)
# Calibrated 2026-06-15 from real centered query scores (prior auth/copay/billing):
# every doc ≥0.30 was on-topic; 0.20-0.30 is the LLM-decides band.
SEMANTIC_MIN_COSINE   = 0.20  # cosine gate to enter the candidate pool
                              # raise toward 0.25 to cut LLM load; lower toward 0.15 for recall
SEMANTIC_AUTO_QUALIFY = 0.30  # cosine alone marks RELEVANT — skips LLM
                              # raise toward 0.33 if adjacent topics auto-qualify wrongly

FUZZY_PHRASE_QUALIFY  = 90    # phrase_score (0-100) needed for the fuzzy bypass:
                              #   100   = exact contiguous phrase present in the comment
                              #   90-99 = phrase with a minor typo
                              #           ("prior authorisation" scores 94.7 — a threshold
                              #            of 95 would miss single-letter typos)
FUZZY_BYPASS_MIN_COSINE = 0.20  # a phrase match must ALSO clear this centered cosine to
                                # bypass the LLM — filters passing mentions in off-topic
                                # comments (matches the semantic floor)

CLASSIFY_BATCH    = 15    # comments per LLM call — small batches keep Gemma accurate;
                          # large ones cause sloppy middle-of-list decisions
RRF_K             = 60    # RRF constant (standard value; higher → flatter merge)

# No caps on result counts anywhere in this pipeline.
# Fuzzy hits: everything above FUZZY_THRESHOLD or with a qualifying phrase match.
# Semantic hits: everything above SEMANTIC_MIN_COSINE (embedding similarity).
# Both sets pass through RRF merge; auto-qualified candidates skip the LLM,
# the rest go to LLM binary classification.

# ── JSON helper ───────────────────────────────────────────────────────────────

def _parse_json(raw: str, anchor: str = "") -> dict:
    """
    Parse LLM JSON output robustly. `anchor` is the expected first key
    (e.g. "decisions", "equivalents") — used to locate the JSON object
    when Gemma wraps it in prose.
    """
    s = raw.strip()
    # Strip markdown fences
    for fence in ["```json", "```"]:
        if s.startswith(fence):
            s = s[len(fence):]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    # Direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Anchored extraction — find the block starting at the expected key
    if anchor:
        m = re.search(r'\{"' + re.escape(anchor) + r'"[\s\S]*\}', s)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    # Generic fallback — first '{' to last '}'
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i : j + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No JSON object found in response", s, 0)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalize once at load so per-query search skips the division."""
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)


def embedding_mean(matrix: np.ndarray) -> np.ndarray:
    """
    Corpus mean vector, subtracted from doc AND query embeddings to undo
    anisotropy. gemini-embedding-2 packs every vector into a narrow cone
    (median cosine between two random comments ≈ 0.69), which crushes the
    usable score range. Subtracting the mean recentres random pairs near 0
    and roughly triples the discrimination spread, so cosine thresholds
    become meaningful again.

    Computed from the raw (un-normalized) matrix; the same vector must be
    applied to query embeddings before cosine (see center_normalize_query).
    """
    return matrix.mean(axis=0).astype(np.float32)


def center_normalize(matrix: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean, then L2-normalize each row. Use at load."""
    centered = matrix - mean
    return centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)


def center_normalize_query(vec: np.ndarray, mean: Optional[np.ndarray]) -> np.ndarray:
    """Apply the same centering + normalization to a single query vector."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if mean is not None:
        v = v - mean
    return v / (np.linalg.norm(v) + 1e-8)

# ═════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION — one LLM call produces inputs for both layers
# ═════════════════════════════════════════════════════════════════════════════

_EXPAND_SYSTEM = """\
You are a US health insurance member-experience search specialist.

Given a search query, produce BOTH lists below.

1. "equivalents" — terms that mean EXACTLY the same thing in US healthcare:
   alternative spellings, abbreviations and their expansions, informal member
   shorthand. EXCLUDE related-but-different processes, downstream effects,
   and broader categories.
   For "prior auth": ["prior authorization", "preauthorization", "preauth",
   "PA", "auth request", "pre-approval"]
   NOT: "claim denial" (different process), "step therapy" (different UM tool)

2. "variants" — 6-8 phrases, 8-20 words each, written the way a MEMBER would
   describe the SAME core problem: natural frustrated-member voice, covering
   different severity levels and journey stages. NOT adjacent topics.
   For "prior auth": "been waiting weeks for prior authorization approval with
   no update", "insurance keeps blocking my procedure I need pre-approval"
   NOT: "pharmacy wouldn't fill my prescription" (different topic)

Return ONLY this JSON — no preamble, no markdown, no explanations.
Do NOT repeat or paraphrase these instructions in your response.
{"equivalents": ["term1", "term2", ...], "variants": ["phrase1", "phrase2", ...]}
"""


def generate_expansions(provider: Provider, query: str) -> Tuple[List[str], List[str]]:
    """
    One LLM call returns both the fuzzy-layer equivalent terms and the
    semantic-layer phrasing variants (previously two separate calls).
    Falls back to the bare query for either list on failure.
    """
    equivalents: List[str] = []
    variants:    List[str] = []
    try:
        raw  = provider.llm_call(_EXPAND_SYSTEM, f'Search query: "{query}"')
        data = _parse_json(raw, anchor="equivalents")
        equivalents = [t for t in data.get("equivalents", []) if isinstance(t, str)]
        variants    = [v for v in data.get("variants", [])    if isinstance(v, str)]
    except Exception as e:
        print(f"  [WARN] Expansion call failed ({type(e).__name__}: {e}) — using bare query.")

    # Deduplicate equivalents, preserve order, original query always first
    seen: set = set()
    deduped: List[str] = []
    for t in [query] + equivalents:
        key = t.lower().strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(t)
    return deduped, (variants or [query])


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Fuzzy match with healthcare-equivalent terms
# ═════════════════════════════════════════════════════════════════════════════

def fuzzy_search(
    equivalent_terms: List[str],
    documents: List[Dict],
    threshold: int = FUZZY_THRESHOLD,
) -> List[Dict]:
    """
    Two lexical signals per comment, computed over punctuation-stripped text
    (rapidfuzz.utils.default_process — without it "auth." != "auth", which made
    true exact matches score ~67 and fall below the old threshold):

      fuzzy_score  — best token_set_ratio across terms (stored 0-1):
                     all term words present somewhere in the comment.
      phrase_score — best contiguous-window score (stored 0-100):
                     partial_ratio for terms ≥5 chars (typo-tolerant);
                     exact word-boundary match for short abbreviations like
                     "PA" (partial_ratio scores 100 inside "package").

    A comment becomes a candidate if either signal clears its gate.
    phrase_score ≥ FUZZY_PHRASE_QUALIFY later feeds the LLM bypass.
    Whole grid is scored in C++ across cores via process.cdist.
    """
    proc_terms_all = [utils.default_process(t) for t in equivalent_terms]
    keep = [i for i, t in enumerate(proc_terms_all) if t]
    if not keep or not documents:
        return []
    proc_terms = [proc_terms_all[i] for i in keep]
    terms      = [equivalent_terms[i] for i in keep]
    proc_texts = [utils.default_process(d["text"]) for d in documents]

    ts_grid = process.cdist(proc_terms, proc_texts,
                            scorer=fuzz.token_set_ratio, workers=-1)

    ph_grid   = np.zeros_like(ts_grid)
    long_ids  = [i for i, t in enumerate(proc_terms) if len(t) >= 5]
    short_ids = [i for i, t in enumerate(proc_terms) if len(t) < 5]
    if long_ids:
        ph_grid[long_ids] = process.cdist(
            [proc_terms[i] for i in long_ids], proc_texts,
            scorer=fuzz.partial_ratio, workers=-1,
        )
    for i in short_ids:
        pattern = re.compile(r"\b" + re.escape(proc_terms[i]) + r"\b")
        ph_grid[i] = [100.0 if pattern.search(t) else 0.0 for t in proc_texts]

    best_ts   = ts_grid.max(axis=0)
    best_term = ts_grid.argmax(axis=0)
    best_ph   = ph_grid.max(axis=0)

    results = []
    for j, doc in enumerate(documents):
        if best_ts[j] >= threshold or best_ph[j] >= FUZZY_PHRASE_QUALIFY:
            results.append({
                **doc,
                "fuzzy_score":  float(best_ts[j]) / 100.0,
                "phrase_score": float(best_ph[j]),
                "matched_term": terms[int(best_term[j])],
                "layers":       {"fuzzy"},
            })
    return sorted(results, key=lambda x: x["fuzzy_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Semantic search with query-time expansion
# ═════════════════════════════════════════════════════════════════════════════

def semantic_search(
    query_variants:  List[str],
    provider:        Provider,
    doc_matrix:      np.ndarray,
    documents:       List[Dict],
    min_cosine:      float = SEMANTIC_MIN_COSINE,
    query_mean:      Optional[np.ndarray] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Embeds all query variants (one batched API call when the provider supports
    embed_texts) and computes cosine similarity against every doc.

    Returns (hits, best_cosines):
      hits         — docs whose best cosine across any variant ≥ min_cosine,
                     RRF-accumulated across variants. No top-K cap.
      best_cosines — best cosine for EVERY doc (len == len(documents)), so the
                     merge step can attach a true cosine to fuzzy-only docs.

    doc_matrix MUST already be centered + normalized (see center_normalize);
    query_mean is the same corpus mean so query vectors get identical treatment.
    """
    # One round trip for all variants when the provider supports it
    embed_many = getattr(provider, "embed_texts", None)
    qvecs: List[List[float]] = []
    if callable(embed_many):
        try:
            qvecs = embed_many(query_variants, task_type="retrieval_query")
        except Exception as e:
            print(f"  [WARN] Batch embed failed ({type(e).__name__}) — falling back to per-variant calls.")
            qvecs = []
    if not qvecs:
        qvecs = [provider.embed_text(v, task_type="retrieval_query") for v in query_variants]

    rrf_scores:  Dict[int, float] = {}   # doc index → accumulated RRF
    best_cosines = np.full(len(documents), -1.0, dtype=np.float32)

    for vec in qvecs:
        qvec = center_normalize_query(vec, query_mean)  # same centering as docs
        sims = doc_matrix @ qvec  # cosine similarity for every doc — fast
        best_cosines = np.maximum(best_cosines, sims)

        above  = np.where(sims >= min_cosine)[0]
        ranked = above[np.argsort(-sims[above])]
        for rank, idx in enumerate(ranked):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

    results = []
    for idx, rrf in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        results.append({
            **documents[idx],
            "semantic_rrf":   rrf,
            "semantic_score": float(best_cosines[idx]),
            "layers":         {"semantic"},
        })
    return results, best_cosines


# ═════════════════════════════════════════════════════════════════════════════
# MERGE — Reciprocal Rank Fusion
# ═════════════════════════════════════════════════════════════════════════════

def rrf_merge(
    fuzzy_results:    List[Dict],
    semantic_results: List[Dict],
    best_cosines:     Optional[np.ndarray] = None,
    documents:        Optional[List[Dict]] = None,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Merges both retrieval layers with no caps on either side.

    Every fuzzy hit (matched a healthcare-equivalent term) and every
    semantic hit (cosine ≥ SEMANTIC_MIN_COSINE) enters the merged pool.
    Documents found by both layers accumulate RRF score from each rank,
    so consistent cross-layer matches naturally surface at the top.

    When best_cosines + documents are provided, fuzzy-only docs get their true
    cosine attached so bypass and floor logic see a real score instead of 0.

    Returns (merged_candidates_sorted_by_rrf, diagnostic_counts).
    """
    scores: Dict[int, float] = {}
    merged: Dict[int, Dict]  = {}

    for rank, r in enumerate(fuzzy_results):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        merged[doc_id] = dict(r)

    for rank, r in enumerate(semantic_results):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        if doc_id in merged:
            merged[doc_id]["layers"].add("semantic")
            merged[doc_id]["semantic_rrf"]   = r.get("semantic_rrf", 0.0)
            merged[doc_id]["semantic_score"] = r.get("semantic_score", 0.0)
        else:
            merged[doc_id] = dict(r)

    for doc_id, r in merged.items():
        r["rrf_score"] = scores[doc_id]

    if best_cosines is not None and documents is not None:
        idx_of = {d["doc_id"]: i for i, d in enumerate(documents)}
        for doc_id, r in merged.items():
            if "semantic_score" not in r:
                i = idx_of.get(doc_id)
                if i is not None:
                    r["semantic_score"] = float(best_cosines[i])

    fuzzy_ids   = {r["doc_id"] for r in fuzzy_results}
    both_layers = sum(
        1 for r in merged.values()
        if "fuzzy" in r.get("layers", set()) and "semantic" in r.get("layers", set())
    )
    counts = {
        "fuzzy":          len(fuzzy_ids),
        "semantic":       len(semantic_results),
        "both_layers":    both_layers,
        "semantic_only":  len(merged) - len(fuzzy_ids),
        "total":          len(merged),
    }
    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True), counts


# ═════════════════════════════════════════════════════════════════════════════
# RERANK — Strict LLM binary classification
# ═════════════════════════════════════════════════════════════════════════════

_RERANK_SYSTEM = """\
You are a health insurance member experience analyst.
Classify member feedback comments as RELEVANT or NOT_RELEVANT to a search query.
Default to NOT_RELEVANT. Only use RELEVANT when the comment's core topic matches the query.
Output ONLY the JSON object — no preamble, no summary, no explanation, no markdown.
Do NOT repeat or paraphrase these instructions in your response.
"""


def _classify_batch(batch: List[Dict], query: str, provider: Provider) -> List[Dict]:
    formatted = "\n\n".join(
        f"[doc_id={d['doc_id']}]\n{d['text']}"
        for d in batch
    )

    # Keep the user message short and task-focused.
    # Long instruction blocks cause Gemma to echo/summarize them before the JSON.
    user_msg = (
        f'Search query: "{query}"\n\n'
        f"For each comment, output RELEVANT if its main topic is directly about "
        f'"{query}", otherwise output NOT_RELEVANT.\n\n'
        f"Comments:\n\n{formatted}\n\n"
        f"Respond with ONLY this JSON. Do not repeat these instructions. "
        f"Do not explain. Do not add any text before or after the JSON.\n"
        f'{{"decisions":[{{"doc_id":1,"decision":"RELEVANT"}},{{"doc_id":2,"decision":"NOT_RELEVANT"}},...]}}'
    )

    raw: str = ""
    try:
        raw = provider.llm_call(_RERANK_SYSTEM, user_msg)
        decisions = {
            int(d["doc_id"]): d["decision"]
            for d in _parse_json(raw, anchor="decisions").get("decisions", [])
        }
        missing = [c["doc_id"] for c in batch if int(c["doc_id"]) not in decisions]
        if missing:
            print(f"  [WARN] LLM response missing {len(missing)}/{len(batch)} doc_id(s) "
                  f"{missing[:8]}{'...' if len(missing) > 8 else ''} — treated as NOT_RELEVANT.")
        return [c for c in batch if decisions.get(int(c["doc_id"]), "NOT_RELEVANT") == "RELEVANT"]
    except Exception as e:
        print(f"  [WARN] Batch classification failed: {type(e).__name__}: {e}")
        if raw:
            print(f"  [WARN] LLM raw response (first 400 chars):\n{raw[:400]}")
        else:
            print(f"  [WARN] LLM call returned no response.")
        print(f"  [WARN] Dropping {len(batch)} candidates — not marking them RELEVANT.")
        return []


def llm_rerank(candidates: List[Dict], query: str, provider: Provider) -> List[Dict]:
    """
    Three tiers:
      1. semantic_score ≥ SEMANTIC_AUTO_QUALIFY              → RELEVANT, no LLM
      2. phrase_score ≥ FUZZY_PHRASE_QUALIFY (term present
         as contiguous phrase, typo-tolerant) AND
         semantic_score ≥ FUZZY_BYPASS_MIN_COSINE            → RELEVANT, no LLM
      3. everything else → LLM binary classification, sequential batches
    """
    if not candidates:
        return []

    auto:      List[Dict] = []
    needs_llm: List[Dict] = []
    for c in candidates:
        cos = c.get("semantic_score", 0.0)
        if cos >= SEMANTIC_AUTO_QUALIFY:
            c["qualified_by"] = "high-cosine"
            auto.append(c)
        elif c.get("phrase_score", 0.0) >= FUZZY_PHRASE_QUALIFY and cos >= FUZZY_BYPASS_MIN_COSINE:
            c["qualified_by"] = "phrase-match"
            auto.append(c)
        else:
            needs_llm.append(c)

    if auto:
        n_cos = sum(1 for c in auto if c["qualified_by"] == "high-cosine")
        n_phr = len(auto) - n_cos
        print(f"        {len(auto)} auto-qualified — skipping LLM "
              f"(high-cosine {n_cos} · phrase-match {n_phr})")

    batches = [needs_llm[i : i + CLASSIFY_BATCH] for i in range(0, len(needs_llm), CLASSIFY_BATCH)]
    if batches:
        print(f"        {len(needs_llm)} candidates → {len(batches)} LLM batch(es) of {CLASSIFY_BATCH}")

    results: List[Dict] = list(auto)
    for i, batch in enumerate(batches, 1):
        print(f"        Batch {i}/{len(batches)} ...")
        results.extend(_classify_batch(batch, query, provider))

    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═════════════════════════════════════════════════════════════════════════════
#
# Each cache consists of two files:
#   <name>.npy        — float32 numpy matrix, one row per comment
#   <name>.meta.json  — sidecar with the four validity signals below
#
# Validity signals (all four must match to reuse the cache):
#   doc_count           — total rows; catches additions / deletions
#   csv_path            — absolute path; catches swapping to a different file
#   text_col            — column name; catches switching columns
#   content_fingerprint — SHA-256 of first-10 + last-10 texts (16 hex chars);
#                         catches edits to existing rows cheaply
#
# Cache filename encodes provider + embed model, so switching either
# automatically creates a new file without touching the existing one.
#
# On every run the user sees a comparison table (cached vs current) and
# must confirm before the cache is used or a rebuild is triggered.

import hashlib


# ── Helpers ───────────────────────────────────────────────────────────────────

def _content_fingerprint(documents: List[Dict], n: int = 10) -> str:
    sample = documents[:n] + (documents[-n:] if len(documents) > n else [])
    return hashlib.sha256("||".join(d["text"] for d in sample).encode()).hexdigest()[:16]


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def _load_meta(cache_path: Path) -> dict:
    mp = _meta_path(cache_path)
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_meta(cache_path: Path, meta: dict) -> None:
    with open(_meta_path(cache_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# ── Cache status check + display ──────────────────────────────────────────────

def _check_and_display_cache(
    cache_path: Path,
    documents:  List[Dict],
    csv_path:   str,
    text_col:   str,
) -> Tuple[bool, bool]:
    """
    Computes current expected values, loads the stored meta (if any), prints
    a side-by-side comparison table, and returns (cache_exists, cache_valid).

    The table is always shown so the user knows exactly what state the cache
    is in before being asked to confirm.
    """
    W = 62  # table width
    current_fp   = _content_fingerprint(documents)
    current_path = str(Path(csv_path).resolve())

    # ── No cache at all ───────────────────────────────────────────────────────
    if not cache_path.exists():
        print(f"\n  {'─'*W}")
        print(f"  Cache  : {cache_path.name}")
        print(f"  Status : NOT FOUND — will build on confirmation")
        print(f"  {'─'*W}")
        return False, False

    stored = _load_meta(cache_path)

    # ── No sidecar (old cache without meta) ───────────────────────────────────
    if not stored:
        print(f"\n  {'─'*W}")
        print(f"  Cache  : {cache_path.name}  (.npy exists, no metadata sidecar)")
        print(f"  Status : UNVERIFIABLE — treating as stale")
        print(f"  {'─'*W}")
        return True, False

    # ── Build comparison rows ─────────────────────────────────────────────────
    rows = [
        ("Documents",        str(stored.get("doc_count", "?")),  str(len(documents))),
        ("CSV file",         Path(stored.get("csv_path", "?")).name,
                             Path(current_path).name),
        ("Text column",      stored.get("text_col", "?"),         text_col),
        ("Content hash",     stored.get("content_fingerprint", "?")[:12] + "…",
                             current_fp[:12] + "…"),
        ("Provider",         stored.get("provider", "?"),         "—"),
        ("Embed model",      stored.get("embed_model", "?"),      "—"),
    ]

    check_keys = {
        "Documents":    (stored.get("doc_count"),           len(documents)),
        "CSV file":     (stored.get("csv_path"),            current_path),
        "Text column":  (stored.get("text_col"),            text_col),
        "Content hash": (stored.get("content_fingerprint"), current_fp),
    }
    mismatches = [label for label, (s, c) in check_keys.items() if s != c]
    is_valid   = len(mismatches) == 0

    # ── Print table ───────────────────────────────────────────────────────────
    col_label = 16
    col_stored = 22
    col_curr   = 18

    header  = f"  {'Field':<{col_label}}  {'Cached':<{col_stored}}  {'Current':<{col_curr}}  Match"
    divider = f"  {'─'*col_label}  {'─'*col_stored}  {'─'*col_curr}  ─────"

    print(f"\n  Cache : {cache_path.name}")
    print(f"  {'─'*W}")
    print(header)
    print(divider)

    for label, stored_val, curr_val in rows:
        is_checked = label in check_keys
        if not is_checked:
            mark = "  —  "   # informational row, not checked
        elif label in mismatches:
            mark = "  ✗  "
        else:
            mark = "  ✓  "

        sv = (stored_val[:col_stored-1] + "…") if len(stored_val) > col_stored else stored_val
        cv = (curr_val[:col_curr-1]   + "…") if len(curr_val)   > col_curr   else curr_val

        print(f"  {label:<{col_label}}  {sv:<{col_stored}}  {cv:<{col_curr}}{mark}")

    print(f"  {'─'*W}")

    if is_valid:
        print(f"  Status : VALID — all signals match")
    else:
        print(f"  Status : STALE — changed: {', '.join(mismatches)}")
    print(f"  {'─'*W}")

    return True, is_valid


# ── Build / load with confirmation ────────────────────────────────────────────

def _confirm(prompt: str, default_yes: bool) -> bool:
    hint    = "[Y/n]" if default_yes else "[y/N]"
    answer  = input(f"  {prompt} {hint}: ").strip().lower()
    if answer == "":
        return default_yes
    return answer in ("y", "yes")


def _embed_and_save(
    documents:  List[Dict],
    provider:   Provider,
    cache_path: Path,
    csv_path:   str,
    text_col:   str,
) -> np.ndarray:
    print(f"\n  Embedding {len(documents)} comments ...")
    vecs: List[List[float]] = []
    for i, doc in enumerate(documents):
        vec = provider.embed_text(doc["text"], task_type="retrieval_document")
        vecs.append(vec)
        if (i + 1) % 25 == 0 or (i + 1) == len(documents):
            print(f"    {i + 1}/{len(documents)}")
    matrix = np.array(vecs, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), matrix)
    _save_meta(cache_path, {
        "doc_count":           len(documents),
        "csv_path":            str(Path(csv_path).resolve()),
        "text_col":            text_col,
        "content_fingerprint": _content_fingerprint(documents),
        "provider":            provider.name,
        "embed_model":         provider.embed_model,
    })
    print(f"  Saved → {cache_path.name}  +  {_meta_path(cache_path).name}")
    return matrix


def build_or_load_embeddings(
    documents:  List[Dict],
    provider:   Provider,
    cache_path: Path,
    csv_path:   str = "",
    text_col:   str = "comment",
) -> np.ndarray:
    """
    Always shows a comparison table of cached vs current state, then asks
    the user to confirm before loading or rebuilding.

    · VALID cache   → asks "Use cached?" (default Y).  N triggers rebuild.
    · STALE cache   → shows what changed, asks "Rebuild?" (default Y).
                      N loads the stale cache anyway (user's explicit choice).
    · NO cache      → asks "Build embeddings now?" (default Y).
                      N exits — useful if user ran the wrong CSV by accident.
    """
    cache_exists, is_valid = _check_and_display_cache(
        cache_path, documents, csv_path, text_col
    )

    if not cache_exists:
        # ── No cache ──────────────────────────────────────────────────────────
        if not _confirm(f"Build embeddings for {len(documents)} comments?", default_yes=True):
            raise SystemExit("Aborted — no embeddings built.")
        return _embed_and_save(documents, provider, cache_path, csv_path, text_col)

    if is_valid:
        # ── Valid cache ───────────────────────────────────────────────────────
        use_cache = _confirm("Cache is valid. Use cached embeddings?", default_yes=True)
        if use_cache:
            matrix = np.load(str(cache_path)).astype(np.float32)
            if matrix.shape[0] != len(documents):
                print(f"  Shape mismatch ({matrix.shape[0]} rows vs {len(documents)} docs).")
                print(f"  Metadata said valid but .npy is inconsistent — rebuilding.")
                return _embed_and_save(documents, provider, cache_path, csv_path, text_col)
            print(f"  Loaded {matrix.shape[0]} embeddings from cache.")
            return matrix
        # User chose N on a valid cache — rebuild anyway
        return _embed_and_save(documents, provider, cache_path, csv_path, text_col)

    else:
        # ── Stale cache ───────────────────────────────────────────────────────
        rebuild = _confirm("Cache is stale. Rebuild embeddings?", default_yes=True)
        if rebuild:
            return _embed_and_save(documents, provider, cache_path, csv_path, text_col)
        # User explicitly chose to load stale cache
        print("  Loading stale cache as requested.")
        matrix = np.load(str(cache_path)).astype(np.float32)
        if matrix.shape[0] != len(documents):
            print(f"\n  WARNING: stale cache has {matrix.shape[0]} rows but CSV has "
                  f"{len(documents)} rows.")
            print(f"  Cosine search will be incorrect. Consider rebuilding.\n")
        return matrix


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def search(
    query:           str,
    documents:       List[Dict],
    doc_matrix:      np.ndarray,
    provider:        Provider,
    fuzzy_threshold: int   = FUZZY_THRESHOLD,
    min_cosine:      float = SEMANTIC_MIN_COSINE,
    verbose:         bool  = True,
    query_mean:      Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Full pipeline — no caps on any layer. Returns every comment both layers
    surface that the LLM then confirms as relevant.

    Tune fuzzy_threshold and min_cosine to control precision vs recall:
      Higher thresholds → fewer candidates, faster LLM rerank, possibly missed results
      Lower  thresholds → more candidates, more LLM calls, higher recall
    """
    if verbose:
        print(f"\n{'='*62}")
        print(f"  Query : '{query}'")
        print(f"  Gates : fuzzy ≥ {fuzzy_threshold}  |  cosine ≥ {min_cosine}")
        print(f"{'='*62}")

    # Step 1 — One LLM call expands the query for both layers
    if verbose: print("\n  [1/3] Query expansion (equivalents + variants) ...")
    equivalents, variants = generate_expansions(provider, query)
    if verbose:
        print("        Equivalent terms:")
        for t in equivalents:
            print(f"          · {t}")
        print("        Semantic variants:")
        for v in variants:
            print(f"          · {v}")

    # Step 2 — Both retrieval layers, no caps
    if verbose: print("\n  [2/3] Running retrieval layers ...")
    fuzzy_res = fuzzy_search(equivalents, documents, fuzzy_threshold)
    semantic_res, best_cos = semantic_search(variants, provider, doc_matrix, documents,
                                             min_cosine, query_mean=query_mean)
    merged, counts = rrf_merge(fuzzy_res, semantic_res,
                               best_cosines=best_cos, documents=documents)

    if verbose:
        print(f"        Fuzzy hits          : {counts['fuzzy']}  (threshold ≥ {fuzzy_threshold})")
        print(f"        Semantic hits       : {counts['semantic']}  (cosine ≥ {min_cosine})")
        print(f"        Found by both       : {counts['both_layers']}")
        print(f"        Semantic-only added : {counts['semantic_only']}")
        print(f"        Total → rerank      : {counts['total']}")

    # Step 3 — Auto-qualify tiers + LLM rerank for the rest
    if verbose: print(f"\n  [3/3] Rerank (auto-qualify + LLM) ...")
    final = llm_rerank(merged, query, provider)

    if verbose:
        only_fuzzy    = sum(1 for r in final if r.get("layers") == {"fuzzy"})
        only_semantic = sum(1 for r in final if r.get("layers") == {"semantic"})
        both          = sum(1 for r in final if "fuzzy" in r.get("layers", set()) and "semantic" in r.get("layers", set()))
        print(f"        Relevant : {len(final)}")
        print(f"          Fuzzy-only {only_fuzzy}  ·  Semantic-only {only_semantic}  ·  Both layers {both}")
        print(f"{'='*62}")

    return final


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def render_results(results: List[Dict], query: str):
    sep = "─" * 62
    if not results:
        print(f"\n  {sep}")
        print(f"  No relevant comments found for: '{query}'")
        print(f"  {sep}")
        return

    print(f"\n  {len(results)} relevant comment(s) for: '{query}'")
    print(f"  {sep}")

    for i, r in enumerate(results, 1):
        layers  = " + ".join(sorted(r.get("layers", set())))
        rrf     = r.get("rrf_score", 0.0)
        fscore  = r.get("fuzzy_score", 0.0)
        sscore  = r.get("semantic_score", 0.0)
        matched = r.get("matched_term", "")

        scores_str = f"RRF {rrf:.4f}"
        if fscore:  scores_str += f"  Fuzzy {fscore:.0%}"
        if r.get("phrase_score"):  scores_str += f"  Phrase {r['phrase_score']:.0f}"
        if sscore:  scores_str += f"  Cosine {sscore:.3f}"
        if r.get("qualified_by"): scores_str += f"  [auto: {r['qualified_by']}]"

        print(f"\n  [{i:>3}]  [{layers}]  {scores_str}")
        if matched:
            print(f"         Matched term : '{matched}'")
        text = r["text"]
        print(f"         {text[:130]}{'...' if len(text) > 130 else ''}")

    print(f"\n  {sep}")


# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def select_provider() -> Provider:
    print("\nSelect AI provider:")
    print("  1  Databricks    (OpenAI-compatible endpoint)")
    print("  2  Gemini        (Google AI — 2 s delay between calls)")
    print("  3  Local         (sentence-transformers + stub LLM — no API key needed)")
    print("  4  OllamaGemini  (ollama embedding + Google Generative AI LLM)")
    choice = input("Provider (1/2/3/4) [default 1]: ").strip() or "1"

    chosen = {"1": DatabricksProvider, "2": GeminiProvider,
              "3": LocalProvider, "4": OllamaGeminiProvider}.get(choice)
    if chosen is None:
        raise SystemExit(
            f"Provider option {choice} is unavailable — its SDK is not installed "
            f"(see provider_*.py for the required pip package)."
        )

    if choice == "4":
        api_key  = os.environ.get("GEMINI_API_KEY") or input("Gemini API key: ").strip()
        provider = OllamaGeminiProvider(api_key)
    elif choice == "3":
        provider = LocalProvider()
    elif choice == "2":
        api_key     = os.environ.get("GEMINI_API_KEY")     or input("Gemini API key: ").strip()
        llm_model   = (os.environ.get("GEMINI_LLM_MODEL")
                       or input(f"LLM model [{GEMINI_LLM_MODELS[0]}]: ").strip()
                       or GEMINI_LLM_MODELS[0])
        embed_model = (os.environ.get("GEMINI_EMBED_MODEL")
                       or input(f"Embed model [{GEMINI_EMBED_MODEL}]: ").strip()
                       or GEMINI_EMBED_MODEL)
        provider = GeminiProvider(api_key, llm_model, embed_model)
    else:
        base_url    = os.environ.get("DATABRICKS_BASE_URL") or input("Databricks Base URL : ").strip()
        token       = os.environ.get("DATABRICKS_TOKEN")    or input("Databricks Token    : ").strip()
        llm_model   = os.environ.get("LLM_MODEL")           or input("LLM model name      : ").strip()
        embed_model = os.environ.get("EMBED_MODEL")         or input("Embedding model name: ").strip()
        provider = DatabricksProvider(base_url, token, llm_model, embed_model)

    print(f"\n  Provider    : {provider.name}")
    print(f"  LLM model   : {provider.llm_model}")
    print(f"  Embed model : {provider.embed_model}")
    return provider


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Member Feedback Search V2 — query-time expansion + direct embeddings"
    )
    parser.add_argument("--csv",             required=True,           help="Input CSV file path")
    parser.add_argument("--text-col",        default="comment",       help="Column containing comment text")
    parser.add_argument("--id-col",          default=None,            help="Column containing document IDs")
    parser.add_argument("--cache-dir",       default="./embed_cache", help="Directory for embedding cache files")
    parser.add_argument("--fuzzy-threshold", type=int, default=FUZZY_THRESHOLD,
                        help=f"Fuzzy match threshold 0-100 (default {FUZZY_THRESHOLD}). "
                             f"Raise to reduce noise, lower to catch more.")
    parser.add_argument("--min-cosine",  type=float, default=SEMANTIC_MIN_COSINE,
                        help=f"Minimum cosine similarity for semantic layer (default {SEMANTIC_MIN_COSINE}). "
                             f"Raise toward 0.70 to cut noise; lower toward 0.45 for more recall.")
    parser.add_argument("--query",          default=None,
                        help="Run a single query and exit (non-interactive mode)")
    args = parser.parse_args()

    # Load CSV
    print(f"\nLoading '{args.csv}' ...")
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(
            f"Column '{args.text_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)

    id_col = args.id_col if args.id_col and args.id_col in df.columns else None
    documents: List[Dict] = [
        {
            "doc_id": int(df[id_col].iloc[i]) if id_col else i + 1,
            "text":   str(df[args.text_col].iloc[i]),
        }
        for i in range(len(df))
    ]
    print(f"  {len(documents)} comments loaded")

    # Provider
    provider = select_provider()

    # Embedding cache — filename encodes provider + model to avoid stale cache
    cache_dir  = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_tag  = provider.embed_model.replace("/", "_").replace(".", "_").replace("-", "_")
    cache_path = cache_dir / f"{provider.name.lower()}_{model_tag}.npy"
    raw_matrix = build_or_load_embeddings(
        documents, provider, cache_path,
        csv_path=args.csv, text_col=args.text_col,
    )
    # Mean-center to undo gemini-embedding-2 anisotropy, then normalize.
    # Same mean is applied to query vectors inside search().
    query_mean = embedding_mean(raw_matrix)
    doc_matrix = center_normalize(raw_matrix, query_mean)

    print(f"\n  Ready — {len(documents)} comments indexed")
    print(f"  Fuzzy threshold : {args.fuzzy_threshold}  (all hits above this pass through)")
    print(f"  Min cosine      : {args.min_cosine}  (semantic gate — no top-K cap)")
    print(f"  Result count    : all that LLM marks RELEVANT — no ceiling\n")

    if args.query:
        results = search(
            query=args.query, documents=documents, doc_matrix=doc_matrix,
            provider=provider, fuzzy_threshold=args.fuzzy_threshold,
            min_cosine=args.min_cosine, query_mean=query_mean,
        )
        render_results(results, args.query)
        return

    print("Type a search query, or 'quit' to exit.\n")
    while True:
        try:
            raw_query = input("Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not raw_query or raw_query.lower() in ("quit", "exit", "q"):
            break
        results = search(
            query=raw_query, documents=documents, doc_matrix=doc_matrix,
            provider=provider, fuzzy_threshold=args.fuzzy_threshold,
            min_cosine=args.min_cosine, query_mean=query_mean,
        )
        render_results(results, raw_query)


if __name__ == "__main__":
    main()
