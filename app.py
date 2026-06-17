# app_search.py — Streamlit search UI for the search_v2 pipeline
#
# Run:
#   streamlit run app_search.py
#
# Prerequisites:
#   1. Run ingest_local.py first to build the embedding cache
#   2. Enter your Gemini API key in the sidebar
#
# pip install streamlit google-generativeai rapidfuzz pandas numpy

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from provider_ollama_gemini import OllamaGeminiProvider
from search_v2 import (
    FUZZY_THRESHOLD, SEMANTIC_MIN_COSINE,
    center_normalize, embedding_mean,
    fuzzy_search, generate_expansions,
    llm_rerank, rrf_merge, semantic_search,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Member Feedback Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    # On Streamlit Community Cloud the key is embedded as a service token via
    # app secrets (Settings -> Secrets -> GEMINI_API_KEY="..."). Reading
    # st.secrets raises if no secrets are configured (e.g. local dev), so we
    # fall back to the env var + manual entry — the original setup.
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("API key loaded from app secrets.", icon="🔑")
    except Exception:
        api_key = st.text_input(
            "Gemini API key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help=(
                "Used for query expansion, reranking (gemma-4-31b-it) and "
                "query embedding (gemini-embedding-2) — all via Google AI."
            ),
        )

    st.divider()

    cache_dir = st.text_input("Cache directory", value="./embed_cache")

    fuzzy_threshold = st.slider(
        "Fuzzy match threshold",
        min_value=40, max_value=95, value=FUZZY_THRESHOLD, step=5,
        help="Raise to reduce noise; lower to catch more term-variant matches.",
    )
    min_cosine = st.slider(
        "Min cosine similarity",
        min_value=0.30, max_value=0.85,
        value=float(SEMANTIC_MIN_COSINE), step=0.05, format="%.2f",
        help="Semantic gate — all docs above this pass to LLM rerank.",
    )

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_provider(key: str) -> Optional[OllamaGeminiProvider]:
    if not key:
        return None
    try:
        return OllamaGeminiProvider(key)
    except SystemExit as e:
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Provider initialisation failed: {e}")
        return None


@st.cache_resource
def load_index(cache_dir_str: str, want_model: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[Dict]], Dict]:
    """
    Load the pre-computed embedding matrix and document list.

    Selects the cache whose .meta.json embed_model matches `want_model`, so the
    doc vectors share the query's model/dimension. Selecting by file mtime is
    unreliable on Streamlit Cloud — git clone gives every file the same
    timestamp, which can pick a stale cache of a different dimension and crash
    the cosine step.

    Returns (centered_matrix, query_mean, documents, meta). The matrix is
    mean-centered + normalized; query_mean must be applied to query vectors.
    """
    d = Path(cache_dir_str)
    candidates = []
    for f in d.glob("*.npy"):
        mp = f.with_suffix(".meta.json")
        if mp.exists():
            try:
                m = json.load(open(mp, encoding="utf-8"))
            except Exception:
                m = {}
            candidates.append((f, m))
    if not candidates:
        return None, None, None, {}

    # Prefer the cache built with the same embedding model the app queries with.
    matched = [(f, m) for f, m in candidates if m.get("embed_model") == want_model]
    npy_path, meta = (matched[0] if matched
                      else max(candidates, key=lambda fm: fm[0].stat().st_mtime))

    docs_path = d / "documents.json"
    if not docs_path.exists():
        return None, None, None, {}

    raw        = np.load(str(npy_path)).astype(np.float32)
    query_mean = embedding_mean(raw)
    matrix     = center_normalize(raw, query_mean)
    documents  = json.load(open(docs_path, encoding="utf-8"))
    return matrix, query_mean, documents, meta


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("Member Feedback Search")
st.caption("Fuzzy + semantic search with auto-qualify and LLM reranking — powered by Google AI")

# ── Guard: API key ────────────────────────────────────────────────────────────

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to get started.")
    st.stop()

provider = get_provider(api_key)
if provider is None:
    st.stop()

# ── Guard: index ──────────────────────────────────────────────────────────────

doc_matrix, query_mean, documents, meta = load_index(cache_dir, OllamaGeminiProvider.embed_model)

if doc_matrix is None:
    st.error(
        f"No embedding cache found in **{cache_dir}**.\n\n"
        "Build the cache first by running:\n"
        "```\npython ingest_local.py --csv your_data.csv --text-col comment\n```"
    )
    st.stop()

# Safety net: the loaded cache must be the model the app queries with (free check,
# no API call). Catches the case where only a stale/wrong-model cache is present.
if meta.get("embed_model") != OllamaGeminiProvider.embed_model:
    st.error(
        f"No `{OllamaGeminiProvider.embed_model}` cache found in **{cache_dir}**. "
        f"Loaded cache reports `{meta.get('embed_model', '?')}` "
        f"({doc_matrix.shape[1]}-dim). Commit the gemini cache or remove stale `.npy` files."
    )
    st.stop()

# Show index stats in sidebar
with st.sidebar:
    st.divider()
    st.caption(f"**{len(documents)}** comments indexed")
    st.caption(f"Embed model: `{meta.get('embed_model', '?')}`")
    st.caption(f"Dimensions: {doc_matrix.shape[1]}")
    if st.button("Reload index"):
        st.cache_resource.clear()
        st.rerun()

# ── Search input ──────────────────────────────────────────────────────────────

query = st.text_input(
    "Search query",
    placeholder="prior auth   ·   billing error   ·   telehealth   ·   copay",
)

if not query.strip():
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────

with st.status("Running search pipeline …", expanded=True) as status:

    st.write("**[1/3]** Expanding query (equivalent terms + semantic variants, one LLM call) …")
    equivalents, variants = generate_expansions(provider, query)

    st.write("**[2/3]** Running fuzzy + semantic retrieval …")
    fuzzy_res = fuzzy_search(equivalents, documents, fuzzy_threshold)
    semantic_res, best_cos = semantic_search(variants, provider, doc_matrix, documents,
                                             min_cosine, query_mean=query_mean)
    merged, counts = rrf_merge(fuzzy_res, semantic_res,
                               best_cosines=best_cos, documents=documents)

    st.write(
        f"**[3/3]** Reranking **{counts['total']}** candidates "
        f"(fuzzy {counts['fuzzy']} + semantic {counts['semantic']}) — "
        f"auto-qualify first, LLM for the rest …"
    )
    final = llm_rerank(merged, query, provider)

    status.update(
        label=f"Done — **{len(final)}** relevant comment(s) found",
        state="complete",
        expanded=False,
    )

# ── Pipeline summary ──────────────────────────────────────────────────────────

with st.expander("Pipeline details", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fuzzy hits",    counts["fuzzy"])
    c2.metric("Semantic hits", counts["semantic"])
    c3.metric("Both layers",   counts["both_layers"])
    c4.metric("→ LLM",        counts["total"])
    c5.metric("Relevant",      len(final))

    st.divider()
    st.caption("**Equivalent terms:** " + "  ·  ".join(equivalents))
    st.caption("**Semantic variants:**")
    for v in variants:
        st.caption(f"  · {v}")

# ── Results ───────────────────────────────────────────────────────────────────

st.divider()

if not final:
    st.warning(f"No relevant comments found for: **{query}**")
    st.stop()

# Header row: count + export button
col_hdr, col_export = st.columns([3, 1])
with col_hdr:
    st.subheader(f"{len(final)} Relevant Comment(s)  —  *{query}*")
with col_export:
    import pandas as pd
    df_export = pd.DataFrame([
        {"rank": i, "doc_id": r["doc_id"], "comment": r["text"]}
        for i, r in enumerate(final, 1)
    ])
    st.download_button(
        label="⬇ Export CSV",
        data=df_export.to_csv(index=False, encoding="utf-8"),
        file_name=f"results_{query[:40].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

for i, r in enumerate(final, 1):
    with st.container(border=True):
        cosine = r.get("semantic_score", 0.0)
        tag = f" ✦ {r['qualified_by']}" if r.get("qualified_by") else ""
        st.markdown(f"**#{i}** &nbsp; `cosine {cosine:.2f}`{tag}")
        st.write(r["text"])
