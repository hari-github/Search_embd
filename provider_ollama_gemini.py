# provider_ollama_gemini.py — Google Generative AI for both embedding and LLM
#
# Embedding : Google Generative AI — gemini-embedding-2
# LLM       : Google Generative AI — gemma-4-31b-it
# API key   : https://aistudio.google.com/apikey
#
# No local ollama required — everything goes through the Google AI API.
#
# Rate limits:
#   2 s sleep before every LLM call.
#   10 s wait between retries on 500 errors.
#
# Install:
#   pip install google-generativeai

import time
from typing import List

import google.generativeai as genai


class OllamaGeminiProvider:
    """
    Pure Google AI provider: Gemini embedding + Gemma LLM.
    Class name kept for import compatibility; ollama is no longer used.
    """

    name        = "GeminiGemma"
    embed_model = "gemini-embedding-2"
    llm_model   = "gemma-4-31b-it"

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self._api_key = api_key
        print(f"  GeminiGemma provider ready")
        print(f"  Embed : {self.embed_model}  (Google Generative AI)")
        print(f"  LLM   : {self.llm_model}  (Google Generative AI)")

    def embed_text(self, text: str, task_type: str = "") -> List[float]:
        """Embed text using Google's gemini-embedding-2 model."""
        result = genai.embed_content(
            model=self.embed_model,
            content=text,
            task_type=task_type or "retrieval_document",
        )
        return result["embedding"]

    def embed_texts(self, texts: List[str], task_type: str = "") -> List[List[float]]:
        """Embed multiple texts in one API call (content accepts a list)."""
        result = genai.embed_content(
            model=self.embed_model,
            content=texts,
            task_type=task_type or "retrieval_document",
        )
        emb = result["embedding"]
        # A single-item request can come back as one flat vector
        if emb and isinstance(emb[0], (int, float)):
            return [emb]
        return emb

    def llm_call(self, system: str, user: str) -> str:
        """
        Call Gemma via Google Generative AI.
        - 2 s sleep before every call to respect rate limits.
        - Up to 3 attempts with a 2 s wait between retries on errors.
        """
        time.sleep(2)
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(
                    model_name=self.llm_model,
                    system_instruction=system,
                )
                response = model.generate_content(user)
                return response.text
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    print(
                        f"  [WARN] LLM call failed "
                        f"(attempt {attempt + 1}/3 — {type(exc).__name__}). "
                        f"Retrying in 2 s ..."
                    )
                    time.sleep(2)
        raise last_exc
