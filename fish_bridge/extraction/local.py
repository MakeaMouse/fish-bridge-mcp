"""OllamaBackend — local Ollama extraction with format= JSON Schema enforcement.

Requires Ollama v0.5+ running at http://localhost:11434 (or configured base_url).
Uses `format=` parameter to enforce JSON Schema at the sampler level — no post-hoc
parsing or retry needed for schema compliance.

Recommended models (ordered by quality/speed tradeoff):
  qwen2.5:7b    — 85% quality, ~200ms/turn, best default
  gemma3:12b    — 91% quality, ~500ms/turn, if hardware allows
  mistral:7b    — 82% quality, fast on CPU

Embedding:
  nomic-embed-text via /api/embed — used by semantic dedup in EmbeddingProvider.
  Must be pulled separately: ollama pull nomic-embed-text
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.extraction.prompts import (
    EXTRACTION_OUTPUT_SCHEMA,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)

_MAX_RETRIES = 2


class OllamaBackend(AbstractExtractionBackend):
    """Ollama-based extraction backend using format= JSON Schema enforcement."""

    def __init__(
        self,
        model:       str = "qwen2.5:7b",
        base_url:    str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        timeout:     float = 120.0,
    ) -> None:
        self._model       = model
        self._base_url    = base_url.rstrip("/")
        self._embed_model = embed_model
        self._timeout     = timeout

    # ------------------------------------------------------------------
    # AbstractExtractionBackend
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, assistant_message: str) -> dict[str, Any]:
        prompt = EXTRACTION_USER_TEMPLATE.format(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._chat_with_schema(prompt)
            except (json.JSONDecodeError, KeyError) as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Ollama extraction failed after {_MAX_RETRIES} retries: {exc}"
                    ) from exc
                # Retry without schema enforcement (plain JSON mode fallback)
                try:
                    return self._chat_json_mode(prompt)
                except Exception:
                    continue

        return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat_with_schema(self, prompt: str) -> dict[str, Any]:
        """Use format= JSON Schema enforcement (Ollama v0.5+)."""
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model":  self._model,
                "stream": False,
                "format": EXTRACTION_OUTPUT_SCHEMA,
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)

    def _chat_json_mode(self, prompt: str) -> dict[str, Any]:
        """Fallback: plain JSON mode without schema enforcement."""
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json={
                "model":  self._model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return json.loads(content)

    # ------------------------------------------------------------------
    # Embedding (used by EmbeddingProvider in dedup.py)
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float] | None:
        """Embed text using nomic-embed-text.  Returns None if unavailable."""
        try:
            resp = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": self._embed_model, "input": text},
                timeout=10.0,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is present."""
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            # Accept prefix match: "qwen2.5:7b" matches "qwen2.5:7b" or "qwen2.5:7b-instruct"
            model_prefix = self._model.split(":")[0]
            return any(m.startswith(model_prefix) for m in models)
        except Exception:
            return False
