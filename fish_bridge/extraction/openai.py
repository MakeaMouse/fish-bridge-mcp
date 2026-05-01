"""OpenAIBackend — OpenAI gpt-4.1-mini extraction via json_schema response format."""
from __future__ import annotations

import json
from typing import Any

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.extraction.prompts import (
    EXTRACTION_OUTPUT_SCHEMA,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)

_MAX_RETRIES = 2


class OpenAIBackend(AbstractExtractionBackend):
    """Uses OpenAI's json_schema response_format for schema-enforced structured output.

    Also works as an OpenAI-compatible backend (LM Studio, Groq, Together, etc.)
    when a custom base_url is provided.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        import openai as _openai
        import os

        api_key = os.environ.get(api_key_env, "not-needed" if base_url else "")
        if not api_key and not base_url:
            raise EnvironmentError(
                f"Environment variable {api_key_env!r} is not set. "
                "Set it to your OpenAI API key before using the OpenAI backend."
            )
        kwargs: dict[str, Any] = {"api_key": api_key or "not-needed"}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = _openai.OpenAI(**kwargs)
        self._model = model
        self._max_tokens = max_tokens

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
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extract_graph",
                            "strict": True,
                            "schema": EXTRACTION_OUTPUT_SCHEMA,
                        },
                    },
                )
                content = response.choices[0].message.content or ""
                return self._parse_response(content)
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"OpenAI extraction failed after {_MAX_RETRIES} retries: {exc}"
                    ) from exc
                # On retry, fall back to plain JSON mode
                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        max_tokens=self._max_tokens,
                        messages=[
                            {"role": "system", "content": EXTRACTION_SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                    return self._parse_response(content)
                except Exception:
                    continue

        return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # Parse helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"nodes": [], "edges": []}
