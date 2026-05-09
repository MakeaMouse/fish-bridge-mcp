"""ClaudeBackend — Anthropic claude-sonnet-4-6 extraction via tool_use schema enforcement."""
from __future__ import annotations

import json
from typing import Any

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.extraction.prompts import (
    EXTRACTION_OUTPUT_SCHEMA,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)

_TOOL_NAME = "extract_graph"
_MAX_RETRIES = 2


class ClaudeBackend(AbstractExtractionBackend):
    """Uses Anthropic's tool_use API for schema-enforced structured output.

    The full JSON Schema is passed as the tool's input_schema, so Claude must
    return valid structured data — no post-hoc JSON parsing required.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-7",  # matches config.py MODEL_DEFAULTS["claude"]
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 2048,
    ) -> None:
        import anthropic as _anthropic
        import os

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env!r} is not set. "
                "Set it to your Anthropic API key before using the Claude backend."
            )
        self._client = _anthropic.Anthropic(api_key=api_key)
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

        tool_def = {
            "name": _TOOL_NAME,
            "description": "Extract a knowledge graph from the provided AI exchange.",
            "input_schema": EXTRACTION_OUTPUT_SCHEMA,
        }

        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=EXTRACTION_SYSTEM,
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": _TOOL_NAME},
                    messages=[{"role": "user", "content": prompt}],
                )
                # Find tool_use block
                for block in response.content:
                    if block.type == "tool_use" and block.name == _TOOL_NAME:
                        return block.input  # already a dict
                # Fallback: try parsing text content
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._parse_json_fallback(block.text)
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(f"Claude extraction failed after {_MAX_RETRIES} retries: {exc}") from exc

        return {"nodes": [], "edges": []}

    # ------------------------------------------------------------------
    # JSON fallback (should rarely be needed with tool_use)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_fallback(text: str) -> dict[str, Any]:
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"nodes": [], "edges": []}
