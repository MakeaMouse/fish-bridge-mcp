"""HybridBackend — local backend for real-time, cloud for consolidation.

Alternates between two backends per turn:
  - realtime_backend (e.g. "local"): used for every turn — fast, zero cost
  - consolidation_backend (e.g. "gemini"): used every N turns — higher quality

This gives the best of both worlds: low latency for continuous ingest during
a live session, with periodic high-quality re-extraction to catch nodes the
local model may have missed or mislabelled.

Config example:
  extraction:
    backend: "hybrid"
    hybrid:
      realtime_backend: "local"
      consolidation_backend: "gemini"
      consolidation_every_n: 10
"""
from __future__ import annotations

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.graph.schema import GraphEdge, GraphNode, RawTurn


class HybridBackend(AbstractExtractionBackend):
    """Wraps two backends: local for real-time, cloud for consolidation."""

    def __init__(
        self,
        realtime_backend:      AbstractExtractionBackend,
        consolidation_backend: AbstractExtractionBackend,
        consolidation_every_n: int = 10,
    ) -> None:
        self._realtime      = realtime_backend
        self._consolidation = consolidation_backend
        self._every_n       = max(1, consolidation_every_n)
        self._turn_counter  = 0

    # ------------------------------------------------------------------
    # AbstractExtractionBackend
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str, assistant_message: str):  # type: ignore[override]
        # Not used directly — extract() is overridden below
        raise NotImplementedError("HybridBackend delegates to sub-backends.")

    def extract(
        self,
        turn: RawTurn,
        exclude_patterns: list[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Route turn to realtime or consolidation backend based on turn counter."""
        self._turn_counter += 1

        if self._turn_counter % self._every_n == 0:
            # Consolidation turn: use higher-quality backend
            try:
                return self._consolidation.extract(turn, exclude_patterns)
            except Exception:
                # Fall back to realtime if consolidation fails (e.g. no API key)
                return self._realtime.extract(turn, exclude_patterns)
        else:
            return self._realtime.extract(turn, exclude_patterns)

    @property
    def turn_counter(self) -> int:
        return self._turn_counter

    def reset_counter(self) -> None:
        self._turn_counter = 0
