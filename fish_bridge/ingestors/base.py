"""Abstract ingestor interface and RawTurn re-export."""
from __future__ import annotations

from abc import ABC, abstractmethod

from fish_bridge.graph.schema import RawTurn


class AbstractIngestor(ABC):
    """Base class for all ingestors.  Each ingestor converts a source
    into a list of RawTurn objects ready for extraction.
    """

    @abstractmethod
    def ingest(self, **kwargs) -> list[RawTurn]:
        """Return normalised turns from this source."""
        ...
