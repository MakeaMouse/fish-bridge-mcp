"""SessionFileIngestor — imports a prior .chatgraph.json export.

Prior session nodes/edges are merged directly into the current session
without re-running LLM extraction.  The dedup logic in merge_extraction()
handles near-duplicates across sessions.

Usage:
    fish-bridge import prior-session.chatgraph.json
    fish-bridge merge --source session --file prior.chatgraph.json

The import stamps each node with metadata.original_session_id so the
provenance of cross-session nodes is queryable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fish_bridge.graph.schema import (
    GraphEdge,
    GraphNode,
)
from fish_bridge.ingestors.base import AbstractIngestor

# RawTurn is not used here — session imports are node/edge-level, not turn-level.
# The result is returned via the special load() method.


class SessionFileIngestor(AbstractIngestor):
    """Loads a .chatgraph.json file and returns parsed nodes + edges.

    Unlike other ingestors, this returns (nodes, edges) directly rather than
    RawTurns, since no re-extraction is needed.
    """

    def ingest(self, **kwargs):  # type: ignore[override]
        """Not used — call load() directly."""
        raise NotImplementedError("Use SessionFileIngestor.load() instead of ingest().")

    @staticmethod
    def load(path: Path) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse a .chatgraph.json export file into nodes and edges."""
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return SessionFileIngestor.load_from_dict(raw)

    @staticmethod
    def load_from_dict(raw: dict[str, Any]) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse a pre-loaded dict (from JSON) into nodes and edges.

        Stamps each node with metadata.original_session_id for provenance.
        Returns (nodes, edges) ready to pass to SessionGraph.merge_extraction().
        """
        original_session_id = raw.get("session_id", "unknown")
        raw_nodes: list[dict] = raw.get("nodes", [])
        raw_edges: list[dict] = raw.get("edges", [])

        nodes: list[GraphNode] = []
        for n in raw_nodes:
            try:
                n.pop("embedding", None)
                n.setdefault("metadata", {})["original_session_id"] = original_session_id
                node = GraphNode(**n)
                nodes.append(node)
            except Exception:
                continue

        edges: list[GraphEdge] = []
        node_id_set = {n.id for n in nodes}
        for e in raw_edges:
            try:
                if e.get("from_id") not in node_id_set or e.get("to_id") not in node_id_set:
                    continue
                edge = GraphEdge(**e)
                edges.append(edge)
            except Exception:
                continue

        return nodes, edges

    @staticmethod
    def summary(path: Path) -> dict[str, Any]:
        """Return a brief summary dict for CLI display without full load."""
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return {
            "session_id":     raw.get("session_id", "unknown"),
            "version":        raw.get("fish_bridge_version", raw.get("chatgraph_version", "?")),
            "node_count":     len(raw.get("nodes", [])),
            "edge_count":     len(raw.get("edges", [])),
            "created_at":     raw.get("created_at", "unknown"),
        }
