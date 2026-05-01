"""Abstract extraction backend + post-extraction quality pipeline.

Pipeline steps implemented here:
  [1]  Chunk turn if > MAX_TURN_TOKENS chars (split by paragraph)
  [3]  Pydantic schema validation
  [3a] Grounding check — phantom entity prevention
  [4]  Dual-signal confidence scoring
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from fish_bridge.graph.schema import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    NodeStatus,
    NodeType,
    RawTurn,
)

MAX_TURN_CHARS = 6000   # ~1500 tokens; chunk turns longer than this
UNCONFIRMED_CONFIDENCE_THRESHOLD = 0.50


class AbstractExtractionBackend(ABC):
    """Base class all extraction backends must implement."""

    @abstractmethod
    def _call_llm(self, user_message: str, assistant_message: str) -> dict[str, Any]:
        """Call the LLM and return raw parsed JSON dict with 'nodes' and 'edges'."""
        ...

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def extract(self, turn: RawTurn, exclude_patterns: list[str] | None = None) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Full extraction pipeline for one RawTurn."""
        user_text = self._mask(turn.role_user, exclude_patterns)
        asst_text = self._mask(turn.role_assistant, exclude_patterns)

        # [1] Chunk if too long
        if len(user_text) + len(asst_text) > MAX_TURN_CHARS:
            return self._extract_chunked(user_text, asst_text, turn.session_id)

        return self._extract_single(user_text, asst_text, turn.session_id)

    # ------------------------------------------------------------------
    # Chunked extraction
    # ------------------------------------------------------------------

    def _extract_chunked(
        self, user_text: str, asst_text: str, session_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Split on paragraphs and extract each chunk independently, then merge."""
        chunks = self._split_paragraphs(asst_text, MAX_TURN_CHARS)
        all_nodes: list[GraphNode] = []
        all_edges: list[GraphEdge] = []
        source_text = user_text + "\n" + asst_text

        for chunk in chunks:
            nodes, edges = self._extract_single(user_text, chunk, session_id)
            all_nodes.extend(nodes)
            all_edges.extend(edges)

        return all_nodes, all_edges

    @staticmethod
    def _split_paragraphs(text: str, max_len: int) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > max_len and current:
                chunks.append(current.strip())
                current = para
            else:
                current = (current + "\n\n" + para).strip()
        if current:
            chunks.append(current)
        return chunks or [text[:max_len]]

    # ------------------------------------------------------------------
    # Single-chunk extraction
    # ------------------------------------------------------------------

    def _extract_single(
        self, user_text: str, asst_text: str, session_id: str
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        source_text = user_text + "\n" + asst_text

        raw = self._call_llm(user_text, asst_text)
        nodes_raw: list[dict] = raw.get("nodes", [])
        edges_raw: list[dict] = raw.get("edges", [])

        # [3] Schema validation → GraphNode objects
        nodes = self._validate_nodes(nodes_raw)

        # [3a] Grounding check
        nodes = self.grounding_check(nodes, source_text)

        # [4] Dual-signal confidence
        for node in nodes:
            node.confidence = self._compute_confidence(node, source_text)
            if node.confidence < UNCONFIRMED_CONFIDENCE_THRESHOLD:
                node.status = NodeStatus.UNCONFIRMED

        # Build label→id map for edge resolution
        label_to_id = {n.label.lower(): n.id for n in nodes}

        # Validate edges + resolve from_label/to_label → UUIDs
        edges = self._validate_edges(edges_raw, label_to_id)

        return nodes, edges

    # ------------------------------------------------------------------
    # [3] Schema validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_nodes(raw: list[dict]) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        for item in raw:
            try:
                node_type_str = item.get("type", "concept")
                try:
                    node_type = NodeType(node_type_str)
                except ValueError:
                    node_type = NodeType.CONCEPT

                status_str = item.get("status", "active")
                try:
                    status = NodeStatus(status_str)
                except ValueError:
                    status = NodeStatus.ACTIVE

                node = GraphNode(
                    type=node_type,
                    label=(item.get("label") or "unlabelled")[:120],
                    summary=(item.get("summary") or "")[:500],
                    status=status,
                    confidence=float(item.get("confidence", 1.0)),
                    subtype=item.get("subtype"),
                    source_url=item.get("source_url"),
                    metadata=item.get("metadata") or {},
                )
                nodes.append(node)
            except Exception:
                continue
        return nodes

    @staticmethod
    def _validate_edges(raw: list[dict], label_to_id: dict[str, str]) -> list[GraphEdge]:
        edges: list[GraphEdge] = []
        for item in raw:
            from_label = (item.get("from_label") or "").lower()
            to_label   = (item.get("to_label") or "").lower()
            from_id = label_to_id.get(from_label)
            to_id   = label_to_id.get(to_label)
            if not from_id or not to_id:
                continue  # phantom edge — skip
            relation_str = item.get("relation", "relates-to")
            try:
                relation = EdgeRelation(relation_str)
            except ValueError:
                relation = EdgeRelation.RELATES_TO
            edges.append(
                GraphEdge(
                    from_id=from_id,
                    to_id=to_id,
                    relation=relation,
                    weight=float(item.get("weight", 1.0)),
                )
            )
        return edges

    # ------------------------------------------------------------------
    # [3a] Grounding check
    # ------------------------------------------------------------------

    @staticmethod
    def grounding_check(nodes: list[GraphNode], source_text: str) -> list[GraphNode]:
        """Mark nodes as UNCONFIRMED if their label words are not in the source text.

        Uses the GraphRAG v3.0.6 approach: require at least max(2, round(n * 0.60))
        words from the label to appear in the source text (case-insensitive).
        """
        source_lower = source_text.lower()
        grounded: list[GraphNode] = []

        for node in nodes:
            words = re.findall(r"\w+", node.label.lower())
            n = len(words)
            min_hits = max(2, round(n * 0.60)) if n >= 2 else 1
            hits = sum(1 for w in words if w in source_lower)
            if hits < min_hits:
                node.status = NodeStatus.UNCONFIRMED
                node.confidence = min(node.confidence, 0.3)
            grounded.append(node)

        return grounded

    # ------------------------------------------------------------------
    # [4] Dual-signal confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(node: GraphNode, source_text: str) -> float:
        """Blend structural grounding ratio with LLM self-reported confidence."""
        words = re.findall(r"\w+", node.label.lower())
        if not words:
            return 0.0
        source_lower = source_text.lower()
        grounding_ratio = sum(1 for w in words if w in source_lower) / len(words)
        structural_score = min(1.0, grounding_ratio * 1.2)  # slight boost
        return round((0.6 * structural_score) + (0.4 * node.confidence), 3)

    # ------------------------------------------------------------------
    # PII / secret masking
    # ------------------------------------------------------------------

    @staticmethod
    def _mask(text: str, patterns: list[str] | None) -> str:
        """Apply regex masking patterns to strip PII/secrets before extraction."""
        if not patterns:
            return text
        for pattern in patterns:
            try:
                text = re.sub(pattern, "[REDACTED]", text)
            except re.error:
                pass
        return text
