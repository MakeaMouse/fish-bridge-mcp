"""Abstract extraction backend + post-extraction quality pipeline.

Pipeline steps implemented here:
  [0.5] Content-zone pre-processor (code blocks, stack traces, URLs, file refs)
  [1]  Chunk turn if > MAX_TURN_TOKENS chars (split by paragraph)
  [3]  Pydantic schema validation
  [3a] Grounding check — phantom entity prevention
  [4]  Dual-signal confidence scoring
"""
from __future__ import annotations

import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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

# ---------------------------------------------------------------------------
# Cloud backend PII warning
# ---------------------------------------------------------------------------

def warn_cloud_backend_once(backend_name: str) -> None:
    """Print a one-time warning when a cloud extraction backend is first used.

    The sentinel file prevents the warning from repeating on every run.
    Users are informed they can switch to a local backend for offline/private mode.
    """
    _env_cfg = os.environ.get("FISH_BRIDGE_CONFIG")
    sentinel = (
        Path(_env_cfg).parent / ".cloud_backend_warned"
        if _env_cfg
        else Path.home() / ".fish_bridge" / ".cloud_backend_warned"
    )
    if sentinel.exists():
        return
    print(
        f"\n⚠  fish_bridge: cloud backend '{backend_name}' is active.\n"
        "   Turn text is sent to the provider's API for graph extraction.\n"
        "   Ensure no passwords, API keys, or private credentials are in your chat turns.\n"
        "   Default PII patterns are masked, but regex masking is not guaranteed to be complete.\n"
        "   For fully offline operation: set 'extraction.backend: local' in config.yaml\n"
        "   and run Ollama locally (https://ollama.com).\n"
        "   (This warning appears once. See docs/configuration.md#privacy for details.)\n",
        file=sys.stderr,
    )
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
    except OSError:
        pass  # failure to write sentinel is non-fatal


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

        # [0.5] Content-zone pre-processor
        try:
            from fish_bridge.extraction.preprocessor import preprocess
            hints = preprocess(user_text, asst_text)
            if not hints.is_empty():
                hints_section = hints.to_prompt_section()
                user_text = hints_section + user_text
                # Attach URL candidates to turn metadata for later source_url assignment
                turn._hints = hints
        except Exception:
            pass  # pre-processor failure must never block extraction

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

        # [3b] Speculative entity check (Fix 4)
        nodes = self.speculative_check(nodes, source_text)

        # [4] Dual-signal confidence
        for node in nodes:
            node.confidence = self._compute_confidence(node, source_text)
            if node.confidence < UNCONFIRMED_CONFIDENCE_THRESHOLD:
                node.status = NodeStatus.UNCONFIRMED

        # Build label→id map for edge resolution
        label_to_id = {n.label.lower(): n.id for n in nodes}

        # Validate edges + resolve from_label/to_label → UUIDs
        edges = self._validate_edges(edges_raw, label_to_id)

        # [4b] Fallback edge density enforcement (Fix 1 insurance net)
        edges = self._ensure_minimum_edges(nodes, edges)

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
    # [3b] Speculative entity check (Fix 4)
    # ------------------------------------------------------------------

    # Regex patterns that signal a speculative / not-yet-verified claim
    _SPECULATIVE_PHRASES: re.Pattern = re.compile(
        r"\b(might|could|maybe|perhaps|possibly|future version|upcoming|not yet|"
        r"not released|hypothetically|we could|planning to|would use|plan to use|"
        r"when it releases|once it's out|when available)\b",
        re.IGNORECASE,
    )

    # Model name patterns that are plausibly fabricated versions
    # Matches things like "gpt-5.5", "gpt-6", "claude-5", "gemini-3.0"
    _PHANTOM_MODEL_RE: re.Pattern = re.compile(
        r"\b(gpt-[5-9][\.\d]*(?![\w-])|gpt-[1-9]\d[\.\d]*|"
        r"claude-[5-9][\.\d]*(?![\w-])|"
        r"gemini-[3-9][\.\d]*(?![\w-]))\b",
        re.IGNORECASE,
    )

    @classmethod
    def speculative_check(
        cls, nodes: list[GraphNode], source_text: str
    ) -> list[GraphNode]:
        """Downgrade nodes that describe speculative or unverified entities.

        Two triggers:
        1. The source text around the node label contains speculative language
           ("might use", "future version", etc.)
        2. The node label contains a model name version that looks fabricated
           (e.g. "gpt-5.5", "claude-5") — these are not released models.

        In both cases: confidence is capped at 0.45 and metadata["speculative"] = True.
        The confidence cap then triggers the UNCONFIRMED_CONFIDENCE_THRESHOLD check
        in the calling code, setting status = UNCONFIRMED automatically.
        """
        source_lower = source_text.lower()

        for node in nodes:
            label_lower = node.label.lower()

            # Check for phantom model name in label
            if cls._PHANTOM_MODEL_RE.search(node.label):
                node.confidence = min(node.confidence, 0.40)
                node.metadata["speculative"] = True
                node.metadata["speculative_reason"] = "model_name_unverified"
                continue

            # Find the window of text around where this label appears
            idx = source_lower.find(label_lower[:20])  # first 20 chars as anchor
            if idx == -1:
                continue  # already handled by grounding_check
            window = source_text[max(0, idx - 120): idx + 200]

            if cls._SPECULATIVE_PHRASES.search(window):
                node.confidence = min(node.confidence, 0.45)
                node.metadata["speculative"] = True
                node.metadata["speculative_reason"] = "speculative_language_in_context"

        return nodes

    # ------------------------------------------------------------------
    # [4b] Fallback edge density enforcement (Fix 1 insurance net)
    # ------------------------------------------------------------------

    # Type-compatible edge rules: (from_type, to_type) → preferred relation.
    # Used only as a last resort when a node has zero edges after LLM extraction.
    _TYPE_EDGE_DEFAULTS: dict[tuple[str, str], str] = {
        ("task",     "question"):  "leads-to",
        ("task",     "error"):     "resolves",
        ("task",     "decision"):  "implements",
        ("task",     "file"):      "references",
        ("task",     "skill"):     "uses",
        ("error",    "file"):      "created-by",
        ("error",    "task"):      "blocks",
        ("decision", "concept"):   "documents",
        ("decision", "skill"):     "uses",
        ("question", "concept"):   "references",
        ("question", "decision"):  "leads-to",
        ("skill",    "concept"):   "relates-to",
        ("concept",  "skill"):     "relates-to",
        ("file",     "decision"):  "implements",
        ("file",     "skill"):     "uses",
    }

    @classmethod
    def _ensure_minimum_edges(
        cls,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> list[GraphEdge]:
        """Add fallback edges for any node that has zero edges after LLM extraction.

        This is a safety net — it should rarely fire if the updated prompts work.
        Only generates a single "relates-to" edge to the nearest compatible node
        rather than trying to infer precise semantics without context.

        Works entirely offline: no LLM call, no embedding. Safe for all backends
        including low-resource environments (no GPU / no API key required).
        """
        if len(nodes) < 2:
            return edges  # can't add edges with only 1 node

        # Build connectivity set
        connected_ids: set[str] = set()
        for e in edges:
            connected_ids.add(e.from_id)
            connected_ids.add(e.to_id)

        new_edges: list[GraphEdge] = []

        for node in nodes:
            if node.id in connected_ids:
                continue  # already connected

            # Find first compatible partner by type-priority rules
            partner: GraphNode | None = None
            best_relation = EdgeRelation.RELATES_TO
            node_type_str = node.type if isinstance(node.type, str) else node.type.value

            for other in nodes:
                if other.id == node.id:
                    continue
                other_type_str = other.type if isinstance(other.type, str) else other.type.value
                key = (node_type_str, other_type_str)
                if key in cls._TYPE_EDGE_DEFAULTS:
                    partner = other
                    best_relation = EdgeRelation(cls._TYPE_EDGE_DEFAULTS[key])
                    break

            # Fall back to first other node if no typed rule matched
            if partner is None:
                for other in nodes:
                    if other.id != node.id:
                        partner = other
                        break

            if partner is not None:
                new_edges.append(GraphEdge(
                    from_id=node.id,
                    to_id=partner.id,
                    relation=best_relation,
                    weight=0.5,  # lower weight signals this was auto-generated
                ))
                connected_ids.add(node.id)
                connected_ids.add(partner.id)

        return edges + new_edges

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
