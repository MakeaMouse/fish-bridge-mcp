"""Tests for extraction base: grounding check, confidence, schema validation."""
from __future__ import annotations

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType, RawTurn


# ---------------------------------------------------------------------------
# Concrete stub backend for testing (no LLM call)
# ---------------------------------------------------------------------------

class StubBackend(AbstractExtractionBackend):
    """Returns pre-configured raw dicts; no network call."""

    def __init__(self, raw_response: dict):
        self._response = raw_response

    def _call_llm(self, user_message: str, assistant_message: str) -> dict:
        return self._response


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

class TestGroundingCheck:

    def test_well_grounded_node_stays_active(self):
        node = GraphNode(type=NodeType.DECISION, label="Redis caching strategy", status=NodeStatus.PROPOSED)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We decided to use Redis for caching with a 24hr strategy."
        )
        assert result[0].status != NodeStatus.UNCONFIRMED

    def test_phantom_node_marked_unconfirmed(self):
        node = GraphNode(type=NodeType.CONCEPT, label="quantum entanglement lattice", status=NodeStatus.ACTIVE)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We discussed Redis and SQLite for storage."
        )
        assert NodeStatus(result[0].status) == NodeStatus.UNCONFIRMED

    def test_single_word_label_requires_one_hit(self):
        node = GraphNode(type=NodeType.SKILL, label="Redis", status=NodeStatus.ACTIVE)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We discussed Redis usage."
        )
        assert NodeStatus(result[0].status) != NodeStatus.UNCONFIRMED

    def test_empty_nodes_list(self):
        result = AbstractExtractionBackend.grounding_check([], "Some text.")
        assert result == []


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

class TestMasking:

    def test_pattern_redacts_matching_text(self):
        text = "My API key is sk-ant-abc123 and token is ghp_xyz789"
        masked = AbstractExtractionBackend._mask(text, [r"sk-ant-\w+", r"ghp_\w+"])
        assert "sk-ant-abc123" not in masked
        assert "ghp_xyz789" not in masked
        assert "[REDACTED]" in masked

    def test_invalid_pattern_is_skipped(self):
        text = "hello world"
        # Should not raise; bad regex is silently skipped
        result = AbstractExtractionBackend._mask(text, ["[invalid(regex"])
        assert result == text

    def test_empty_patterns_returns_original(self):
        text = "unchanged text"
        assert AbstractExtractionBackend._mask(text, []) == text


# ---------------------------------------------------------------------------
# Full extraction pipeline (stub LLM)
# ---------------------------------------------------------------------------

class TestExtractionPipeline:

    def _make_turn(self, user: str, asst: str) -> RawTurn:
        return RawTurn(session_id="test", turn_number=1, role_user=user, role_assistant=asst)

    def test_valid_extraction_returns_nodes_and_edges(self):
        raw = {
            "nodes": [
                {"type": "decision", "label": "Use Redis", "summary": "Caching layer", "status": "proposed", "confidence": 0.9},
                {"type": "skill",    "label": "Redis",     "summary": "In-memory cache", "status": "active", "confidence": 0.95},
            ],
            "edges": [
                {"from_label": "Use Redis", "to_label": "Redis", "relation": "uses"},
            ],
        }
        backend = StubBackend(raw)
        turn = self._make_turn(
            "Should we use Redis for caching?",
            "Yes, Redis is a great choice as an in-memory cache. Use Redis for the caching layer.",
        )
        nodes, edges = backend.extract(turn)
        assert any(n.label == "Use Redis" for n in nodes)
        assert any(n.label == "Redis" for n in nodes)
        assert len(edges) == 1
        assert edges[0].relation == "uses"

    def test_phantom_edge_dropped(self):
        """Edge whose endpoints aren't in the node list should be silently dropped."""
        raw = {
            "nodes": [
                {"type": "concept", "label": "SQLite WAL", "summary": "Write-ahead logging", "status": "active", "confidence": 0.9},
            ],
            "edges": [
                {"from_label": "SQLite WAL", "to_label": "NonExistentNode", "relation": "uses"},
            ],
        }
        backend = StubBackend(raw)
        turn = self._make_turn("Tell me about SQLite WAL mode.", "SQLite WAL mode enables concurrent reads.")
        nodes, edges = backend.extract(turn)
        assert len(edges) == 0  # phantom edge dropped

    def test_invalid_node_type_defaults_to_concept(self):
        raw = {
            "nodes": [{"type": "bogustype", "label": "something", "summary": "", "status": "active", "confidence": 0.8}],
            "edges": [],
        }
        backend = StubBackend(raw)
        turn = self._make_turn("Tell me about something.", "Something is important.")
        nodes, _ = backend.extract(turn)
        assert all(n.type == NodeType.CONCEPT for n in nodes)

    def test_empty_llm_response(self):
        backend = StubBackend({"nodes": [], "edges": []})
        turn = self._make_turn("Hello", "Hi")
        nodes, edges = backend.extract(turn)
        assert nodes == []
        assert edges == []
