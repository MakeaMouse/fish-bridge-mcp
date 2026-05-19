"""Tests for orphan fallback edge sync, relates-to upgrade, orphan healing, and skill-concept edges."""
from __future__ import annotations
import pytest
from pathlib import Path

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeStatus, NodeType
from fish_bridge.graph.session import SessionGraph, _ORPHAN_FALLBACK_EDGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(label: str, ntype: str, status: str = "active") -> GraphNode:
    return GraphNode(type=NodeType(ntype), label=label, status=NodeStatus(status))


def _edge(from_id: str, to_id: str, rel: str, weight: float = 1.0) -> GraphEdge:
    return GraphEdge(from_id=from_id, to_id=to_id, relation=EdgeRelation(rel), weight=weight)


@pytest.fixture
def sg(tmp_path: Path) -> SessionGraph:
    s = SessionGraph.open("test", tmp_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Fix K: _ORPHAN_FALLBACK_EDGES completeness
# ---------------------------------------------------------------------------

class TestOrphanFallbackEdgesFixK:
    """_ORPHAN_FALLBACK_EDGES must contain all pairs added by Fix J to base.py."""

    def test_task_task_maps_to_depends_on(self):
        assert _ORPHAN_FALLBACK_EDGES.get(("task", "task")) == "depends-on"

    def test_decision_task_maps_to_implements(self):
        assert _ORPHAN_FALLBACK_EDGES.get(("decision", "task")) == "implements"

    def test_concept_task_maps_to_references(self):
        assert _ORPHAN_FALLBACK_EDGES.get(("concept", "task")) == "references"

    def test_concept_question_maps_to_references(self):
        assert _ORPHAN_FALLBACK_EDGES.get(("concept", "question")) == "references"

    def test_no_relates_to_in_fallback_edges(self):
        """No entry in _ORPHAN_FALLBACK_EDGES should map to 'relates-to'."""
        for (ft, tt), rel in _ORPHAN_FALLBACK_EDGES.items():
            assert rel != "relates-to", (
                f"({ft!r}, {tt!r}) still maps to 'relates-to'"
            )


# ---------------------------------------------------------------------------
# Fix L: upgrade_relates_to_edges
# ---------------------------------------------------------------------------

class TestUpgradeRelatesToEdges:
    """upgrade_relates_to_edges upgrades cross-type relates-to to typed relations."""

    def _setup(self, sg: SessionGraph, from_type: str, to_type: str) -> tuple[str, str]:
        a = _node("source node", from_type)
        b = _node("target node", to_type)
        sg.add_node(a); sg.add_node(b)
        sg.add_edge(_edge(a.id, b.id, "relates-to", 1.0))
        return a.id, b.id

    def test_task_decision_upgraded_to_implements(self, sg: SessionGraph):
        a_id, b_id = self._setup(sg, "task", "decision")
        count = sg.upgrade_relates_to_edges()
        assert count == 1
        edges = sg.all_edges()
        assert any(e.from_id == a_id and e.relation == EdgeRelation.IMPLEMENTS for e in edges)

    def test_task_skill_upgraded_to_uses(self, sg: SessionGraph):
        a_id, b_id = self._setup(sg, "task", "skill")
        count = sg.upgrade_relates_to_edges()
        assert count == 1
        edges = sg.all_edges()
        assert any(e.from_id == a_id and e.relation == EdgeRelation.USES for e in edges)

    def test_error_task_upgraded_to_blocks(self, sg: SessionGraph):
        a_id, b_id = self._setup(sg, "error", "task")
        count = sg.upgrade_relates_to_edges()
        assert count == 1
        edges = sg.all_edges()
        assert any(e.from_id == a_id and e.relation == EdgeRelation.BLOCKS for e in edges)

    def test_decision_concept_upgraded_to_documents(self, sg: SessionGraph):
        a_id, b_id = self._setup(sg, "decision", "concept")
        count = sg.upgrade_relates_to_edges()
        assert count == 1
        edges = sg.all_edges()
        assert any(e.from_id == a_id and e.relation == EdgeRelation.DOCUMENTS for e in edges)

    def test_same_type_task_task_not_upgraded(self, sg: SessionGraph):
        """task↔task relates-to must be left untouched."""
        a = _node("task A", "task"); b = _node("task B", "task")
        sg.add_node(a); sg.add_node(b)
        sg.add_edge(_edge(a.id, b.id, "relates-to"))
        count = sg.upgrade_relates_to_edges()
        assert count == 0
        edges = sg.all_edges()
        assert all(e.relation == EdgeRelation.RELATES_TO for e in edges)

    def test_non_rt_edge_not_touched(self, sg: SessionGraph):
        """Already-typed edges must not be modified."""
        a = _node("task", "task"); b = _node("decision", "decision")
        sg.add_node(a); sg.add_node(b)
        sg.add_edge(_edge(a.id, b.id, "implements"))
        count = sg.upgrade_relates_to_edges()
        assert count == 0

    def test_idempotent(self, sg: SessionGraph):
        """Calling upgrade twice should return 0 on the second call."""
        self._setup(sg, "task", "decision")
        sg.upgrade_relates_to_edges()
        second = sg.upgrade_relates_to_edges()
        assert second == 0

    def test_no_rt_in_upgrade_rules(self):
        """_RT_UPGRADE_RULES must not map any pair to 'relates-to'."""
        for (ft, tt), rel in SessionGraph._RT_UPGRADE_RULES.items():
            assert rel != "relates-to", f"({ft!r},{tt!r}) still maps to 'relates-to'"


# ---------------------------------------------------------------------------
# Fix M: heal_orphan_edges (token-overlap path — no embeddings needed)
# ---------------------------------------------------------------------------

class TestHealOrphanEdges:
    """heal_orphan_edges connects isolated nodes using token overlap (no embeddings)."""

    def test_orphan_gets_connected(self, sg: SessionGraph):
        connected = _node("extract nodes from session", "task")
        orphan    = _node("session extraction pipeline", "concept")
        sg.add_node(connected); sg.add_node(orphan)
        # connected has an edge, orphan does not
        partner = _node("ingest turn", "task")
        sg.add_node(partner)
        sg.add_edge(_edge(connected.id, partner.id, "uses"))

        count = sg.heal_orphan_edges()
        assert count >= 1
        edge_node_ids = set()
        for e in sg.all_edges():
            edge_node_ids.add(e.from_id)
            edge_node_ids.add(e.to_id)
        assert orphan.id in edge_node_ids

    def test_no_orphans_returns_zero(self, sg: SessionGraph):
        a = _node("alpha", "task"); b = _node("beta", "decision")
        sg.add_node(a); sg.add_node(b)
        sg.add_edge(_edge(a.id, b.id, "implements"))
        count = sg.heal_orphan_edges()
        assert count == 0

    def test_empty_graph_returns_zero(self, sg: SessionGraph):
        assert sg.heal_orphan_edges() == 0


# ---------------------------------------------------------------------------
# Fix N: infer_cross_session_edges — skill↔concept uses USES not RELATES_TO
# ---------------------------------------------------------------------------

class TestInferCrossSessionEdgesFixN:
    """infer_cross_session_edges must use EdgeRelation.USES for skill↔concept."""

    def test_skill_concept_token_overlap_uses_not_relates_to(self, sg: SessionGraph):
        # Two nodes sharing 2 significant words
        skill   = _node("python async programming skill", "skill")
        concept = _node("python async concurrency concept", "concept")
        sg.add_node(skill); sg.add_node(concept)
        # Add a dummy edge so the session has at least some connectivity
        anchor = _node("anchor", "task")
        sg.add_node(anchor)
        sg.add_edge(_edge(skill.id, anchor.id, "uses"))

        sg.infer_cross_session_edges(min_shared_words=2)
        edges = sg.all_edges()
        skill_concept_edges = [
            e for e in edges
            if {e.from_id, e.to_id} == {skill.id, concept.id}
        ]
        assert skill_concept_edges, "Expected inferred edge between skill and concept"
        for e in skill_concept_edges:
            assert e.relation == EdgeRelation.USES, (
                f"skill↔concept should be USES, got {e.relation}"
            )
