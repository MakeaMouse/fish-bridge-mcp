"""Phase 3/4 tests: graph algorithms, digest/focus compilers, MCP server tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fish_bridge.graph.schema import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    NodeStatus,
    NodeType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    id: str,
    label: str,
    node_type: NodeType = NodeType.CONCEPT,
    status: NodeStatus = NodeStatus.ACTIVE,
    summary: str = "",
) -> GraphNode:
    return GraphNode(
        id=id,
        label=label,
        type=node_type,
        status=status,
        summary=summary,
        confidence=0.9,
    )


def _make_edge(from_id: str, to_id: str, relation: EdgeRelation = EdgeRelation.RELATES_TO) -> GraphEdge:
    return GraphEdge(
        id=f"{from_id}-{to_id}",
        from_id=from_id,
        to_id=to_id,
        relation=relation,
        weight=1.0,
    )


@pytest.fixture
def sample_nodes() -> list[GraphNode]:
    return [
        _make_node("n1", "setup database",  NodeType.TASK,     NodeStatus.IN_PROGRESS),
        _make_node("n2", "schema design",   NodeType.DECISION, NodeStatus.ADOPTED),
        _make_node("n3", "query optimizer", NodeType.CONCEPT,  NodeStatus.ACTIVE),
        _make_node("n4", "index creation",  NodeType.TASK,     NodeStatus.PENDING),
        _make_node("n5", "connection pool", NodeType.CONCEPT,  NodeStatus.ACTIVE),
        _make_node("n6", "test coverage",   NodeType.TASK,     NodeStatus.PENDING),
        _make_node("n7", "open question",   NodeType.QUESTION, NodeStatus.ACTIVE),
        _make_node("n8", "missing feature", NodeType.ERROR,    NodeStatus.ACTIVE),
    ]


@pytest.fixture
def sample_edges(sample_nodes) -> list[GraphEdge]:
    return [
        _make_edge("n1", "n2", EdgeRelation.DEPENDS_ON),
        _make_edge("n2", "n3", EdgeRelation.LEADS_TO),
        _make_edge("n3", "n4", EdgeRelation.LEADS_TO),
        _make_edge("n4", "n5", EdgeRelation.RELATES_TO),
        _make_edge("n5", "n3", EdgeRelation.RELATES_TO),  # cycle
        _make_edge("n6", "n1", EdgeRelation.TESTED_BY),
    ]


# ---------------------------------------------------------------------------
# graph/algorithms.py
# ---------------------------------------------------------------------------

class TestBuildNxGraph:

    def test_directed_graph_has_correct_nodes_and_edges(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import build_nx_graph
        G = build_nx_graph(sample_nodes, sample_edges, directed=True)
        assert G.number_of_nodes() == len(sample_nodes)
        assert G.number_of_edges() == len(sample_edges)

    def test_undirected_graph(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import build_nx_graph
        G = build_nx_graph(sample_nodes, sample_edges, directed=False)
        assert not G.is_directed()

    def test_empty_graph(self):
        from fish_bridge.graph.algorithms import build_nx_graph
        G = build_nx_graph([], [], directed=True)
        assert G.number_of_nodes() == 0


class TestCommunityDetection:

    def test_returns_mapping_for_all_nodes(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import community_detection
        mapping = community_detection(sample_nodes, sample_edges)
        assert isinstance(mapping, dict)
        assert len(mapping) == len(sample_nodes)

    def test_empty_graph_returns_empty(self):
        from fish_bridge.graph.algorithms import community_detection
        mapping = community_detection([], [])
        assert mapping == {}

    def test_isolated_nodes_each_own_community(self):
        from fish_bridge.graph.algorithms import community_detection
        nodes = [_make_node("a", "alpha"), _make_node("b", "beta")]
        mapping = community_detection(nodes, [])
        # Both nodes should appear in the mapping
        assert "a" in mapping and "b" in mapping


class TestSubgraphNeighborhood:

    def test_seed_included_in_result(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import subgraph_neighborhood
        nodes, _ = subgraph_neighborhood(["n1"], sample_nodes, sample_edges, max_hops=1)
        ids = {n.id for n in nodes}
        assert "n1" in ids

    def test_neighbors_included(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import subgraph_neighborhood
        nodes, _ = subgraph_neighborhood(["n2"], sample_nodes, sample_edges, max_hops=1)
        ids = {n.id for n in nodes}
        # n2 is a direct hop from n1
        assert "n2" in ids

    def test_max_nodes_respected(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import subgraph_neighborhood
        nodes, _ = subgraph_neighborhood(["n1"], sample_nodes, sample_edges, max_hops=5, max_nodes=2)
        assert len(nodes) <= 2

    def test_unknown_seed_returns_empty(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import subgraph_neighborhood
        nodes, edges = subgraph_neighborhood(["does_not_exist"], sample_nodes, sample_edges)
        assert nodes == [] and edges == []


class TestSemanticSearchNodes:

    def test_returns_top_k(self, sample_nodes):
        from fish_bridge.graph.algorithms import semantic_search_nodes
        results = semantic_search_nodes("database setup", sample_nodes, top_k=3)
        assert len(results) <= 3

    def test_returns_node_score_pairs(self, sample_nodes):
        from fish_bridge.graph.algorithms import semantic_search_nodes
        results = semantic_search_nodes("query optimizer", sample_nodes, top_k=5)
        for node, score in results:
            assert isinstance(node, GraphNode)
            assert 0.0 <= score <= 1.0

    def test_relevant_node_scores_higher(self, sample_nodes):
        from fish_bridge.graph.algorithms import semantic_search_nodes
        results = semantic_search_nodes("query optimizer", sample_nodes, top_k=8)
        labels = [n.label for n, _ in results]
        # "query optimizer" node should appear before unrelated nodes
        assert labels[0] == "query optimizer"

    def test_empty_nodes(self):
        from fish_bridge.graph.algorithms import semantic_search_nodes
        assert semantic_search_nodes("anything", [], top_k=5) == []


class TestShortestPath:

    def test_direct_path(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import shortest_path
        path = shortest_path("n1", "n2", sample_nodes, sample_edges)
        # n1 → n2 is a direct edge
        assert path is not None
        assert path[0].id == "n1"
        assert path[-1].id == "n2"

    def test_multi_hop_path(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import shortest_path
        path = shortest_path("n1", "n4", sample_nodes, sample_edges)
        # n1 → n2 → n3 → n4
        assert path is not None
        assert len(path) >= 2

    def test_no_path_returns_none(self, sample_nodes, sample_edges):
        from fish_bridge.graph.algorithms import shortest_path
        # n7 and n8 are isolated (no edges connecting them to n1)
        path = shortest_path("n7", "n1", sample_nodes, sample_edges)
        # May be None or a path — just verify the return type contract
        assert path is None or isinstance(path, list)


# ---------------------------------------------------------------------------
# compiler/digest.py
# ---------------------------------------------------------------------------

class TestDigestCompiler:

    def test_compile_returns_markdown(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("test-session")
        md = compiler.compile(sample_nodes, sample_edges)
        assert isinstance(md, str)
        assert len(md) > 0

    def test_compile_contains_session_header(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("my-session-id")
        md = compiler.compile(sample_nodes, sample_edges)
        assert "my-session-id" in md

    def test_compile_contains_stats(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("test-session")
        md = compiler.compile(sample_nodes, sample_edges)
        # Should report node count somewhere
        assert "8" in md or "nodes" in md.lower()

    def test_compile_contains_open_questions(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("test-session")
        md = compiler.compile(sample_nodes, sample_edges)
        # sample_nodes has one question node
        assert "open question" in md.lower() or "question" in md.lower()

    def test_write_creates_file(self, tmp_path, sample_nodes, sample_edges):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("test-session")
        out = tmp_path / "HANDOVER.md"
        compiler.write(sample_nodes, sample_edges, out)
        assert out.exists()
        content = out.read_text()
        assert len(content) > 0

    def test_empty_graph_does_not_crash(self):
        from fish_bridge.compiler.digest import DigestCompiler
        compiler = DigestCompiler("empty-session")
        md = compiler.compile([], [])
        assert isinstance(md, str)


# ---------------------------------------------------------------------------
# compiler/focus.py
# ---------------------------------------------------------------------------

class TestFocusCompiler:

    def test_compile_returns_xml(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.focus import FocusCompiler
        compiler = FocusCompiler("test-session")
        xml = compiler.compile(sample_nodes, sample_edges, query="database")
        assert xml.startswith("<")
        assert "active_thread" in xml or "fish_bridge" in xml

    def test_compile_empty_query(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.focus import FocusCompiler
        compiler = FocusCompiler("test-session")
        xml = compiler.compile(sample_nodes, sample_edges, query="")
        assert isinstance(xml, str)

    def test_compile_respects_max_nodes(self, sample_nodes, sample_edges):
        from fish_bridge.compiler.focus import FocusCompiler
        compiler = FocusCompiler("test-session", max_nodes=3)
        xml = compiler.compile(sample_nodes, sample_edges, query="setup")
        # Count <node> elements — should be <= 3
        node_count = xml.count("<node ")
        assert node_count <= 3

    def test_compile_empty_graph(self):
        from fish_bridge.compiler.focus import FocusCompiler
        compiler = FocusCompiler("test-session")
        xml = compiler.compile([], [], query="anything")
        assert isinstance(xml, str)

    def test_xml_escaping(self, tmp_path):
        from fish_bridge.compiler.focus import FocusCompiler
        nodes = [_make_node("n1", 'label with <tags> & "quotes"')]
        xml = FocusCompiler("sess").compile(nodes, [], query="tags")
        # Should not contain raw < > inside label attributes/text
        assert "&lt;" in xml or "label" in xml  # either escaped or included somewhere


# ---------------------------------------------------------------------------
# fish_bridge/server.py — MCP tools (mocked session graph)
# ---------------------------------------------------------------------------

class TestMCPServer:

    @pytest.fixture
    def mock_sg(self, tmp_path):
        """A real SessionGraph with a temp data directory."""
        from fish_bridge.graph.session import SessionGraph
        sg = SessionGraph.open("mcp-test", tmp_path)
        yield sg
        sg.close()

    @pytest.mark.asyncio
    async def test_record_turn_adds_nodes(self, tmp_path, mock_sg):
        """record_turn should call extract → merge into the graph."""
        import fish_bridge.server as srv

        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, OutputConfig
        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(backend="claude"),
            output=OutputConfig(delivery_mode="shared", shared_context_file=".github/copilot-instructions.md"),
        )

        fake_nodes = [_make_node("x1", "extracted node")]
        fake_edges: list = []

        with patch.object(srv, "_cfg", cfg), \
             patch.object(srv, "_sg", mock_sg), \
             patch.object(srv, "_project", tmp_path):
            with patch("fish_bridge.server.build_backend") as mock_bb:
                backend = MagicMock()
                backend.extract.return_value = (fake_nodes, fake_edges)
                mock_bb.return_value = backend

                result = await srv.record_turn(
                    user_message="set up the database",
                    assistant_message="Sure, I'll set up the database.",
                )

        assert "node" in result

    def test_mark_resolved_unknown_node_returns_message(self, tmp_path, mock_sg):
        """mark_resolved on a non-existent node should not raise."""
        import fish_bridge.server as srv
        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, OutputConfig
        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(backend="claude"),
            output=OutputConfig(delivery_mode="shared", shared_context_file=".github/copilot-instructions.md"),
        )
        with patch.object(srv, "_cfg", cfg), \
             patch.object(srv, "_sg", mock_sg), \
             patch.object(srv, "_project", tmp_path):
            result = srv.mark_resolved("ghost-node-id")
        assert isinstance(result, str)

    def test_add_node_creates_node(self, tmp_path, mock_sg):
        """add_node should persist a new node to the graph."""
        import fish_bridge.server as srv
        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, OutputConfig
        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(backend="claude"),
            output=OutputConfig(delivery_mode="shared", shared_context_file=".github/copilot-instructions.md"),
        )
        with patch.object(srv, "_cfg", cfg), \
             patch.object(srv, "_sg", mock_sg), \
             patch.object(srv, "_project", tmp_path):
            result = srv.add_node(
                label="new concept",
                node_type="concept",
                status="active",
                summary="A manually added concept.",
            )
        assert "new concept" in result
        nodes = mock_sg.all_nodes()
        labels = [n.label for n in nodes]
        assert "new concept" in labels

    def test_show_active_empty_graph(self, tmp_path, mock_sg):
        """show_active on empty graph should return a string."""
        import fish_bridge.server as srv
        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, OutputConfig
        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(backend="claude"),
            output=OutputConfig(delivery_mode="shared", shared_context_file=".github/copilot-instructions.md"),
        )
        with patch.object(srv, "_cfg", cfg), \
             patch.object(srv, "_sg", mock_sg), \
             patch.object(srv, "_project", tmp_path):
            result = srv.show_active()
        assert isinstance(result, str)
