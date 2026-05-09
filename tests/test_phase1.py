"""Phase 1 tests: semantic dedup, Ollama backend, session identity."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fish_bridge.extraction.dedup import (
    MERGE_THRESHOLD,
    RELATE_THRESHOLD,
    EmbeddingProvider,
    cosine_similarity,
    find_best_match,
    semantic_merge,
)
from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeType, NodeStatus


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        # cosine of 180° is -1; clipped to -1.0 (not clipped — raw cosine returned)
        score = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert score == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_empty_vectors_returns_zero(self):
        assert cosine_similarity([], []) == pytest.approx(0.0)

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_near_identical_vectors_high_score(self):
        # Two unit vectors with small angle should give score close to 1
        v1 = [0.99, 0.14]
        v2 = [0.98, 0.20]
        score = cosine_similarity(v1, v2)
        assert score > 0.99


# ---------------------------------------------------------------------------
# EmbeddingProvider (no network calls)
# ---------------------------------------------------------------------------

class TestEmbeddingProviderFallback:
    """Test that EmbeddingProvider falls back gracefully when Ollama is down."""

    def _unit_vec(self, dim: int, hot_index: int) -> list[float]:
        v = [0.0] * dim
        v[hot_index] = 1.0
        return v

    def test_cache_hit_avoids_second_call(self):
        provider = EmbeddingProvider()
        call_count = 0

        def mock_compute(text):
            nonlocal call_count
            call_count += 1
            return [0.1, 0.2, 0.3]

        provider._compute = mock_compute
        provider.embed("hello")
        provider.embed("hello")  # cache hit
        assert call_count == 1

    def test_none_returned_when_no_backend(self):
        provider = EmbeddingProvider()
        provider._backend = "none"  # simulate no backend available
        result = provider.embed("test")
        assert result is None

    def test_available_false_when_no_backend(self):
        provider = EmbeddingProvider()
        provider._backend = "none"
        assert provider.available is False


# ---------------------------------------------------------------------------
# find_best_match
# ---------------------------------------------------------------------------

class TestFindBestMatch:

    def _node(self, label: str, node_type: NodeType, embedding: list[float]) -> GraphNode:
        n = GraphNode(type=node_type, label=label)
        n.embedding = embedding
        return n

    def _provider_returning(self, vec: list[float]) -> EmbeddingProvider:
        p = EmbeddingProvider()
        p._backend = "mock"
        p._compute = lambda _text: vec  # type: ignore[method-assign]
        return p

    def test_exact_same_type_matched(self):
        existing = self._node("Redis caching", NodeType.DECISION, [1.0, 0.0])
        incoming = self._node("Redis cache layer", NodeType.DECISION, None)
        provider = self._provider_returning([0.99, 0.14])  # high similarity to existing

        # Manually embed existing in cache
        provider._cache["Redis caching"] = [1.0, 0.0]

        match, score = find_best_match(incoming, [existing], provider)
        assert match is not None
        assert score > MERGE_THRESHOLD

    def test_different_type_not_matched(self):
        existing = self._node("Redis", NodeType.SKILL, [1.0, 0.0])
        incoming = self._node("Redis adoption decision", NodeType.DECISION, None)
        provider = self._provider_returning([1.0, 0.0])

        provider._cache["Redis"] = [1.0, 0.0]

        match, score = find_best_match(incoming, [existing], provider)
        assert match is None or score < RELATE_THRESHOLD

    def test_no_embedding_returns_none(self):
        provider = EmbeddingProvider()
        provider._backend = "none"
        node = GraphNode(type=NodeType.QUESTION, label="test question")
        match, score = find_best_match(node, [], provider)
        assert match is None
        assert score == 0.0


# ---------------------------------------------------------------------------
# semantic_merge integration
# ---------------------------------------------------------------------------

class TestSemanticMerge:

    def _make_provider(self, vecs: dict[str, list[float]]) -> EmbeddingProvider:
        """Provider that returns pre-defined embeddings by label."""
        p = EmbeddingProvider()
        p._backend = "mock"
        p._cache = dict(vecs)

        def _compute(text: str) -> list[float] | None:
            return None  # only cache hits work

        p._compute = _compute  # type: ignore[method-assign]
        return p

    def test_high_similarity_causes_merge(self):
        """Score > MERGE_THRESHOLD → incoming node should NOT be in to_add."""
        existing = GraphNode(type=NodeType.QUESTION, label="Setup LLM locally")
        existing.embedding = [1.0, 0.0]

        incoming = GraphNode(type=NodeType.QUESTION, label="Local LLM setup necessity")
        incoming.embedding = [0.997, 0.077]  # cosine ≈ 0.997 → above threshold

        # Score pre-check
        score = cosine_similarity(existing.embedding, incoming.embedding)
        assert score > MERGE_THRESHOLD

        provider = self._make_provider({
            existing.label: existing.embedding,
            incoming.label: incoming.embedding,
        })

        to_add, new_edges, id_map = semantic_merge([incoming], [existing], provider)

        assert len(to_add) == 0, "Merged node should not appear in to_add"
        assert id_map[incoming.id] == existing.id, "id_map must redirect incoming → existing"

    def test_mid_similarity_adds_relates_to_edge(self):
        """Score in [RELATE_THRESHOLD, MERGE_THRESHOLD) → keep both, add edge."""
        existing = GraphNode(type=NodeType.CONCEPT, label="context window pressure")
        existing.embedding = [1.0, 0.0]

        incoming = GraphNode(type=NodeType.CONCEPT, label="token budget limit")
        # cosine between [1,0] and [0.75, 0.66] ≈ 0.75 → in the relate range
        incoming.embedding = [0.75, 0.661]
        score = cosine_similarity(existing.embedding, incoming.embedding)
        assert RELATE_THRESHOLD <= score < MERGE_THRESHOLD

        provider = self._make_provider({
            existing.label: existing.embedding,
            incoming.label: incoming.embedding,
        })

        to_add, new_edges, id_map = semantic_merge([incoming], [existing], provider)

        assert len(to_add) == 1, "Near-dup should be kept"
        assert len(new_edges) == 1
        assert new_edges[0].relation == EdgeRelation.RELATES_TO
        assert id_map[incoming.id] == incoming.id

    def test_low_similarity_node_added_independently(self):
        """Score < RELATE_THRESHOLD → completely new node, no edge."""
        existing = GraphNode(type=NodeType.SKILL, label="AWS CDK")
        existing.embedding = [1.0, 0.0]

        incoming = GraphNode(type=NodeType.SKILL, label="fish_bridge dedup")
        incoming.embedding = [0.0, 1.0]  # orthogonal → cosine = 0.0

        provider = self._make_provider({
            existing.label: existing.embedding,
            incoming.label: incoming.embedding,
        })

        to_add, new_edges, id_map = semantic_merge([incoming], [existing], provider)

        assert len(to_add) == 1
        assert len(new_edges) == 0
        assert id_map[incoming.id] == incoming.id

    def test_no_embedding_available_adds_node(self):
        """If provider returns None, node is added without dedup."""
        existing = GraphNode(type=NodeType.QUESTION, label="Q1")
        incoming = GraphNode(type=NodeType.QUESTION, label="Q1 rephrased")

        provider = EmbeddingProvider()
        provider._backend = "none"  # simulate no embedding backend

        to_add, new_edges, id_map = semantic_merge([incoming], [existing], provider)

        assert len(to_add) == 1
        assert len(new_edges) == 0


# ---------------------------------------------------------------------------
# Session identity
# ---------------------------------------------------------------------------

class TestSessionIdentity:

    def test_derives_from_workspace_name(self, tmp_path: Path):
        from fish_bridge.config import get_active_session_id
        import datetime as dt

        today = dt.date.today().isoformat()
        workspace = tmp_path / "my-project"
        workspace.mkdir()

        session_id = get_active_session_id(workspace)
        assert session_id == f"my-project-{today}"

    def test_persists_to_lock_file(self, tmp_path: Path):
        from fish_bridge.config import get_active_session_id

        workspace = tmp_path / "test-ws"
        workspace.mkdir()

        session_id = get_active_session_id(workspace)
        lock_file  = workspace / ".fish_bridge" / "session.lock"
        assert lock_file.exists()
        assert lock_file.read_text(encoding="utf-8").strip() == session_id

    def test_reads_existing_lock_file(self, tmp_path: Path):
        from fish_bridge.config import get_active_session_id

        workspace = tmp_path / "ws"
        workspace.mkdir()
        lock_dir  = workspace / ".fish_bridge"
        lock_dir.mkdir()
        lock_file = lock_dir / "session.lock"
        lock_file.write_text("custom-session-id", encoding="utf-8")

        assert get_active_session_id(workspace) == "custom-session-id"

    def test_stable_across_calls(self, tmp_path: Path):
        from fish_bridge.config import get_active_session_id

        workspace = tmp_path / "stable-ws"
        workspace.mkdir()

        id1 = get_active_session_id(workspace)
        id2 = get_active_session_id(workspace)
        assert id1 == id2


# ---------------------------------------------------------------------------
# OllamaBackend (mocked httpx)
# ---------------------------------------------------------------------------

class TestOllamaBackend:

    def _mock_response(self, content: str) -> MagicMock:
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"message": {"content": content}}
        return mock

    def test_extract_calls_ollama_chat(self):
        from fish_bridge.extraction.local import OllamaBackend
        import json

        response_payload = json.dumps({
            "nodes": [{"type": "concept", "label": "test node", "summary": "a test",
                       "status": "active", "confidence": 0.9}],
            "edges": [],
        })

        backend = OllamaBackend(model="qwen2.5:7b")

        with patch("httpx.post") as mock_post:
            mock_post.return_value = self._mock_response(response_payload)
            from fish_bridge.graph.schema import RawTurn
            turn = RawTurn(
                session_id="test",
                turn_number=1,
                role_user="how should I implement caching?",
                role_assistant="Use Redis with a 24hr TTL.",
            )
            nodes, edges = backend.extract(turn)

        assert len(nodes) == 1
        assert nodes[0].label == "test node"
        assert mock_post.called

    def test_is_available_false_when_connection_refused(self):
        from fish_bridge.extraction.local import OllamaBackend
        backend = OllamaBackend(base_url="http://localhost:19999")  # nothing there
        assert backend.is_available() is False

    def test_build_backend_local(self, tmp_path: Path):
        """build_backend('local') should return OllamaBackend."""
        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, LocalConfig, build_backend
        from fish_bridge.extraction.local import OllamaBackend

        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(backend="local")
        )
        backend = build_backend(cfg)
        assert isinstance(backend, OllamaBackend)
