"""Semantic deduplication for fish_bridge.

Replaces exact-label dedup in session.py with embedding + cosine similarity,
preventing near-duplicate nodes like:
  "Setup LLM locally?" / "Local LLM setup necessity" / "Is machine good enough?"

Embedding sources (tried in order):
  1. Ollama nomic-embed-text  (via /api/embed — zero cost, offline)
  2. sentence-transformers    (CI fallback — pip install fish-bridge-mcp[local])
  3. None                     (falls back to exact-label matching in session.py)

Thresholds:
  >0.88 → merge (incoming node updates existing node in place)
  0.70–0.88 → keep both, add relates-to edge
  <0.70 → independent node (new addition)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

MERGE_THRESHOLD  = 0.88   # above this → merge into single node
RELATE_THRESHOLD = 0.70   # above this (and below MERGE) → add relates-to edge


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [0, 1] between two equal-length vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot    = sum(x * y for x, y in zip(a, b))
    mag_a  = math.sqrt(sum(x * x for x in a))
    mag_b  = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

def _embed_ollama(label: str, base_url: str = "http://localhost:11434") -> list[float] | None:
    """Embed a label string using Ollama nomic-embed-text.  Returns None if unavailable."""
    try:
        import httpx  # already a core dependency
        resp = httpx.post(
            f"{base_url}/api/embed",
            json={"model": "nomic-embed-text", "input": label},
            timeout=10.0,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        if embeddings and len(embeddings) > 0:
            return embeddings[0]
        return None
    except Exception:
        return None


def _embed_sentence_transformers(label: str) -> list[float] | None:
    """Fallback: sentence-transformers all-MiniLM-L6-v2 (optional dep)."""
    try:
        _model = _get_st_model()
        if _model is None:
            return None
        vec = _model.encode(label, convert_to_numpy=True)
        return vec.tolist()
    except Exception:
        return None


_st_model_cache: object | None = None
_st_model_loaded: bool = False


def _get_st_model() -> object | None:
    global _st_model_cache, _st_model_loaded
    if _st_model_loaded:
        return _st_model_cache
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _st_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _st_model_cache = None
    _st_model_loaded = True
    return _st_model_cache


class EmbeddingProvider:
    """Manages embedding source selection and caches embeddings for a session."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434") -> None:
        self._ollama_base_url = ollama_base_url
        self._cache: dict[str, list[float]] = {}
        self._backend: str | None = None   # "ollama" | "sentence-transformers" | None

    def embed(self, text: str) -> list[float] | None:
        """Return embedding vector, using cache to avoid redundant calls."""
        if text in self._cache:
            return self._cache[text]

        vec = self._compute(text)
        if vec is not None:
            self._cache[text] = vec
        return vec

    def _compute(self, text: str) -> list[float] | None:
        # Try Ollama first (or use whichever backend worked last time)
        if self._backend in (None, "ollama"):
            vec = _embed_ollama(text, self._ollama_base_url)
            if vec is not None:
                self._backend = "ollama"
                return vec

        # Fallback to sentence-transformers
        if self._backend in (None, "sentence-transformers"):
            vec = _embed_sentence_transformers(text)
            if vec is not None:
                self._backend = "sentence-transformers"
                return vec

        # No embedding available
        self._backend = "none"
        return None

    @property
    def available(self) -> bool:
        """True if at least one embedding provider is reachable."""
        if self._backend == "none":
            return False
        if self._backend is not None:
            return True
        # Probe by embedding a test string
        return self.embed("test") is not None


# ---------------------------------------------------------------------------
# Deduplication logic
# ---------------------------------------------------------------------------

def find_best_match(
    node: GraphNode,
    candidates: list[GraphNode],
    provider: EmbeddingProvider,
) -> tuple[GraphNode | None, float]:
    """Find the highest-similarity existing node for *node* among *candidates*.

    Only compares nodes of the same type.  Returns (match, score).
    If embeddings are unavailable, returns (None, 0.0).
    """
    node_vec = provider.embed(node.label)
    if node_vec is None:
        return None, 0.0

    best_node: GraphNode | None = None
    best_score = 0.0

    node_type = node.type if isinstance(node.type, str) else node.type.value

    for candidate in candidates:
        cand_type = candidate.type if isinstance(candidate.type, str) else candidate.type.value
        if cand_type != node_type:
            continue
        # Compute embedding lazily
        if candidate.embedding is None:
            candidate.embedding = provider.embed(candidate.label)
        if candidate.embedding is None:
            continue
        score = cosine_similarity(node_vec, candidate.embedding)
        if score > best_score:
            best_score = score
            best_node = candidate

    return best_node, best_score


def semantic_merge(
    incoming_nodes: list[GraphNode],
    existing_nodes: list[GraphNode],
    provider: EmbeddingProvider,
    merge_threshold: float = MERGE_THRESHOLD,
    relate_threshold: float = RELATE_THRESHOLD,
) -> tuple[list[GraphNode], list[GraphEdge], dict[str, str]]:
    """Apply semantic dedup across *incoming_nodes* vs *existing_nodes*.

    Args:
        incoming_nodes:  Freshly extracted nodes to check for duplicates.
        existing_nodes:  Already-persisted nodes in the graph.
        provider:        Embedding provider to use for similarity.
        merge_threshold: Cosine similarity above which nodes are merged (default 0.88).
        relate_threshold: Cosine similarity above which a relates-to edge is added (default 0.70).

    Returns:
        to_add    — new nodes that should be inserted (not merged into existing)
        new_edges — new relates-to edges for near-duplicates kept separate
        id_map    — {incoming_id: canonical_id}  (identity map for non-merged nodes)

    The caller is responsible for persisting to_add and updating merged
    existing nodes.  Mutates existing nodes' embeddings in-place for caching.
    """
    to_add:    list[GraphNode]  = []
    new_edges: list[GraphEdge]  = []
    id_map:    dict[str, str]   = {}

    # Accumulate newly added nodes so duplicates within the same batch are also caught
    accumulated = list(existing_nodes)

    for node in incoming_nodes:
        # Embed incoming node
        node.embedding = provider.embed(node.label)

        match, score = find_best_match(node, accumulated, provider)

        if match is not None and score >= merge_threshold:
            # Merge: incoming node collapses into existing
            if node.summary and node.summary != match.summary:
                match.summary = node.summary
            match.confidence = max(match.confidence, node.confidence)
            match.touch()
            id_map[node.id] = match.id

        elif match is not None and score >= relate_threshold:
            # Near-duplicate: keep both, add relates-to edge
            id_map[node.id] = node.id
            node.embedding = node.embedding  # already set above
            to_add.append(node)
            accumulated.append(node)
            new_edges.append(
                GraphEdge(
                    from_id=   node.id,
                    to_id=     match.id,
                    relation=  EdgeRelation.RELATES_TO,
                    weight=    score,
                    created_at=node.created_at,
                )
            )

        else:
            # No match — add as independent node
            id_map[node.id] = node.id
            to_add.append(node)
            accumulated.append(node)

    return to_add, new_edges, id_map
