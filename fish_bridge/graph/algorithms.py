"""Graph algorithms for fish_bridge session graphs.

Provides:
  - build_nx_graph()          → convert nodes/edges to a networkx DiGraph
  - community_detection()     → Louvain clustering via networkx (greedy fallback)
  - subgraph_neighborhood()   → BFS neighborhood around a set of seed nodes
  - semantic_search_nodes()   → find nodes most similar to a query by label/summary text
  - shortest_path()           → shortest path between two nodes
"""
from __future__ import annotations


from fish_bridge.graph.schema import GraphEdge, GraphNode

# networkx is a core dependency (in pyproject.toml)
import networkx as nx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_nx_graph(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    directed: bool = True,
) -> nx.DiGraph | nx.Graph:
    """Convert fish_bridge nodes/edges to a networkx graph.

    Node attributes:
        label, type, status, confidence, summary

    Edge attributes:
        relation, weight
    """
    G: nx.DiGraph | nx.Graph = nx.DiGraph() if directed else nx.Graph()

    for n in nodes:
        G.add_node(
            n.id,
            label=      n.label,
            type=       str(n.type),
            status=     str(n.status),
            confidence= n.confidence,
            summary=    n.summary,
        )

    for e in edges:
        if G.has_node(e.from_id) and G.has_node(e.to_id):
            G.add_edge(
                e.from_id,
                e.to_id,
                relation= str(e.relation),
                weight=   e.weight,
            )

    return G


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def community_detection(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    resolution: float = 1.0,
) -> dict[str, int]:
    """Detect communities using the Louvain method (or greedy modularity fallback).

    Returns a dict mapping node_id → community_id (int, 0-based).
    Uses an undirected graph for clustering (direction is irrelevant for topology).

    Args:
        nodes:      All nodes in the session.
        edges:      All edges in the session.
        resolution: Louvain resolution parameter — higher = more communities.
    """
    G_undirected = build_nx_graph(nodes, edges, directed=False)

    if len(G_undirected.nodes) == 0:
        return {}

    # Isolate nodes (no edges) each get their own community
    node_id_map: dict[str, int] = {}

    try:
        # networkx 3.x: greedy_modularity_communities is reliable
        # nx.community.louvain_communities is available in networkx >= 3.3
        communities = nx.community.louvain_communities(
            G_undirected, resolution=resolution, seed=42
        )
    except AttributeError:
        # Fallback for older networkx versions
        communities = nx.community.greedy_modularity_communities(G_undirected)

    for community_id, member_set in enumerate(communities):
        for node_id in member_set:
            node_id_map[node_id] = community_id

    # Assign isolated nodes (not in any community) their own community
    next_id = len(communities)
    for node in nodes:
        if node.id not in node_id_map:
            node_id_map[node.id] = next_id
            next_id += 1

    return node_id_map


def group_nodes_by_community(
    nodes: list[GraphNode],
    community_map: dict[str, int],
) -> dict[int, list[GraphNode]]:
    """Group nodes by their community ID."""
    groups: dict[int, list[GraphNode]] = {}
    for node in nodes:
        cid = community_map.get(node.id, -1)
        groups.setdefault(cid, []).append(node)
    return groups


# ---------------------------------------------------------------------------
# Subgraph neighborhood
# ---------------------------------------------------------------------------

def subgraph_neighborhood(
    seed_ids: list[str],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    max_hops: int = 2,
    max_nodes: int = 20,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Return the BFS neighborhood of seed nodes within max_hops.

    Useful for FocusCompiler: given a set of relevant seed nodes, expand
    to include their graph neighborhood for richer context.

    Returns (nodes_in_subgraph, edges_in_subgraph).
    """
    G = build_nx_graph(nodes, edges, directed=False)  # undirected for BFS
    node_by_id = {n.id: n for n in nodes}
    edge_by_endpoints: dict[tuple[str, str], GraphEdge] = {}
    for e in edges:
        edge_by_endpoints[(e.from_id, e.to_id)] = e
        edge_by_endpoints[(e.to_id, e.from_id)] = e  # undirected access

    visited: set[str] = set()
    frontier = [nid for nid in seed_ids if G.has_node(nid)]

    for _hop in range(max_hops):
        next_frontier: list[str] = []
        for nid in frontier:
            if nid in visited:
                continue
            visited.add(nid)
            if len(visited) >= max_nodes:
                break
            for neighbor in G.neighbors(nid):
                if neighbor not in visited:
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier or len(visited) >= max_nodes:
            break

    # Also add seed nodes themselves
    visited.update(nid for nid in seed_ids if G.has_node(nid))

    sub_nodes = [node_by_id[nid] for nid in visited if nid in node_by_id]
    sub_edges = [
        e for e in edges
        if e.from_id in visited and e.to_id in visited
    ]
    return sub_nodes[:max_nodes], sub_edges


# ---------------------------------------------------------------------------
# Semantic search (label + summary text matching)
# ---------------------------------------------------------------------------

def semantic_search_nodes(
    query: str,
    nodes: list[GraphNode],
    top_k: int = 10,
) -> list[tuple[GraphNode, float]]:
    """Find nodes most relevant to *query* by simple TF-IDF-style term overlap.

    Returns list of (node, score) sorted by descending score.
    Used when embeddings are not available (no Ollama, no sentence-transformers).

    For richer semantic search, use EmbeddingProvider.embed() + cosine_similarity
    from extraction.dedup directly.
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    scored: list[tuple[GraphNode, float]] = []
    for node in nodes:
        text  = f"{node.label} {node.summary}"
        terms = set(_tokenize(text))
        if not terms:
            continue
        # Jaccard similarity
        intersection = len(query_terms & terms)
        union        = len(query_terms | terms)
        score = intersection / union if union > 0 else 0.0
        # Boost by confidence
        score *= node.confidence
        if score > 0.0:
            scored.append((node, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer — strips punctuation."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Shortest path
# ---------------------------------------------------------------------------

def shortest_path(
    from_id: str,
    to_id: str,
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[GraphNode] | None:
    """Return the shortest path between two nodes, or None if no path exists."""
    G = build_nx_graph(nodes, edges, directed=False)
    node_by_id = {n.id: n for n in nodes}
    try:
        path_ids = nx.shortest_path(G, from_id, to_id)
        return [node_by_id[nid] for nid in path_ids if nid in node_by_id]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ---------------------------------------------------------------------------
# Community summary helpers
# ---------------------------------------------------------------------------

def dominant_type(nodes: list[GraphNode]) -> str:
    """Return the most common node type in a list."""
    from collections import Counter
    if not nodes:
        return "concept"
    counts = Counter(str(n.type) for n in nodes)
    return counts.most_common(1)[0][0]


def community_label(nodes: list[GraphNode]) -> str:
    """Generate a short label for a community based on its highest-confidence nodes."""
    sorted_nodes = sorted(nodes, key=lambda n: n.confidence, reverse=True)
    top = sorted_nodes[:3]
    return " / ".join(n.label for n in top)
