"""fish_bridge — session-scoped knowledge graph engine for AI chat context compression."""
from fish_bridge.graph.schema import GraphEdge, GraphNode, NodeStatus, NodeType, RawTurn
from fish_bridge.graph.session import SessionGraph
from fish_bridge.config import load_config, build_backend

__all__ = [
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "NodeStatus",
    "RawTurn",
    "SessionGraph",
    "load_config",
    "build_backend",
]
