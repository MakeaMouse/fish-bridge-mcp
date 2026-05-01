"""SessionGraph — high-level CRUD operations on top of SessionStore.

Each method here corresponds to a logical operation the pipeline or CLI needs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fish_bridge.graph.schema import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    NodeStatus,
    NodeType,
    SessionGraph as SessionGraphModel,
)
from fish_bridge.graph.store import SessionStore


class SessionGraph:
    """Manages one session's graph: add nodes/edges, query, persist."""

    def __init__(
        self,
        session_id: str,
        db_path: Path,
        lock_path: Path,
    ) -> None:
        self.session_id = session_id
        self._store = SessionStore(db_path, lock_path)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, session_id: str, data_dir: Path) -> "SessionGraph":
        """Open (or create) a session graph in data_dir."""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path   = data_dir / f"{session_id}.db"
        lock_path = data_dir / "session.lock"
        return cls(session_id, db_path, lock_path)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> GraphNode:
        """Persist a node.  Returns the node (may have been modified)."""
        self._store.upsert_node(self.session_id, node)
        return node

    def update_node(self, node: GraphNode) -> None:
        """Persist updates to an existing node."""
        node.updated_at = datetime.now(timezone.utc)
        self._store.upsert_node(self.session_id, node)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._store.get_node(node_id)

    def all_nodes(self) -> list[GraphNode]:
        return self._store.list_nodes(self.session_id)

    def active_nodes(self) -> list[GraphNode]:
        """Nodes that belong in the active compiled thread."""
        active_statuses = {
            NodeStatus.ACTIVE,
            NodeStatus.PROPOSED,
            NodeStatus.ADOPTED,
            NodeStatus.PENDING,
            NodeStatus.IN_PROGRESS,
            NodeStatus.BLOCKED,
            NodeStatus.CONFLICTED,
            # errors that are still open
            NodeStatus.UNCONFIRMED,
        }
        return [n for n in self.all_nodes() if NodeStatus(n.status) in active_statuses]

    def find_by_label(self, label: str, node_type: NodeType | None = None) -> list[GraphNode]:
        nodes = self.all_nodes()
        label_lower = label.lower()
        hits = [n for n in nodes if n.label.lower() == label_lower]
        if node_type:
            hits = [n for n in hits if n.type == node_type]
        return hits

    def set_status(self, node_id: str, status: NodeStatus, note: str | None = None) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        node.push_status(status, note)
        self.update_node(node)
        return True

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        self._store.upsert_edge(self.session_id, edge)
        return edge

    def all_edges(self) -> list[GraphEdge]:
        return self._store.list_edges(self.session_id)

    def edges_for_node(self, node_id: str) -> list[GraphEdge]:
        return [
            e for e in self.all_edges()
            if e.from_id == node_id or e.to_id == node_id
        ]

    # ------------------------------------------------------------------
    # Bulk merge (from extraction engine output)
    # ------------------------------------------------------------------

    def merge_extraction(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Add extracted nodes/edges, de-duplicating by label+type.

        Returns the final (possibly merged) nodes and edges that were stored.
        Edges whose endpoints don't exist after merge are skipped.
        """
        # Build label→id map for existing nodes
        existing = {(n.type, n.label.lower()): n for n in self.all_nodes()}
        id_map: dict[str, str] = {}  # incoming id → canonical stored id

        stored_nodes: list[GraphNode] = []
        for node in nodes:
            key = (node.type if isinstance(node.type, str) else node.type.value,
                   node.label.lower())
            if key in existing:
                # Update existing: merge summary and confidence
                existing_node = existing[key]
                if node.summary and node.summary != existing_node.summary:
                    existing_node.summary = node.summary
                existing_node.confidence = max(existing_node.confidence, node.confidence)
                existing_node.touch()
                self.update_node(existing_node)
                id_map[node.id] = existing_node.id
                stored_nodes.append(existing_node)
            else:
                self.add_node(node)
                existing[key] = node
                id_map[node.id] = node.id
                stored_nodes.append(node)

        # Resolve edge endpoints using id_map
        stored_edges: list[GraphEdge] = []
        node_ids = {n.id for n in self.all_nodes()}
        for edge in edges:
            resolved_from = id_map.get(edge.from_id, edge.from_id)
            resolved_to   = id_map.get(edge.to_id,   edge.to_id)
            if resolved_from not in node_ids or resolved_to not in node_ids:
                continue  # skip phantom edges
            edge.from_id = resolved_from
            edge.to_id   = resolved_to
            self.add_edge(edge)
            stored_edges.append(edge)

        return stored_nodes, stored_edges

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def to_model(self) -> SessionGraphModel:
        return SessionGraphModel(
            session_id=self.session_id,
            nodes=self.all_nodes(),
            edges=self.all_edges(),
        )

    def export_json(self, path: Path) -> None:
        model = self.to_model()
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def import_json(cls, path: Path, data_dir: Path) -> "SessionGraph":
        raw = json.loads(path.read_text(encoding="utf-8"))
        model = SessionGraphModel.model_validate(raw)
        sg = cls.open(model.session_id, data_dir)
        for node in model.nodes:
            sg.add_node(node)
        for edge in model.edges:
            sg.add_edge(edge)
        return sg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._store.close()
