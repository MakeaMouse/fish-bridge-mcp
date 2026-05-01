"""SQLite persistence layer for fish_bridge session graphs.

Uses WAL journal mode for concurrent read safety.
Uses fcntl.flock (POSIX) or msvcrt.locking (Windows) to serialize writes.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from fish_bridge.graph.schema import GraphEdge, GraphNode


# ---------------------------------------------------------------------------
# Platform-aware advisory write lock
# ---------------------------------------------------------------------------

if sys.platform == "win32":
    import msvcrt

    def _acquire_write_lock(fp) -> None:  # type: ignore[type-arg]
        msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)

    def _release_write_lock(fp) -> None:  # type: ignore[type-arg]
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _acquire_write_lock(fp) -> None:  # type: ignore[type-arg]
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)

    def _release_write_lock(fp) -> None:  # type: ignore[type-arg]
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS nodes (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    type            TEXT NOT NULL,
    label           TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,
    subtype         TEXT,
    source_url      TEXT,
    metadata        TEXT NOT NULL DEFAULT '{}',
    status_history  TEXT NOT NULL DEFAULT '[]',
    embedding       TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_session ON nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type    ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_status  ON nodes(status);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    from_id     TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    to_id       TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_session  ON edges(session_id);
CREATE INDEX IF NOT EXISTS idx_edges_from     ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to       ON edges(to_id);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SessionStore:
    """Low-level SQLite store for nodes and edges.

    One store instance per database file.  The write_lock file path
    (session_lock_path) is used as the target for fcntl.flock so that
    concurrent CLI invocations don't race on the SQLite file directly.
    """

    def __init__(self, db_path: Path, lock_path: Path) -> None:
        self._db_path = db_path
        self._lock_path = lock_path
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        db = self._open_conn()
        db.executescript(_DDL)
        db.commit()

    def _open_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write context manager (advisory lock + EXCLUSIVE txn)
    # ------------------------------------------------------------------

    @contextmanager
    def _write_lock(self) -> Generator[sqlite3.Connection, None, None]:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a") as lock_fp:
            _acquire_write_lock(lock_fp)
            try:
                conn = self._open_conn()
                conn.execute("BEGIN EXCLUSIVE")
                try:
                    yield conn
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            finally:
                _release_write_lock(lock_fp)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def upsert_node(self, session_id: str, node: GraphNode) -> None:
        """Insert or replace a node."""
        row = _node_to_row(session_id, node)
        with self._write_lock() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO nodes
                    (id, session_id, type, label, summary, status, confidence,
                     subtype, source_url, metadata, status_history, embedding,
                     created_at, updated_at)
                VALUES
                    (:id, :session_id, :type, :label, :summary, :status, :confidence,
                     :subtype, :source_url, :metadata, :status_history, :embedding,
                     :created_at, :updated_at)
                """,
                row,
            )

    def get_node(self, node_id: str) -> GraphNode | None:
        conn = self._open_conn()
        row = conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return _row_to_node(row) if row else None

    def list_nodes(self, session_id: str) -> list[GraphNode]:
        conn = self._open_conn()
        rows = conn.execute(
            "SELECT * FROM nodes WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_row_to_node(r) for r in rows]

    def delete_node(self, node_id: str) -> None:
        with self._write_lock() as conn:
            conn.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def upsert_edge(self, session_id: str, edge: GraphEdge) -> None:
        with self._write_lock() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO edges
                    (id, session_id, from_id, to_id, relation, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    session_id,
                    edge.from_id,
                    edge.to_id,
                    edge.relation if isinstance(edge.relation, str) else edge.relation.value,
                    edge.weight,
                    _dt_str(edge.created_at),
                ),
            )

    def list_edges(self, session_id: str) -> list[GraphEdge]:
        conn = self._open_conn()
        rows = conn.execute(
            "SELECT * FROM edges WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def delete_edge(self, edge_id: str) -> None:
        with self._write_lock() as conn:
            conn.execute("DELETE FROM edges WHERE id=?", (edge_id,))


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------

def _dt_str(dt: datetime) -> str:
    return dt.isoformat()


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _node_to_row(session_id: str, node: GraphNode) -> dict:
    return {
        "id":             node.id,
        "session_id":     session_id,
        "type":           node.type if isinstance(node.type, str) else node.type.value,
        "label":          node.label,
        "summary":        node.summary,
        "status":         node.status if isinstance(node.status, str) else node.status.value,
        "confidence":     node.confidence,
        "subtype":        node.subtype,
        "source_url":     node.source_url,
        "metadata":       json.dumps(node.metadata),
        "status_history": json.dumps([e.model_dump(mode="json") for e in node.status_history]),
        "embedding":      json.dumps(node.embedding) if node.embedding is not None else None,
        "created_at":     _dt_str(node.created_at),
        "updated_at":     _dt_str(node.updated_at),
    }


def _row_to_node(row: sqlite3.Row) -> GraphNode:
    from fish_bridge.graph.schema import NodeStatus, NodeType, StatusHistoryEntry

    history_raw = json.loads(row["status_history"])
    history = [
        StatusHistoryEntry(
            status=NodeStatus(h["status"]),
            timestamp=_parse_dt(h["timestamp"]),
            note=h.get("note"),
        )
        for h in history_raw
    ]
    embedding_raw = row["embedding"]
    return GraphNode(
        id=row["id"],
        type=NodeType(row["type"]),
        label=row["label"],
        summary=row["summary"],
        status=NodeStatus(row["status"]),
        confidence=row["confidence"],
        subtype=row["subtype"],
        source_url=row["source_url"],
        metadata=json.loads(row["metadata"]),
        status_history=history,
        embedding=json.loads(embedding_raw) if embedding_raw else None,
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
    )


def _row_to_edge(row: sqlite3.Row) -> GraphEdge:
    from fish_bridge.graph.schema import EdgeRelation

    return GraphEdge(
        id=row["id"],
        from_id=row["from_id"],
        to_id=row["to_id"],
        relation=EdgeRelation(row["relation"]),
        weight=row["weight"],
        created_at=_parse_dt(row["created_at"]),
    )
