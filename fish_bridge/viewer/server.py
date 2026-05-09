"""fish_bridge local web graph viewer.

Serves the Cytoscape.js graph viewer at http://localhost:PORT.
Binds to 127.0.0.1 only — no external network access.

Security note: validates Host header on every request to prevent
DNS rebinding attacks (a malicious page re-resolving a domain to 127.0.0.1).

Usage (via CLI):
    fish-bridge serve                     # default port 8080
    fish-bridge serve --port 9090
    fish-bridge serve --session my-id
    fish-bridge serve --no-open
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

VIEWER_DIR = Path(__file__).parent

# Platform-aware write lock (mirrors store.py)
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


# Allowed node statuses (must match NodeStatus enum)
_VALID_STATUSES = frozenset({
    "active", "proposed", "adopted", "pending", "in_progress",
    "resolved", "done", "fixed", "deferred", "rejected",
    "blocked", "unconfirmed", "conflicted", "superseded",
})


class _GraphHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for the graph viewer."""

    # Injected by the factory — populated before server starts
    graph_json: str = "{}"
    allowed_hosts: frozenset[str] = frozenset({"localhost", "127.0.0.1"})
    # Optional db_path / lock_path for live node edits (None → read-only mode)
    db_path: Path | None = None
    lock_path: Path | None = None
    # Optional sessions directory for the session switcher (B1)
    data_dir: Path | None = None

    # ------------------------------------------------------------------
    # Security: Host header validation (DNS rebinding prevention)
    # ------------------------------------------------------------------

    def _is_host_allowed(self) -> bool:
        host = self.headers.get("Host", "").split(":")[0].lower()
        return host in self.allowed_hosts or host == ""

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        if not self._is_host_allowed():
            self._send_403()
            return

        # Split path and query string
        if "?" in self.path:
            raw_path, query_string = self.path.split("?", 1)
        else:
            raw_path, query_string = self.path, ""
        path = raw_path.rstrip("/") or "/"

        # Parse query params
        params: dict[str, str] = {}
        for part in query_string.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v

        if path == "/api/graph":
            session_param = params.get("session")
            if session_param and self.data_dir:
                self._serve_json(self._load_session_graph(session_param))
            else:
                self._serve_json(self.graph_json)
        elif path == "/api/sessions":
            self._serve_json(self._list_sessions())
        elif path in ("/", "/index.html", ""):
            self._serve_file(VIEWER_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/viewer.css":
            self._serve_file(VIEWER_DIR / "viewer.css", "text/css; charset=utf-8")
        elif path == "/viewer.js":
            self._serve_file(VIEWER_DIR / "viewer.js", "application/javascript; charset=utf-8")
        elif path == "/cytoscape.min.js":
            cjs = VIEWER_DIR / "cytoscape.min.js"
            if cjs.exists():
                self._serve_file(cjs, "application/javascript; charset=utf-8")
            else:
                self._send_404()
        else:
            self._send_404()

    def do_POST(self) -> None:
        if not self._is_host_allowed():
            self._send_403()
            return

        path = self.path.split("?")[0]

        # Only endpoint: POST /api/node/<id>
        if not path.startswith("/api/node/"):
            self._send_404()
            return

        node_id = path[len("/api/node/"):]
        if not node_id or "/" in node_id:
            self._send_error_json(400, "Invalid node id")
            return

        if self.db_path is None:
            self._send_error_json(503, "Database not available (read-only mode)")
            return

        # Read body
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65_536:
                self._send_error_json(413, "Payload too large")
                return
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_error_json(400, "Invalid JSON body")
            return

        if not isinstance(payload, dict):
            self._send_error_json(400, "Body must be a JSON object")
            return

        # Validate fields
        new_status  = payload.get("status")
        new_summary = payload.get("summary")

        if new_status is not None and new_status not in _VALID_STATUSES:
            self._send_error_json(400, f"Unknown status: {new_status!r}")
            return
        if new_summary is not None and not isinstance(new_summary, str):
            self._send_error_json(400, "summary must be a string")
            return

        # Persist to SQLite under write lock
        try:
            self._update_node(node_id, new_status, new_summary)
        except LookupError:
            self._send_error_json(404, f"Node not found: {node_id}")
            return
        except Exception as exc:
            self._send_error_json(500, str(exc))
            return

        self._serve_json(json.dumps({"ok": True, "id": node_id}))

    # ------------------------------------------------------------------
    # Session switcher helpers (B1)
    # ------------------------------------------------------------------

    def _list_sessions(self) -> str:
        """Return a JSON array of session objects with id, nodes, and updated."""
        if self.data_dir is None or not self.data_dir.exists():
            return json.dumps([])
        import datetime
        sessions = []
        for p in sorted(self.data_dir.glob("*.db")):
            node_count = 0
            updated = ""
            try:
                conn = sqlite3.connect(str(p))
                row = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
                node_count = row[0] if row else 0
                # Use SQLite file mtime as fallback for updated time
                mtime = p.stat().st_mtime
                updated = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                conn.close()
            except Exception:
                pass
            sessions.append({"id": p.stem, "nodes": node_count, "updated": updated})
        return json.dumps(sessions)

    def _load_session_graph(self, session_id: str) -> str:
        """Dynamically load a session from SQLite and return graph JSON.

        Validates session_id to prevent path traversal.
        Falls back to the initial graph_json on any error.
        """
        # Only allow alphanumeric, hyphen, underscore, dot (no slashes, etc.)
        if not re.fullmatch(r"[\w\-\.]+", session_id):
            return self.graph_json

        db_path = self.data_dir / f"{session_id}.db"  # type: ignore[operator]
        if not db_path.exists():
            return self.graph_json

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            nodes_rows = conn.execute("SELECT * FROM nodes").fetchall()
            edges_rows = conn.execute("SELECT * FROM edges").fetchall()
            conn.close()
        except Exception:
            return self.graph_json

        nodes = [
            {
                "id":         r["id"],
                "label":      r["label"],
                "type":       r["type"],
                "status":     r["status"],
                "summary":    r["summary"] or "",
                "confidence": r["confidence"] or 1.0,
            }
            for r in nodes_rows
        ]
        edges = [
            {
                "id":      r["id"],
                "from_id": r["from_id"],
                "to_id":   r["to_id"],
                "relation": r["relation"],
                "weight":   r["weight"] or 1.0,
            }
            for r in edges_rows
        ]
        return json.dumps({"session_id": session_id, "nodes": nodes, "edges": edges})

    # ------------------------------------------------------------------
    # Database update
    # ------------------------------------------------------------------

    def _update_node(
        self,
        node_id: str,
        new_status: str | None,
        new_summary: str | None,
    ) -> None:
        """Update a node's status and/or summary under an advisory write lock."""
        from datetime import datetime, timezone

        lock_path = self.lock_path or (self.db_path.parent / "session.lock")  # type: ignore[union-attr]
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "a") as lock_fp:
            _acquire_write_lock(lock_fp)
            try:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    # Verify node exists
                    row = conn.execute(
                        "SELECT id FROM nodes WHERE id=?", (node_id,)
                    ).fetchone()
                    if row is None:
                        raise LookupError(node_id)

                    now = datetime.now(timezone.utc).isoformat()
                    if new_status is not None and new_summary is not None:
                        conn.execute(
                            "UPDATE nodes SET status=?, summary=?, updated_at=? WHERE id=?",
                            (new_status, new_summary, now, node_id),
                        )
                    elif new_status is not None:
                        conn.execute(
                            "UPDATE nodes SET status=?, updated_at=? WHERE id=?",
                            (new_status, now, node_id),
                        )
                    elif new_summary is not None:
                        conn.execute(
                            "UPDATE nodes SET summary=?, updated_at=? WHERE id=?",
                            (new_summary, now, node_id),
                        )
                    conn.commit()
                finally:
                    conn.close()
            finally:
                _release_write_lock(lock_fp)

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _serve_json(self, data: str) -> None:
        body = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_404()
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        self.send_response(404)
        self.end_headers()

    def _send_403(self) -> None:
        self.send_response(403)
        self.end_headers()

    def _send_error_json(self, code: int, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass  # suppress access log noise


def run_viewer(
    graph_json: str,
    port: int = 8080,
    open_browser: bool = True,
    db_path: Path | None = None,
    lock_path: Path | None = None,
    data_dir: Path | None = None,
) -> None:
    """Start the graph viewer HTTP server.

    Blocks until the user sends Ctrl+C.

    Args:
        graph_json:   Serialised graph for the initial load.
        port:         TCP port to bind (127.0.0.1 only).
        open_browser: Open the default browser automatically.
        db_path:      SQLite database file; enables live node editing when set.
        lock_path:    Advisory lock file; defaults to <db_path parent>/session.lock.
        data_dir:     Sessions directory; enables the session switcher when set.
    """
    # Inject graph data and optional db paths into handler class
    handler_class = type(
        "_Handler",
        (_GraphHandler,),
        {
            "graph_json":    graph_json,
            "allowed_hosts": frozenset({"localhost", "127.0.0.1"}),
            "db_path":       db_path,
            "lock_path":     lock_path,
            "data_dir":      data_dir,
        },
    )

    # Bind to 127.0.0.1 only — not 0.0.0.0
    server = HTTPServer(("127.0.0.1", port), handler_class)

    url = f"http://localhost:{port}"
    edit_note = "  Node editing: [bold]enabled[/bold]" if db_path else "  Node editing: read-only"
    print(f"  Graph viewer:  {url}")
    print(edit_note)
    print(f"  Press Ctrl+C to stop.")

    if open_browser:
        # Open after a short delay so the server is ready
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

