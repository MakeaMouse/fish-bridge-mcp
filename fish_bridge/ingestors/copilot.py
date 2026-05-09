"""CopilotTranscriptIngestor — reads VS Code Copilot Chat JSONL transcripts.

Auto-discovers the workspace hash by scanning workspaceStorage for directories
that contain GitHub.copilot-chat/transcripts/ or debug-logs/.  Cross-references
workspace.json to match the current working directory.

Supports:
  - One-shot batch ingest (ingest all turns in a session file)
  - Watch/tail mode (yields new turns as they appear in the JSONL)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterator

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.base import AbstractIngestor

logger = logging.getLogger(__name__)

# Subdirectories under GitHub.copilot-chat/ that may contain JSONL files.
# Listed in priority order; the first one with JSONL files wins.
_COPILOT_TRANSCRIPT_SUBDIRS: tuple[str, ...] = (
    "GitHub.copilot-chat/transcripts",   # original path (pre-v1.99)
    "GitHub.copilot-chat/debug-logs",    # observed in VS Code 1.99+
    "GitHub.copilot-chat/sessions",      # reserved for future VS Code versions
)


class CopilotTranscriptIngestor(AbstractIngestor):
    """Reads Copilot Chat JSONL transcript files from the local workspaceStorage."""

    # ------------------------------------------------------------------
    # Platform-aware storage root
    # ------------------------------------------------------------------

    @staticmethod
    def _storage_root() -> Path:
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "Code" / "User" / "workspaceStorage"
        elif sys.platform == "win32":
            appdata = os.environ.get("APPDATA", "")
            return Path(appdata) / "Code" / "User" / "workspaceStorage"
        else:  # Linux / XDG
            xdg = os.environ.get("XDG_CONFIG_HOME", "")
            base = Path(xdg) if xdg else Path.home() / ".config"
            return base / "Code" / "User" / "workspaceStorage"

    # ------------------------------------------------------------------
    # Copilot directory discovery (multi-path probe)
    # ------------------------------------------------------------------

    @classmethod
    def _find_copilot_dir(cls, workspace_entry: Path) -> Path | None:
        """Return the first transcript/session directory that contains JSONL files.

        Probes all known subdirectory paths so the ingestor survives VS Code
        internal path changes between releases.
        """
        for subdir in _COPILOT_TRANSCRIPT_SUBDIRS:
            candidate = workspace_entry / subdir
            if candidate.exists() and any(candidate.glob("*.jsonl")):
                return candidate
        return None

    # ------------------------------------------------------------------
    # Workspace discovery
    # ------------------------------------------------------------------

    @classmethod
    def find_workspace_hash(cls, workspace_path: str | Path | None = None) -> str | None:
        """Return the workspaceStorage hash that corresponds to workspace_path.

        If workspace_path is None, returns the hash of the most recently
        modified transcripts directory (best-guess for current workspace).
        """
        storage_root = cls._storage_root()
        if not storage_root.exists():
            return None

        candidates: list[tuple[float, str]] = []

        for entry in storage_root.iterdir():
            if not entry.is_dir():
                continue
            # Use the multi-path probe instead of a hardcoded transcripts/ path
            copilot_dir = cls._find_copilot_dir(entry)
            if copilot_dir is None:
                continue

            if workspace_path is not None:
                ws_json = entry / "workspace.json"
                if ws_json.exists():
                    try:
                        data = json.loads(ws_json.read_text(encoding="utf-8"))
                        # workspace.json can have folder or folders key
                        folders = data.get("folder") or ""
                        if isinstance(folders, str):
                            folders = [folders]
                        folders = data.get("folders", folders)
                        if isinstance(folders, list):
                            for f in folders:
                                uri = f.get("uri", f) if isinstance(f, dict) else f
                                # uri may be file:///path/to/dir
                                uri_path = uri.replace("file://", "").replace("file:", "")
                                if Path(uri_path).resolve() == Path(workspace_path).resolve():
                                    return entry.name
                    except (json.JSONDecodeError, OSError):
                        pass

            # Collect most-recently modified as fallback
            try:
                mtime = max(
                    (f.stat().st_mtime for f in copilot_dir.iterdir() if f.is_file()),
                    default=0.0,
                )
                candidates.append((mtime, entry.name))
            except OSError:
                pass

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Transcript discovery
    # ------------------------------------------------------------------

    @classmethod
    def find_transcripts(cls, workspace_hash: str) -> list[Path]:
        """Return all JSONL transcript paths for a workspace hash, newest first.

        Uses the multi-path probe so this keeps working when VS Code changes
        the internal directory name between releases.
        """
        entry = cls._storage_root() / workspace_hash
        copilot_dir = cls._find_copilot_dir(entry)
        if copilot_dir is None:
            return []
        files = sorted(
            copilot_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files

    @classmethod
    def latest_transcript(cls, workspace_hash: str) -> Path | None:
        files = cls.find_transcripts(workspace_hash)
        return files[0] if files else None

    # ------------------------------------------------------------------
    # JSONL parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_jsonl(path: Path, start_byte: int = 0) -> Iterator[dict]:
        """Yield parsed JSON objects from a JSONL file, starting at start_byte."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                if start_byte > 0:
                    fh.seek(start_byte)
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return

    @classmethod
    def parse_turns(cls, jsonl_path: Path) -> list[RawTurn]:
        """Extract user/assistant turn pairs from a JSONL transcript."""
        session_id = jsonl_path.stem  # file name without .jsonl

        # Collect messages grouped by turn
        pending_user: str = ""
        pending_assistant: str = ""
        turn_number = 0
        turns: list[RawTurn] = []
        ts = datetime.now(timezone.utc)
        _schema_warned = False  # emit at most one unknown-schema warning per file

        for record in cls._iter_jsonl(jsonl_path):
            event_type = record.get("type", "")

            # Schema version probe: detect and warn on unknown layouts.
            # v1 (current): payload nested under "data" key.
            # flat (hypothetical future): content at top level.
            if not _schema_warned and event_type and event_type not in (
                "session.start", "user.message", "assistant.message",
                "assistant.turn_start", "assistant.turn_end", "tool.execution_complete",
                "tool.execution_start",
            ):
                logger.warning(
                    "fish_bridge: unknown Copilot JSONL event type %r in %s — "
                    "VS Code may have changed the transcript schema. "
                    "Ingestion will continue, but some turns may be missed. "
                    "Please report this at https://github.com/fish-bridge/fish-bridge/issues",
                    event_type, jsonl_path.name,
                )
                _schema_warned = True

            # Real VS Code JSONL wraps payload inside a "data" dict;
            # fall back to top-level keys for older/different schemas.
            data = record.get("data") or {}
            if isinstance(data, str):
                # data may be a serialized dict string in some versions; try parsing
                try:
                    import ast as _ast
                    data = _ast.literal_eval(data)
                except Exception:
                    data = {}

            if event_type == "session.start":
                ts_raw = data.get("startTime") or record.get("startTime")
                if ts_raw:
                    try:
                        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        pass

            elif event_type == "user.message":
                # Flush the *previous* turn before starting the new one.
                # This strategy is required for agent-mode transcripts where
                # the assistant may cycle through multiple tool calls (each ending
                # with assistant.turn_end) before emitting a text response.
                # Flushing here ensures the complete user+assistant pair is
                # captured rather than splitting it across multiple RawTurns.
                if pending_user or pending_assistant:
                    turn_number += 1
                    turns.append(
                        RawTurn(
                            session_id=session_id,
                            turn_number=turn_number,
                            role_user=pending_user,
                            role_assistant=pending_assistant,
                            source="copilot_jsonl",
                            timestamp=ts,
                        )
                    )
                    pending_user = ""
                    pending_assistant = ""

                content = (data.get("content") or record.get("content") or "").strip()
                # Process file/image attachments (VS Code 1.99+ drag-drop and clip attachments)
                attachment_parts: list[str] = []
                for att in (data.get("attachments") or record.get("attachments") or []):
                    att_name = (
                        att.get("name") or att.get("fileName")
                        or att.get("uri") or "attachment"
                    )
                    att_content = att.get("content") or att.get("text") or ""
                    att_mime = (att.get("mimeType") or att.get("type") or "").lower()
                    att_kind = (att.get("kind") or "").lower()
                    if att_content and isinstance(att_content, str):
                        # Text / code / markdown attachment — inline its content
                        attachment_parts.append(
                            f"[Attached file: {att_name}]\n{att_content}"
                        )
                    elif att_mime.startswith("image/") or att_kind == "image" or any(
                        str(att_name).lower().endswith(ext)
                        for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
                    ):
                        # Image attachment — note the reference for multimodal backends
                        attachment_parts.append(f"[Attached image: {att_name}]")
                    elif att_name and att_name != "attachment":
                        # Unknown attachment type — note it so it appears in the turn
                        attachment_parts.append(f"[Attached: {att_name}]")

                if content or attachment_parts:
                    full_content = content
                    if attachment_parts:
                        full_content = (
                            (content + "\n\n" if content else "")
                            + "\n\n".join(attachment_parts)
                        )
                    pending_user = full_content.strip()

            elif event_type == "assistant.message":
                content = (data.get("content") or record.get("content") or "").strip()
                if content:
                    pending_assistant = content

            elif event_type == "assistant.turn_end":
                # Do NOT flush here — agent-mode sessions may have many
                # tool-call turn_end events before the actual text response
                # arrives.  The flush happens on the next user.message or
                # at end-of-file (see below).
                pass

        # Handle unclosed turn at end of file
        if pending_user or pending_assistant:
            turn_number += 1
            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_number=turn_number,
                    role_user=pending_user,
                    role_assistant=pending_assistant,
                    source="copilot_jsonl",
                    timestamp=ts,
                )
            )

        return turns

    # ------------------------------------------------------------------
    # AbstractIngestor implementation
    # ------------------------------------------------------------------

    def ingest(
        self,
        workspace_path: str | Path | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> list[RawTurn]:
        """Ingest all turns from the latest (or specified) Copilot transcript.

        Args:
            workspace_path: Path to the VS Code workspace folder.
                            If None, uses the most-recently-modified session.
            session_id:     Specific transcript file stem (without .jsonl).
                            If None, uses the latest transcript.
        """
        workspace_hash = self.find_workspace_hash(workspace_path)
        if workspace_hash is None:
            raise RuntimeError(
                "Could not find a workspaceStorage directory with Copilot transcripts. "
                "Make sure VS Code with GitHub Copilot is installed and has been used. "
                f"Searched subdirectories: {', '.join(_COPILOT_TRANSCRIPT_SUBDIRS)}"
            )

        if session_id is not None:
            # Try every known subdirectory for the specific session file
            entry = self._storage_root() / workspace_hash
            copilot_dir = self._find_copilot_dir(entry)
            if copilot_dir is not None:
                jsonl_path: Path | None = copilot_dir / f"{session_id}.jsonl"
                if not jsonl_path.exists():
                    jsonl_path = None
            else:
                jsonl_path = None
        else:
            jsonl_path = self.latest_transcript(workspace_hash)

        if jsonl_path is None or not jsonl_path.exists():
            return []

        return self.parse_turns(jsonl_path)

    # ------------------------------------------------------------------
    # Watch / tail mode
    # ------------------------------------------------------------------

    def watch(
        self,
        workspace_path: str | Path | None = None,
        session_id: str | None = None,
        poll_interval: float = 2.0,
    ) -> Generator[RawTurn, None, None]:
        """Tail a transcript JSONL and yield new turns as they appear.

        Tracks file size to detect appends; re-parses only new bytes.
        """
        workspace_hash = self.find_workspace_hash(workspace_path)
        if workspace_hash is None:
            raise RuntimeError("Could not find workspaceStorage with Copilot transcripts.")

        if session_id is not None:
            jsonl_path = (
                self._storage_root()
                / workspace_hash
                / "GitHub.copilot-chat"
                / "transcripts"
                / f"{session_id}.jsonl"
            )
        else:
            jsonl_path = self.latest_transcript(workspace_hash)

        if jsonl_path is None or not jsonl_path.exists():
            raise RuntimeError(f"Transcript file not found: {jsonl_path}")

        # Bootstrap: ingest all existing turns first
        already_seen = set[int]()
        for turn in self.parse_turns(jsonl_path):
            already_seen.add(turn.turn_number)
            yield turn

        last_size = jsonl_path.stat().st_size

        while True:
            time.sleep(poll_interval)
            try:
                current_size = jsonl_path.stat().st_size
            except OSError:
                continue

            if current_size <= last_size:
                continue

            # Re-parse; yield only turns we haven't seen before
            new_turns = self.parse_turns(jsonl_path)
            for turn in new_turns:
                if turn.turn_number not in already_seen:
                    already_seen.add(turn.turn_number)
                    yield turn

            last_size = current_size
