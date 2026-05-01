"""ChatTurnIngestor — paste or file-based chat turn ingestor.

Handles:
  - Plain text pasted via --paste (opens $EDITOR)
  - Claude JSON export files (--file claude-export.json)
  - Generic text files with user/assistant delimiters
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.base import AbstractIngestor


class ChatTurnIngestor(AbstractIngestor):
    """Ingest chat turns from pasted text or exported files."""

    # ------------------------------------------------------------------
    # AbstractIngestor
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str | None = None,
        file_path: Path | str | None = None,
        session_id: str = "pasted",
        **kwargs,
    ) -> list[RawTurn]:
        """Parse turns from either raw text or a file path.

        Args:
            text:       Raw chat text (user/assistant pairs).
            file_path:  Path to a file (Claude JSON export, plain text).
            session_id: Session identifier for produced RawTurns.
        """
        if file_path is not None:
            return self._from_file(Path(file_path), session_id)
        if text is not None:
            return self._from_text(text, session_id)
        return []

    # ------------------------------------------------------------------
    # File-based ingestion
    # ------------------------------------------------------------------

    def _from_file(self, path: Path, session_id: str) -> list[RawTurn]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Try Claude JSON export first
        try:
            data = json.loads(raw)
            turns = self._parse_claude_json(data, session_id)
            if turns:
                return turns
        except (json.JSONDecodeError, KeyError):
            pass
        # Fall back to plain-text delimiter parsing
        return self._from_text(raw, session_id)

    @staticmethod
    def _parse_claude_json(data: dict | list, session_id: str) -> list[RawTurn]:
        """Parse Claude's JSON export format.

        Claude exports as either:
          - list of {"role": "user"|"assistant", "content": "..."}
          - dict with "messages" key containing the above list
        """
        messages: list[dict] = []
        if isinstance(data, list):
            messages = data
        elif isinstance(data, dict):
            messages = data.get("messages", [])

        turns: list[RawTurn] = []
        i = 0
        turn_number = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = ""
            raw_content = msg.get("content", "")
            if isinstance(raw_content, str):
                content = raw_content.strip()
            elif isinstance(raw_content, list):
                # Claude sometimes uses content blocks
                content = " ".join(
                    block.get("text", "") for block in raw_content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()

            if role == "user" and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.get("role") == "assistant":
                    asst_content = next_msg.get("content", "")
                    if isinstance(asst_content, list):
                        asst_content = " ".join(
                            b.get("text", "") for b in asst_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ).strip()
                    turn_number += 1
                    turns.append(
                        RawTurn(
                            session_id=session_id,
                            turn_number=turn_number,
                            role_user=content,
                            role_assistant=str(asst_content).strip(),
                            source="claude_json",
                        )
                    )
                    i += 2
                    continue
            i += 1

        return turns

    # ------------------------------------------------------------------
    # Plain-text delimiter parsing
    # ------------------------------------------------------------------

    # Patterns for common AI chat export formats
    _USER_PATTERNS = [
        re.compile(r"^(?:You|User|Human):\s*", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#+\s*(?:You|User|Human)\s*$", re.IGNORECASE | re.MULTILINE),
    ]
    _ASST_PATTERNS = [
        re.compile(r"^(?:Assistant|Claude|Copilot|GPT|AI):\s*", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^#+\s*(?:Assistant|Claude|Copilot|GPT|AI)\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    def _from_text(self, text: str, session_id: str) -> list[RawTurn]:
        """Split plain text into user/assistant turns using delimiter patterns."""
        # Try to split by "User:" / "Assistant:" style markers
        segments = re.split(
            r"\n(?=(?:You|User|Human|Assistant|Claude|Copilot|GPT|AI)\s*:)",
            text,
            flags=re.IGNORECASE,
        )

        turns: list[RawTurn] = []
        turn_number = 0
        pending_user = ""
        pending_asst = ""

        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if re.match(r"^(?:You|User|Human)\s*:", seg, re.IGNORECASE):
                content = re.sub(r"^(?:You|User|Human)\s*:\s*", "", seg, flags=re.IGNORECASE)
                if pending_user and pending_asst:
                    turn_number += 1
                    turns.append(
                        RawTurn(
                            session_id=session_id,
                            turn_number=turn_number,
                            role_user=pending_user.strip(),
                            role_assistant=pending_asst.strip(),
                            source="paste",
                        )
                    )
                    pending_asst = ""
                pending_user = content.strip()
            elif re.match(r"^(?:Assistant|Claude|Copilot|GPT|AI)\s*:", seg, re.IGNORECASE):
                content = re.sub(
                    r"^(?:Assistant|Claude|Copilot|GPT|AI)\s*:\s*", "", seg, flags=re.IGNORECASE
                )
                pending_asst = content.strip()
            else:
                # Unrecognised — append to pending_asst or treat as a single-turn
                if pending_user:
                    pending_asst += "\n" + seg
                else:
                    pending_user = seg

        # Flush
        if pending_user:
            turn_number += 1
            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_number=turn_number,
                    role_user=pending_user.strip(),
                    role_assistant=pending_asst.strip(),
                    source="paste",
                )
            )

        # If no structure found, treat whole text as a single user turn
        if not turns and text.strip():
            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_number=1,
                    role_user=text.strip(),
                    role_assistant="",
                    source="paste",
                )
            )

        return turns

    # ------------------------------------------------------------------
    # Interactive paste helper (opens $EDITOR)
    # ------------------------------------------------------------------

    @classmethod
    def open_editor(cls) -> str:
        """Open $EDITOR (or 'vi') for the user to paste chat text.
        Returns the typed/pasted content.
        """
        import os
        editor = os.environ.get("EDITOR", "vi")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tf:
            tf.write("# Paste your chat exchange below, then save and quit.\n\n")
            tmp_path = tf.name
        try:
            subprocess.run([editor, tmp_path], check=True)
            content = Path(tmp_path).read_text(encoding="utf-8")
            # Strip comment header
            lines = [l for l in content.splitlines() if not l.startswith("#")]
            return "\n".join(lines).strip()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
