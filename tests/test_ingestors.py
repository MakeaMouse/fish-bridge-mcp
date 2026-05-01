"""Tests for CopilotTranscriptIngestor and ChatTurnIngestor."""
from __future__ import annotations

from pathlib import Path

import pytest

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.copilot import CopilotTranscriptIngestor
from fish_bridge.ingestors.chat import ChatTurnIngestor


# ---------------------------------------------------------------------------
# CopilotTranscriptIngestor
# ---------------------------------------------------------------------------

class TestCopilotTranscriptIngestor:

    def test_parse_turns_from_real_fixture(self, sample_jsonl_path: Path):
        """Should extract at least one turn from the real Copilot JSONL fixture."""
        turns = CopilotTranscriptIngestor.parse_turns(sample_jsonl_path)
        assert isinstance(turns, list)
        assert len(turns) > 0, "Expected at least one turn in the real fixture"

    def test_turn_fields_populated(self, sample_jsonl_path: Path):
        """Each turn should have a non-empty user or assistant message."""
        turns = CopilotTranscriptIngestor.parse_turns(sample_jsonl_path)
        for turn in turns:
            assert isinstance(turn, RawTurn)
            assert turn.session_id == sample_jsonl_path.stem
            assert turn.turn_number > 0
            assert turn.role_user or turn.role_assistant, "Turn must have at least one message"
            assert turn.source == "copilot_jsonl"

    def test_turn_numbers_are_sequential(self, sample_jsonl_path: Path):
        turns = CopilotTranscriptIngestor.parse_turns(sample_jsonl_path)
        numbers = [t.turn_number for t in turns]
        assert numbers == list(range(1, len(numbers) + 1))

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        turns = CopilotTranscriptIngestor.parse_turns(empty)
        assert turns == []

    def test_malformed_json_lines_are_skipped(self, tmp_path: Path):
        """Malformed JSONL lines should not raise; valid lines should parse."""
        jl = tmp_path / "test.jsonl"
        jl.write_text(
            '{"type":"session.start","sessionId":"s1","startTime":"2026-01-01T00:00:00Z"}\n'
            'THIS IS NOT JSON\n'
            '{"type":"user.message","content":"Hello world"}\n'
            '{"type":"assistant.message","content":"Hi there"}\n'
            '{"type":"assistant.turn_end","turnId":"t1"}\n',
            encoding="utf-8",
        )
        turns = CopilotTranscriptIngestor.parse_turns(jl)
        assert len(turns) == 1
        assert turns[0].role_user == "Hello world"
        assert turns[0].role_assistant == "Hi there"

    def test_storage_root_returns_path(self):
        root = CopilotTranscriptIngestor._storage_root()
        assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# ChatTurnIngestor
# ---------------------------------------------------------------------------

class TestChatTurnIngestor:

    def test_parse_plain_text_exchange(self):
        text = (
            "User: How should I handle Redis caching?\n\n"
            "Assistant: Use ElastiCache with a 24hr TTL.\n\n"
            "User: What about memory limits?\n\n"
            "Assistant: Set maxmemory-policy allkeys-lru."
        )
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(text=text, session_id="test")
        assert len(turns) == 2
        assert turns[0].role_user == "How should I handle Redis caching?"
        assert "ElastiCache" in turns[0].role_assistant
        assert turns[1].role_user == "What about memory limits?"

    def test_parse_claude_json_export(self, tmp_path: Path):
        import json
        export = [
            {"role": "user",      "content": "What is WAL mode in SQLite?"},
            {"role": "assistant", "content": "WAL stands for Write-Ahead Logging..."},
            {"role": "user",      "content": "When should I use it?"},
            {"role": "assistant", "content": "Use it for concurrent read/write workloads."},
        ]
        p = tmp_path / "export.json"
        p.write_text(json.dumps(export), encoding="utf-8")
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(file_path=p, session_id="test")
        assert len(turns) == 2
        assert "WAL" in turns[0].role_user
        assert "Write-Ahead" in turns[0].role_assistant

    def test_empty_text_returns_empty_list(self):
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(text="", session_id="test")
        assert turns == []

    def test_unstructured_text_becomes_single_turn(self):
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(text="Just some random text with no structure.", session_id="test")
        assert len(turns) == 1
        assert turns[0].role_user == "Just some random text with no structure."
