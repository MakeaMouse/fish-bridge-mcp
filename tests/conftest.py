"""Shared pytest fixtures for fish_bridge tests."""
from __future__ import annotations

import pytest
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / "fixtures"

SAMPLE_JSONL = FIXTURES_DIR / "sample-copilot-session.jsonl"


@pytest.fixture
def sample_jsonl_path() -> Path:
    assert SAMPLE_JSONL.exists(), f"Fixture missing: {SAMPLE_JSONL}"
    return SAMPLE_JSONL


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "fish_bridge_data"
    d.mkdir()
    return d


@pytest.fixture
def session_graph(tmp_data_dir: Path):
    from fish_bridge.graph.session import SessionGraph
    sg = SessionGraph.open("test-session", tmp_data_dir)
    yield sg
    sg.close()
