"""
conftest.py for tests/e2e/

Provides:
  - `restore_global_config` autouse fixture: backs up and restores
    ~/.fish_bridge/config.yaml so config-mutation tests leave no side effects.
  - `cfg` fixture: returns a path to a temp config.yaml with a valid (local) backend.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REAL_CONFIG = Path.home() / ".fish_bridge" / "config.yaml"

_GOOD_CONFIG_CONTENT = """\
extraction:
  backend: local
  gemini:
    model: models/gemini-2.5-flash
  claude:
    model: claude-opus-4-7
  openai:
    model: gpt-4.1-mini
  local:
    provider: ollama
    base_url: http://localhost:11434
    model: qwen2.5:7b
    embed_model: nomic-embed-text
"""


@pytest.fixture(autouse=True)
def restore_global_config():
    """Back up ~/.fish_bridge/config.yaml before each test and restore it after."""
    existed = _REAL_CONFIG.exists()
    backup: bytes | None = _REAL_CONFIG.read_bytes() if existed else None

    yield

    if existed and backup is not None:
        _REAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        _REAL_CONFIG.write_bytes(backup)
    elif not existed and _REAL_CONFIG.exists():
        _REAL_CONFIG.unlink()


@pytest.fixture()
def cfg(tmp_path: Path) -> Path:
    """Return a path to a temp config.yaml with a valid (local) backend."""
    p = tmp_path / "config.yaml"
    p.write_text(_GOOD_CONFIG_CONTENT)
    return p
