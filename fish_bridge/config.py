"""Configuration loader for fish_bridge.

Loads ~/.fish_bridge/config.yaml (or a path given by FISH_BRIDGE_CONFIG env var).
Falls back to sensible defaults if no config file exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClaudeConfig:
    model:       str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class OpenAIConfig:
    model:       str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url:    str | None = None


@dataclass
class LocalConfig:
    provider:    str = "ollama"
    base_url:    str = "http://localhost:11434"
    model:       str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"


@dataclass
class ExtractionConfig:
    backend:          str = "claude"   # claude | openai | local | openai-compatible
    claude:           ClaudeConfig = field(default_factory=ClaudeConfig)
    openai:           OpenAIConfig = field(default_factory=OpenAIConfig)
    local:            LocalConfig  = field(default_factory=LocalConfig)
    exclude_patterns: list[str]    = field(default_factory=list)


@dataclass
class OutputConfig:
    default_file:  str = ".github/copilot-instructions.md"
    token_budget:  int = 800


@dataclass
class FishBridgeConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output:     OutputConfig     = field(default_factory=OutputConfig)
    data_dir:   str              = "~/.fish_bridge/sessions"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    env = os.environ.get("FISH_BRIDGE_CONFIG")
    if env:
        return Path(env)
    return Path.home() / ".fish_bridge" / "config.yaml"


def load_config(path: Path | None = None) -> FishBridgeConfig:
    """Load config from YAML file.  Missing file → all defaults."""
    config_path = path or _default_config_path()

    if not config_path.exists():
        return FishBridgeConfig()

    try:
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return FishBridgeConfig()

    extraction_raw = raw.get("extraction", {})
    output_raw     = raw.get("output", {})

    claude_raw = extraction_raw.get("claude", {})
    openai_raw = extraction_raw.get("openai", {})
    local_raw  = extraction_raw.get("local",  {})

    return FishBridgeConfig(
        extraction=ExtractionConfig(
            backend=extraction_raw.get("backend", "claude"),
            claude=ClaudeConfig(
                model=       claude_raw.get("model",       "claude-sonnet-4-6"),
                api_key_env= claude_raw.get("api_key_env", "ANTHROPIC_API_KEY"),
            ),
            openai=OpenAIConfig(
                model=       openai_raw.get("model",       "gpt-4.1-mini"),
                api_key_env= openai_raw.get("api_key_env", "OPENAI_API_KEY"),
                base_url=    openai_raw.get("base_url"),
            ),
            local=LocalConfig(
                provider=    local_raw.get("provider",    "ollama"),
                base_url=    local_raw.get("base_url",    "http://localhost:11434"),
                model=       local_raw.get("model",       "qwen2.5:7b"),
                embed_model= local_raw.get("embed_model", "nomic-embed-text"),
            ),
            exclude_patterns=extraction_raw.get("exclude_patterns", []),
        ),
        output=OutputConfig(
            default_file= output_raw.get("default_file",  ".github/copilot-instructions.md"),
            token_budget= output_raw.get("token_budget",  800),
        ),
        data_dir=raw.get("data_dir", "~/.fish_bridge/sessions"),
    )


def get_data_dir(config: FishBridgeConfig) -> Path:
    return Path(config.data_dir).expanduser()


def build_backend(config: FishBridgeConfig):
    """Instantiate the configured extraction backend."""
    backend_name = config.extraction.backend
    if backend_name == "claude":
        from fish_bridge.extraction.claude import ClaudeBackend
        return ClaudeBackend(
            model=       config.extraction.claude.model,
            api_key_env= config.extraction.claude.api_key_env,
        )
    elif backend_name in ("openai", "openai-compatible"):
        from fish_bridge.extraction.openai import OpenAIBackend
        return OpenAIBackend(
            model=       config.extraction.openai.model,
            api_key_env= config.extraction.openai.api_key_env,
            base_url=    config.extraction.openai.base_url,
        )
    else:
        raise ValueError(
            f"Unknown backend {backend_name!r}. Valid values: claude, openai, openai-compatible. "
            "Local Ollama backend is added in Phase 1."
        )
