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
# Model defaults per backend.
# ---------------------------------------------------------------------------

MODEL_DEFAULTS: dict[str, str] = {
    "claude": "claude-opus-4-7",           # latest stable as of May 2026
    "openai": "gpt-4.1-mini",
    "gemini": "models/gemini-2.5-flash",   # models/ prefix required by new API keys
    "local":  "qwen2.5:7b",
}

# ---------------------------------------------------------------------------
# File paths that trigger a deprecation warning.
# ---------------------------------------------------------------------------
FILE_PATH_DEPRECATIONS: dict[str, str] = {
    # VS Code: the old "shared" default before delivery modes were added
    ".github/copilot-instructions.md": (
        "Using .github/copilot-instructions.md as local_context_file mixes session data "
        "into a committed file. Switch to delivery_mode: local (default) or set "
        "delivery_mode: shared if you intentionally want to commit the context."
    ),
}

# Known stale/deprecated model names → (replacement, reason).
# Any user config containing a key here triggers a startup warning.
MODEL_DEPRECATIONS: dict[str, tuple[str, str]] = {
    # Gemini
    "gemini-2.0-flash":            ("models/gemini-2.5-flash", "Returns 404 for new API keys; use models/ prefix"),
    "gemini-1.5-pro":              ("models/gemini-2.5-flash", "Deprecated May 2026"),
    "gemini-1.5-flash":            ("models/gemini-2.5-flash", "Deprecated May 2026"),
    # Claude
    "claude-sonnet-4-6":           ("claude-opus-4-7",         "Replaced by claude-opus-4-7 (GA Apr 16 2026)"),
    "claude-3-5-sonnet-20241022":  ("claude-opus-4-7",         "Replaced by claude-opus-4-7 (GA Apr 16 2026)"),
    "claude-3-7-sonnet-20250219":  ("claude-opus-4-7",         "Replaced by claude-opus-4-7 (GA Apr 16 2026)"),
    # OpenAI / Copilot
    "gpt-4.1":                     ("gpt-4.1-mini",            "Use gpt-4.1-mini for cost efficiency"),
    "gpt-5.1":                     ("gpt-4.1-mini",            "Deprecated Apr 3 2026 in Copilot"),
    "gpt-5.2":                     ("gpt-4.1-mini",            "Deprecated May 1 2026 in Copilot"),
    "gpt-5.3":                     ("gpt-4.1-mini",            "Deprecated Apr 27 2026 in Copilot"),
}

# Cloud backends that send turn text to external APIs — shown in PII warning.
CLOUD_BACKENDS: frozenset[str] = frozenset({"claude", "openai", "gemini", "openai-compatible"})

# Default PII masking patterns applied before any extraction backend sees text.
# These are best-effort; they do not guarantee full PII removal.
_DEFAULT_EXCLUDE_PATTERNS: list[str] = [
    r"(?i)(password|passwd|secret|token|api[_\-]?key)\s*[=:]\s*\S+",
    r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*",
    r"AKIA[0-9A-Z]{16}",                   # AWS Access Key ID prefix
    r"(?i)postgres(?:ql)?://[^\s\"']+",    # PostgreSQL connection strings
    r"(?i)mysql://[^\s\"']+",              # MySQL connection strings
    r"(?i)mongodb\+srv://[^\s\"']+",       # MongoDB Atlas SRV URIs
]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClaudeConfig:
    model:       str = field(default_factory=lambda: MODEL_DEFAULTS["claude"])
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class OpenAIConfig:
    model:       str = field(default_factory=lambda: MODEL_DEFAULTS["openai"])
    api_key_env: str = "OPENAI_API_KEY"
    base_url:    str | None = None


@dataclass
class GeminiConfig:
    model:       str = field(default_factory=lambda: MODEL_DEFAULTS["gemini"])
    api_key_env: str = "GEMINI_API_KEY"
    # Gemini exposes an OpenAI-compatible endpoint; no separate SDK needed.
    # Requires the models/ prefix: e.g. models/gemini-2.5-flash.
    base_url:    str = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class LocalConfig:
    provider:    str = "ollama"
    base_url:    str = "http://localhost:11434"
    model:       str = field(default_factory=lambda: MODEL_DEFAULTS["local"])
    embed_model: str = "nomic-embed-text"


@dataclass
class HybridConfig:
    realtime_backend:      str = "local"   # backend name for every-turn extraction
    consolidation_backend: str = "gemini"  # backend name for quality consolidation
    consolidation_every_n: int = 10        # run consolidation every N turns


@dataclass
class DedupConfig:
    merge_threshold:        float = 0.88   # cosine similarity > this → merge nodes
    relate_threshold:       float = 0.70   # similarity in (relate, merge) → relates-to edge
    # "manual" — require fish-bridge conflict --resolve
    # "newer"  — auto-adopt the more recent status, log reversal in status_history
    # "older"  — keep the older status, log the incoming change
    auto_resolve_conflicts: str   = "manual"


@dataclass
class ExtractionConfig:
    backend:          str = "claude"   # claude | openai | gemini | local | hybrid | openai-compatible
    claude:           ClaudeConfig  = field(default_factory=ClaudeConfig)
    openai:           OpenAIConfig  = field(default_factory=OpenAIConfig)
    gemini:           GeminiConfig  = field(default_factory=GeminiConfig)
    local:            LocalConfig   = field(default_factory=LocalConfig)
    hybrid:           HybridConfig  = field(default_factory=HybridConfig)
    exclude_patterns: list[str]     = field(default_factory=lambda: list(_DEFAULT_EXCLUDE_PATTERNS))


@dataclass
class OutputConfig:
    # delivery_mode controls where the compiled context block is written:
    #   "local"  — .fish_bridge/context.md  (gitignored, per-developer, default)
    #   "shared" — .github/copilot-instructions.md or CLAUDE.md (committed, team-wide)
    delivery_mode:       str = "local"
    local_context_file:  str = ".fish_bridge/context.md"
    shared_context_file: str = ".github/copilot-instructions.md"
    token_budget:        int = 800

    # Additional output targets. When non-empty, compile writes to each path
    # in addition to the primary default_file.
    extra_targets:       list[str] = field(default_factory=list)

    @property
    def default_file(self) -> str:
        if self.delivery_mode == "shared":
            return self.shared_context_file
        return self.local_context_file

    @property
    def all_output_files(self) -> list[str]:
        """Return every file that should be written on compile.

        The primary default_file is always first; extra_targets follow.
        Duplicates are removed while preserving order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for p in [self.default_file] + list(self.extra_targets):
            if p not in seen:
                seen.add(p)
                result.append(p)
        return result


@dataclass
class FishBridgeConfig:
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output:     OutputConfig     = field(default_factory=OutputConfig)
    dedup:      DedupConfig      = field(default_factory=DedupConfig)
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
    dedup_raw      = raw.get("dedup", {})

    claude_raw  = extraction_raw.get("claude",  {})
    openai_raw  = extraction_raw.get("openai",  {})
    gemini_raw  = extraction_raw.get("gemini",  {})
    local_raw   = extraction_raw.get("local",   {})
    hybrid_raw  = extraction_raw.get("hybrid",  {})

    # User-supplied exclude_patterns extend (not replace) the defaults.
    user_patterns: list[str] = extraction_raw.get("exclude_patterns", [])
    merged_patterns = list(_DEFAULT_EXCLUDE_PATTERNS) + [
        p for p in user_patterns if p not in _DEFAULT_EXCLUDE_PATTERNS
    ]

    return FishBridgeConfig(
        extraction=ExtractionConfig(
            backend=extraction_raw.get("backend", "claude"),
            claude=ClaudeConfig(
                model=       claude_raw.get("model",       MODEL_DEFAULTS["claude"]),
                api_key_env= claude_raw.get("api_key_env", "ANTHROPIC_API_KEY"),
            ),
            openai=OpenAIConfig(
                model=       openai_raw.get("model",       MODEL_DEFAULTS["openai"]),
                api_key_env= openai_raw.get("api_key_env", "OPENAI_API_KEY"),
                base_url=    openai_raw.get("base_url"),
            ),
            gemini=GeminiConfig(
                model=       gemini_raw.get("model",       MODEL_DEFAULTS["gemini"]),
                api_key_env= gemini_raw.get("api_key_env", "GEMINI_API_KEY"),
                base_url=    gemini_raw.get("base_url",    "https://generativelanguage.googleapis.com/v1beta/openai/"),
            ),
            local=LocalConfig(
                provider=    local_raw.get("provider",    "ollama"),
                base_url=    local_raw.get("base_url",    "http://localhost:11434"),
                model=       local_raw.get("model",       MODEL_DEFAULTS["local"]),
                embed_model= local_raw.get("embed_model", "nomic-embed-text"),
            ),
            hybrid=HybridConfig(
                realtime_backend=      hybrid_raw.get("realtime_backend",      "local"),
                consolidation_backend= hybrid_raw.get("consolidation_backend", "gemini"),
                consolidation_every_n= hybrid_raw.get("consolidation_every_n", 10),
            ),
            exclude_patterns=merged_patterns,
        ),
        output=OutputConfig(
            delivery_mode=       output_raw.get("delivery_mode",       "local"),
            local_context_file=  output_raw.get("local_context_file",  ".fish_bridge/context.md"),
            shared_context_file= output_raw.get("shared_context_file", ".github/copilot-instructions.md"),
            token_budget=        output_raw.get("token_budget",        800),
            extra_targets=       list(output_raw.get("extra_targets",  [])),
        ),
        dedup=DedupConfig(
            merge_threshold=        dedup_raw.get("merge_threshold",        0.88),
            relate_threshold=       dedup_raw.get("relate_threshold",       0.70),
            auto_resolve_conflicts= dedup_raw.get("auto_resolve_conflicts", "manual"),
        ),
        data_dir=raw.get("data_dir", "~/.fish_bridge/sessions"),
    )


def get_data_dir(config: FishBridgeConfig) -> Path:
    return Path(config.data_dir).expanduser()


def check_file_path_deprecations(config: FishBridgeConfig) -> list[tuple[str, str]]:
    """Return a list of (field_name, warning_message) for any deprecated file paths
    found in the user's output config.

    Currently warns when local_context_file is set to a path that belongs in
    delivery_mode=shared (e.g. the old default .github/copilot-instructions.md).
    """
    warnings: list[tuple[str, str]] = []
    if (
        config.output.delivery_mode == "local"
        and config.output.local_context_file in FILE_PATH_DEPRECATIONS
    ):
        msg = FILE_PATH_DEPRECATIONS[config.output.local_context_file]
        warnings.append(("output.local_context_file", msg))
    return warnings


def _write_config_output(
    extra_targets: list[str] | None = None,
    delivery_mode: str | None = None,
    shared_context_file: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Persist output-section changes back to config.yaml (A1 persistence)."""
    target = config_path or _default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        try:
            raw: dict = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            raw = {}
    else:
        raw = {}

    out = raw.setdefault("output", {})
    if extra_targets is not None:
        out["extra_targets"] = extra_targets
    if delivery_mode is not None:
        out["delivery_mode"] = delivery_mode
    if shared_context_file is not None:
        out["shared_context_file"] = shared_context_file

    target.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")


# Public alias (callers import write_config_output directly)
write_config_output = _write_config_output


def check_model_staleness(config: FishBridgeConfig) -> list[tuple[str, str, str]]:
    """Return a list of (backend, stale_model, suggested_fix) for any deprecated
    model names found in the user's active config.

    Only checks the active backend, so a stale openai.model doesn't warn when
    backend=claude.
    """
    warnings: list[tuple[str, str, str]] = []
    backend = config.extraction.backend

    checks: list[tuple[str, str]] = []
    if backend in ("claude", "hybrid"):
        checks.append(("claude", config.extraction.claude.model))
    if backend in ("openai", "openai-compatible", "hybrid"):
        checks.append(("openai", config.extraction.openai.model))
    if backend in ("gemini",):
        checks.append(("gemini", config.extraction.gemini.model))
    if backend in ("local", "hybrid"):
        # Local models can't be "deprecated" in the same way, but flag old embed models
        pass

    for backend_name, model_name in checks:
        if model_name in MODEL_DEPRECATIONS:
            replacement, reason = MODEL_DEPRECATIONS[model_name]
            warnings.append((backend_name, model_name, f"{reason} → use '{replacement}'"))

    return warnings


def get_active_session_id(
    workspace_path: Path | None = None,
    config: FishBridgeConfig | None = None,
) -> str:
    """Return the active session ID for a workspace.

    Resolution order:
      1. Read from <workspace>/.fish_bridge/session.lock  (if it exists)
      2. Derive from <workspace-basename>-<YYYY-MM-DD>
      3. Fall back to 'default-<YYYY-MM-DD>' if workspace_path is None

    The resolved ID is persisted to session.lock so subsequent calls are stable.
    """
    import datetime as _dt

    today = _dt.date.today().isoformat()

    if workspace_path is None:
        workspace_path = Path.cwd()

    lock_dir  = workspace_path / ".fish_bridge"
    lock_file = lock_dir / "session.lock"

    if lock_file.exists():
        session_id = lock_file.read_text(encoding="utf-8").strip()
        if session_id:
            return session_id

    # Derive from workspace basename + today
    basename   = workspace_path.name.lower().replace(" ", "-") or "default"
    session_id = f"{basename}-{today}"

    # Persist for future calls
    try:
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(session_id, encoding="utf-8")
    except OSError:
        pass  # non-fatal — just return the derived ID

    return session_id


def build_backend(config: FishBridgeConfig):
    """Instantiate the configured extraction backend."""
    from fish_bridge.extraction.base import warn_cloud_backend_once
    backend_name = config.extraction.backend
    if backend_name == "claude":
        warn_cloud_backend_once("claude")
        from fish_bridge.extraction.claude import ClaudeBackend
        return ClaudeBackend(
            model=       config.extraction.claude.model,
            api_key_env= config.extraction.claude.api_key_env,
        )
    elif backend_name in ("openai", "openai-compatible"):
        warn_cloud_backend_once(backend_name)
        from fish_bridge.extraction.openai import OpenAIBackend
        return OpenAIBackend(
            model=       config.extraction.openai.model,
            api_key_env= config.extraction.openai.api_key_env,
            base_url=    config.extraction.openai.base_url,
        )
    elif backend_name == "gemini":
        warn_cloud_backend_once("gemini")
        from fish_bridge.extraction.openai import OpenAIBackend
        # Gemini exposes an OpenAI-compatible endpoint — reuse OpenAIBackend
        return OpenAIBackend(
            model=       config.extraction.gemini.model,
            api_key_env= config.extraction.gemini.api_key_env,
            base_url=    config.extraction.gemini.base_url,
        )
    elif backend_name == "local":
        from fish_bridge.extraction.local import OllamaBackend
        return OllamaBackend(
            model=       config.extraction.local.model,
            base_url=    config.extraction.local.base_url,
            embed_model= config.extraction.local.embed_model,
        )
    elif backend_name == "hybrid":
        from fish_bridge.extraction.hybrid import HybridBackend
        # Temporarily override backend to build each sub-backend
        import copy
        realtime_cfg = copy.deepcopy(config)
        realtime_cfg.extraction.backend = config.extraction.hybrid.realtime_backend
        cloud_cfg = copy.deepcopy(config)
        cloud_cfg.extraction.backend = config.extraction.hybrid.consolidation_backend
        return HybridBackend(
            realtime_backend=      build_backend(realtime_cfg),
            consolidation_backend= build_backend(cloud_cfg),
            consolidation_every_n= config.extraction.hybrid.consolidation_every_n,
        )
    else:
        raise ValueError(
            f"Unknown backend {backend_name!r}. "
            "Valid values: claude | openai | gemini | local | hybrid | openai-compatible"
        )
