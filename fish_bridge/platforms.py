"""Platform adapter registry for fish_bridge.

Maps AI tool / IDE names → the output files they read, ingest capabilities,
and setup notes.  Adding support for a new IDE is a single dict entry here;
no changes are needed in cli.py or config.py.

Usage:
    from fish_bridge.platforms import get_adapter, PLATFORM_ADAPTERS, list_tools

    adapter = get_adapter("jetbrains")
    print(adapter.output_targets)   # [".github/copilot-instructions.md"]
    print(adapter.mcp_supported)    # True
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformAdapter:
    """Describes how fish_bridge should deliver context to a specific AI tool."""

    # Human-readable display name shown in CLI output
    display_name: str

    # Canonical output files for this platform, in preference order.
    # The FIRST entry is the primary; subsequent entries are written
    # when multi_target=True in OutputConfig.
    output_targets: tuple[str, ...]

    # Whether VS Code-style nested Markdown link resolution works
    # (i.e., a link inside .github/copilot-instructions.md → .fish_bridge/context.md
    # is auto-followed). Only true for VS Code 1.99+.
    nested_link_supported: bool = False

    # Whether an MCP client is known to be available for this tool
    mcp_supported: bool = False

    # Known JSONL transcript source name for auto-discovery.
    # None = no automatic ingest; user must use --paste or --file.
    ingest_source: str | None = None

    # Whether delivery_mode="local" (gitignored) is safe here.
    # False means the primary output file must be committed to the repo.
    local_mode_safe: bool = True

    # Short note printed to the user after `fish-bridge init --tool <name>`
    setup_note: str = ""

    # Aliases that map to this adapter (e.g. "idea" → "jetbrains")
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Keys are canonical tool names (lower-case, hyphen-separated).
# Entries MUST remain sorted alphabetically for readability.

PLATFORM_ADAPTERS: dict[str, PlatformAdapter] = {
    # ------------------------------------------------------------------
    # GitHub Copilot in VS Code (primary target, all modes)
    # ------------------------------------------------------------------
    "copilot": PlatformAdapter(
        display_name="GitHub Copilot (VS Code)",
        output_targets=(".fish_bridge/context.md",),
        nested_link_supported=True,
        mcp_supported=True,
        ingest_source="copilot",
        local_mode_safe=True,
        setup_note=(
            "Context written to .fish_bridge/context.md (gitignored).\n"
            "VS Code 1.99+ picks it up via the nested link in .github/copilot-instructions.md.\n"
            "MCP server optional: fish-bridge serve-mcp"
        ),
        aliases=("vscode", "vs-code"),
    ),

    # ------------------------------------------------------------------
    # GitHub Copilot in JetBrains IDEs (IntelliJ IDEA, PyCharm, etc.)
    # Reads .github/copilot-instructions.md and .github/instructions/*.
    # AGENTS.md / CLAUDE.md / GEMINI.md supported in cloud agent mode.
    # ------------------------------------------------------------------
    "jetbrains": PlatformAdapter(
        display_name="GitHub Copilot (JetBrains)",
        output_targets=(".github/copilot-instructions.md",),
        nested_link_supported=False,  # no nested link support
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,  # file must be committed (or delivery_mode=shared)
        setup_note=(
            "Context written directly to .github/copilot-instructions.md.\n"
            "Commit this file so JetBrains Copilot can read it.\n"
            "JetBrains Copilot Chat does not write transcript files to disk.\n"
            "To ingest a conversation: fish-bridge ingest --source jetbrains\n"
            "  (This opens your editor; copy chat from the JetBrains panel, paste, save & quit.)\n"
            "MCP (agent mode): add fish-bridge to your JetBrains MCP config."
        ),
        aliases=("idea", "pycharm", "webstorm", "intellij"),
    ),

    # ------------------------------------------------------------------
    # GitHub Copilot in Eclipse
    # ------------------------------------------------------------------
    "eclipse": PlatformAdapter(
        display_name="GitHub Copilot (Eclipse)",
        output_targets=(".github/copilot-instructions.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to .github/copilot-instructions.md.\n"
            "Commit this file so Eclipse Copilot can read it.\n"
            "No JSONL auto-ingest — use: fish-bridge ingest --paste"
        ),
    ),

    # ------------------------------------------------------------------
    # GitHub Copilot in Xcode
    # ------------------------------------------------------------------
    "xcode": PlatformAdapter(
        display_name="GitHub Copilot (Xcode)",
        output_targets=(".github/copilot-instructions.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to .github/copilot-instructions.md.\n"
            "Commit this file so Xcode Copilot can read it.\n"
            "No JSONL auto-ingest — use: fish-bridge ingest --paste"
        ),
    ),

    # ------------------------------------------------------------------
    # GitHub Copilot in Visual Studio (Windows)
    # ------------------------------------------------------------------
    "visual-studio": PlatformAdapter(
        display_name="GitHub Copilot (Visual Studio)",
        output_targets=(".github/copilot-instructions.md",),
        nested_link_supported=False,
        mcp_supported=False,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to .github/copilot-instructions.md.\n"
            "Commit this file so Visual Studio Copilot can read it.\n"
            "No JSONL auto-ingest — use: fish-bridge ingest --paste"
        ),
        aliases=("vs",),
    ),

    # ------------------------------------------------------------------
    # GitHub Copilot CLI
    # Reads .github/copilot-instructions.md and AGENTS.md.
    # ------------------------------------------------------------------
    "copilot-cli": PlatformAdapter(
        display_name="GitHub Copilot CLI",
        output_targets=("AGENTS.md", ".github/copilot-instructions.md"),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to AGENTS.md (primary) and .github/copilot-instructions.md.\n"
            "Both files should be committed to the repo.\n"
            "MCP: fish-bridge serve-mcp"
        ),
        aliases=("gh-copilot",),
    ),

    # ------------------------------------------------------------------
    # Codex (github.com/codex, GitHub cloud agent)
    # Reads .github/copilot-instructions.md, AGENTS.md, CLAUDE.md, GEMINI.md.
    # ------------------------------------------------------------------
    "codex": PlatformAdapter(
        display_name="GitHub Codex / Copilot cloud agent",
        output_targets=("AGENTS.md", ".github/copilot-instructions.md"),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to AGENTS.md (primary) and .github/copilot-instructions.md.\n"
            "Commit both files. Codex reads AGENTS.md nearest the file being edited.\n"
            "MCP: configure fish-bridge in .vscode/mcp.json or repo-level MCP config."
        ),
        aliases=("cloud-agent",),
    ),

    # ------------------------------------------------------------------
    # Claude Code (Anthropic CLI)
    # Reads CLAUDE.md at repo root; also respects AGENTS.md.
    # ------------------------------------------------------------------
    "claude": PlatformAdapter(
        display_name="Claude Code (Anthropic CLI)",
        output_targets=("CLAUDE.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=True,
        setup_note=(
            "Context written to CLAUDE.md.\n"
            "Claude Code reads CLAUDE.md automatically at session start.\n"
            "MCP: add fish-bridge to ~/.claude/claude_desktop_config.json"
        ),
        aliases=("claude-code",),
    ),

    # ------------------------------------------------------------------
    # Cursor IDE
    # Reads .cursorrules, AGENTS.md, CLAUDE.md.
    # ------------------------------------------------------------------
    "cursor": PlatformAdapter(
        display_name="Cursor",
        output_targets=("AGENTS.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=True,
        setup_note=(
            "Context written to AGENTS.md.\n"
            "Cursor reads AGENTS.md and .cursorrules automatically.\n"
            "MCP: add fish-bridge to .cursor/mcp.json"
        ),
    ),

    # ------------------------------------------------------------------
    # Windsurf (Codeium)
    # Reads AGENTS.md, CLAUDE.md.
    # ------------------------------------------------------------------
    "windsurf": PlatformAdapter(
        display_name="Windsurf (Codeium)",
        output_targets=("AGENTS.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=True,
        setup_note=(
            "Context written to AGENTS.md.\n"
            "MCP: add fish-bridge to .windsurf/mcp.json"
        ),
    ),

    # ------------------------------------------------------------------
    # Gemini CLI / Gemini in IDEs
    # Reads GEMINI.md and AGENTS.md.
    # ------------------------------------------------------------------
    "gemini": PlatformAdapter(
        display_name="Gemini CLI / Gemini in IDE",
        output_targets=("GEMINI.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=True,
        setup_note=(
            "Context written to GEMINI.md.\n"
            "Gemini reads GEMINI.md automatically at session start.\n"
            "MCP: add fish-bridge to ~/.gemini/settings.json"
        ),
    ),

    # ------------------------------------------------------------------
    # AGENTS.md — universal agent instruction format
    # Works in: VS Code cloud agent, Codex, JetBrains cloud agent, etc.
    # ------------------------------------------------------------------
    "agents": PlatformAdapter(
        display_name="AGENTS.md (universal)",
        output_targets=("AGENTS.md",),
        nested_link_supported=False,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Context written to AGENTS.md (universal agent instruction format).\n"
            "Supported by: VS Code cloud agent, Codex, JetBrains, Cursor, Windsurf.\n"
            "Commit this file to the repo."
        ),
    ),

    # ------------------------------------------------------------------
    # Multi / All — write to all known targets simultaneously
    # ------------------------------------------------------------------
    "all": PlatformAdapter(
        display_name="All platforms (multi-target)",
        output_targets=(
            ".fish_bridge/context.md",         # VS Code local (via nested link)
            ".github/copilot-instructions.md",  # JetBrains, Eclipse, Xcode, VS, github.com
            "AGENTS.md",                        # Codex, Cursor, Windsurf, cloud agents
            "CLAUDE.md",                        # Claude Code
            "GEMINI.md",                        # Gemini CLI
        ),
        nested_link_supported=True,
        mcp_supported=True,
        ingest_source=None,
        local_mode_safe=False,
        setup_note=(
            "Writes context to ALL known instruction files simultaneously.\n"
            "Use when working across multiple AI tools on the same project.\n"
            "Commit .github/copilot-instructions.md, AGENTS.md, CLAUDE.md, GEMINI.md.\n"
            "WARNING: do not include secrets/API keys in sessions ingested in this mode."
        ),
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_alias_map() -> dict[str, str]:
    """Return alias → canonical_name mapping."""
    m: dict[str, str] = {}
    for canonical, adapter in PLATFORM_ADAPTERS.items():
        for alias in adapter.aliases:
            m[alias] = canonical
    return m


_ALIAS_MAP: dict[str, str] = _build_alias_map()


def get_adapter(tool: str) -> PlatformAdapter | None:
    """Return the PlatformAdapter for *tool*, resolving aliases.

    Returns None if the tool name is not recognised.
    """
    canonical = _ALIAS_MAP.get(tool.lower(), tool.lower())
    return PLATFORM_ADAPTERS.get(canonical)


def list_tools() -> list[str]:
    """Return sorted list of all canonical tool names."""
    return sorted(PLATFORM_ADAPTERS.keys())


def resolve_tool(tool: str) -> str | None:
    """Return canonical tool name for *tool* (including alias resolution).

    Returns None if unrecognised.
    """
    canonical = _ALIAS_MAP.get(tool.lower(), tool.lower())
    return canonical if canonical in PLATFORM_ADAPTERS else None
