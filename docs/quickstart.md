# fish_bridge — Quickstart

## Prerequisites

- Python 3.11+
- An LLM API key (Gemini, Anthropic, or local Ollama)

## Install

```bash
pip install fish-bridge-mcp
# or with Claude backend extras
pip install "fish-bridge-mcp[claude]"
```

## 1-minute setup

```bash
# Create your session
fish-bridge init --session my-project

# Configure your backend (set the API key as an environment variable)
export GEMINI_API_KEY=your-key-here
fish-bridge config set backend gemini
# or: anthropic, openai, local (Ollama)

# Ingest a conversation log (VS Code Copilot, Claude, etc.)
fish-bridge ingest --file path/to/session.jsonl --session my-project

# Compile context to your instructions file
fish-bridge compile --output .github/copilot-instructions.md
```

## Ingest sources

| Command | What it ingests |
|---|---|
| `ingest --source copilot` | VS Code Copilot JSONL — auto-discovered from workspaceStorage |
| `ingest --source jetbrains` | JetBrains Copilot Chat — guided paste (no transcript file on disk) |
| `ingest --paste` | Any chat tool — opens `$EDITOR`, paste and save |
| `merge --source document --file HANDOVER.md` | Markdown / text docs |
| `merge --source codebase --path .` | Git log + README |
| `merge --source obsidian --vault ~/notes` | Obsidian vault |
| `merge --source deps --path .` | package.json / pyproject.toml etc. |
| `merge --source testout --file results.json` | Jest/pytest/JUnit test output |

## View your graph

```bash
fish-bridge serve        # opens http://localhost:8080 (Cytoscape.js)
fish-bridge show         # pretty-print nodes in terminal
```

## MCP server

```bash
fish-bridge serve-mcp    # starts MCP server on stdio
```

Add to VS Code `settings.json` (see `examples/copilot-mcp-config.json`) or
Claude Desktop `claude_desktop_config.json` (see `examples/claude-mcp-config.json`).

## Next steps

- [Configuration](configuration.md) — all config file options
- [Graph schema](graph-schema.md) — node types, edge relations, status values
- [Compiler modes](compiler-modes.md) — active thread, focus, digest
- [Ingestors](ingestors.md) — format details for each ingest source
