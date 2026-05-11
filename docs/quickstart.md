# fish_bridge — Quickstart

## Prerequisites

- Python 3.11+
- An LLM API key (Gemini, Anthropic, or local Ollama)

## Install

```bash
pip install fish-bridge-mcp
# or with a cloud backend extra (Gemini recommended)
pip install "fish-bridge-mcp[gemini]"
# See README for uv tool install (recommended) and other backends
```

## 1-minute setup

```bash
# Initialize for this project folder
fish-bridge init

# Configure your backend (set the API key as an environment variable)
export GEMINI_API_KEY=your-key-here
fish-bridge config --backend gemini
# or: --backend claude / --backend openai / --backend local

# Ingest the latest VS Code Copilot session (auto-discovered)
fish-bridge ingest --source copilot
# Or ingest from a saved export file:
fish-bridge ingest --source file --file path/to/export.json

# Compile context to your instructions file
fish-bridge compile --output .github/copilot-instructions.md
```

## Ingest sources

| Command | What it ingests |
|---|---|
| `ingest --source copilot` | VS Code Copilot JSONL — auto-discovered from workspaceStorage |
| `ingest --source paste` | Any chat tool — opens `$EDITOR`, paste and save |
| `ingest --source file --file export.json` | From a saved export file (Claude JSON, plain text) |
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
