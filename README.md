# fish_bridge

[![CI](https://github.com/MakeaMouse/fish-bridge-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/MakeaMouse/fish-bridge-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fish-bridge-mcp)](https://pypi.org/project/fish-bridge-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/fish-bridge-mcp)](https://pypi.org/project/fish-bridge-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Session-scoped knowledge graph engine for AI chat context compression.**

Converts raw AI chat (40k+ tokens) into a compact typed knowledge graph (~300–800 tokens) and writes it to `.github/copilot-instructions.md` or `CLAUDE.md` — automatically included in every AI turn across all modes (ask, edit, agent). No MCP server required for the core workflow.

```
Raw session (40k tokens) → [fish_bridge] → Compressed graph (350 tokens)
                                              written to copilot-instructions.md
                                              picked up by every AI turn automatically
```

## Install

```bash
# Minimal install — use the local Ollama backend (free, offline)
pip install fish-bridge-mcp

# With Gemini backend (best cost/quality balance — ~$0.0002/turn)
pip install "fish-bridge-mcp[gemini]"
export GEMINI_API_KEY=...

# With Claude backend (highest quality — ~$0.002/turn)
pip install "fish-bridge-mcp[claude]"
export ANTHROPIC_API_KEY=sk-ant-...

# With OpenAI backend
pip install "fish-bridge-mcp[openai]"
export OPENAI_API_KEY=sk-...

# Everything
pip install "fish-bridge-mcp[all]"
```

## 2-minute quickstart

```bash
# 1. Initialize for your project
fish-bridge init --tool copilot --project ./

# 2. Ingest the latest Copilot session (auto-discovers JSONL on macOS/Linux/Windows)
fish-bridge ingest --source copilot

# 3. View the graph
fish-bridge show

# 4. Compile to your instructions file (done automatically after ingest)
fish-bridge compile
```

That's it. `.github/copilot-instructions.md` now contains a ~350-token compressed summary of your session, replacing raw history in every future turn.

## Backends

| Backend | Install extra | Model | Quality | Cost/turn |
|---|---|---|---|---|
| `local` (Ollama) | *(none — requires [Ollama](https://ollama.com))* | qwen2.5:7b | ~85% | $0 |
| `gemini` | `[gemini]` | gemini-2.5-flash | ~95% | ~$0.0002 |
| `openai` | `[openai]` | gpt-4.1-mini | ~93% | ~$0.0003 |
| `claude` | `[claude]` | claude-opus-4-7 | ~97% | ~$0.002 |
| `hybrid` | `[claude]` or `[openai]` | local+cloud | best | mixed |

Configure with:
```bash
fish-bridge config set backend gemini
# or set GEMINI_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY as env vars
```

## Full CLI reference

```bash
# --- Session init ---
fish-bridge init                          # create session for current project
fish-bridge init --tool claude            # → writes to CLAUDE.md instead

# --- Ingest chat turns ---
fish-bridge ingest --source copilot       # auto-discover latest VS Code Copilot session
fish-bridge ingest --source copilot --session <id>  # target specific session
fish-bridge ingest --source jetbrains     # JetBrains Copilot Chat (guided paste — no transcript file)
fish-bridge ingest --paste                # paste any chat text (universal fallback)
fish-bridge watch --source copilot        # tail JSONL, auto-update on new turns

# --- Merge external knowledge ---
fish-bridge merge --source document --file HANDOVER.md
fish-bridge merge --source codebase --path ./            # git log + README
fish-bridge merge --source obsidian --vault ~/notes
fish-bridge merge --source deps --path ./                # package.json / pyproject.toml etc.
fish-bridge merge --source testout --file results.json   # jest / pytest / JUnit
fish-bridge merge --source iac --path ./                 # Terraform / CDK / CloudFormation
fish-bridge merge --source openapi --file openapi.yaml
fish-bridge merge --source session --file prior.chatgraph.json

# --- Compile & view ---
fish-bridge compile                       # update instruction file (runs after ingest by default)
fish-bridge compile --mode digest         # full handover markdown
fish-bridge compile --mode focus --query "Redis caching"
fish-bridge show                          # pretty-print active nodes
fish-bridge show --resolved               # include resolved items
fish-bridge serve                         # open Cytoscape.js graph viewer at localhost:8080
fish-bridge digest                        # generate handover digest

# --- Node management ---
fish-bridge resolve "DNC caching strategy"
fish-bridge defer "v16 index validation"
fish-bridge add "Use Redis for session cache" --type decision
fish-bridge conflict show
fish-bridge conflict resolve <node-id> --keep old

# --- Export / import / diff ---
fish-bridge export                        # save .chatgraph.json
fish-bridge import prior-session.chatgraph.json
fish-bridge diff session-a.chatgraph.json session-b.chatgraph.json

# --- Config ---
fish-bridge config show
fish-bridge config set backend gemini
```

## MCP server (optional — agent mode only)

The MCP server adds real-time `record_turn` capture when using VS Code agent mode. It is **not required** — the file-based workflow above works in all modes without it.

```bash
# Zero-install (recommended)
uvx fish-bridge-mcp

# Or: pip install "fish-bridge-mcp[claude]" && fish-bridge serve-mcp
```

Add to `.vscode/mcp.json`:
```json
{
  "servers": {
    "fish-bridge": {
      "command": "uvx",
      "args": ["fish-bridge-mcp"],
      "env": { "FISH_BRIDGE_BACKEND": "gemini", "GEMINI_API_KEY": "${env:GEMINI_API_KEY}" }
    }
  }
}
```

See `examples/` for Claude Desktop, Cursor, and Windsurf configs.

**MCP tools**: `record_turn`, `get_context`, `get_focus`, `mark_resolved`, `add_node`, `export_session`, `import_session`, `show_active`, `list_deferred`

## Ingest sources

| Source | Command | What it ingests |
|---|---|---|
| Copilot | `ingest --source copilot` | VS Code Copilot JSONL transcript (auto-discovered) |
| Paste | `ingest --paste` | Any chat text — universal fallback |
| Document | `merge --source document` | Markdown, JSON, YAML specs and ADRs |
| Codebase | `merge --source codebase` | Git commits + README + HANDOVER |
| Obsidian | `merge --source obsidian` | Vault notes with wikilinks and frontmatter |
| Session | `merge --source session` | Prior `.chatgraph.json` export |
| Deps | `merge --source deps` | package.json, pyproject.toml, Cargo.toml, go.mod, Gemfile, pom.xml |
| Test output | `merge --source testout` | Jest JSON, pytest JSON, JUnit XML — error nodes per failing test |
| IaC | `merge --source iac` | Terraform, CDK (synth output), CloudFormation, docker-compose |
| OpenAPI | `merge --source openapi` | OpenAPI 3.x / Swagger 2.0 / AsyncAPI specs |

## How it works

1. **Ingest** — reads raw chat turns from JSONL (Copilot), paste, or any file format
2. **Extract** — LLM extracts typed nodes (questions, decisions, errors, tasks, skills, files) and edges
3. **Dedup** — semantic similarity merges near-duplicates; conflict detection flags status reversals
4. **Compile** — graph is compressed to ~300–800 token XML/markdown block
5. **Write** — block is written to `.github/copilot-instructions.md` (or `CLAUDE.md`)
6. **Deliver** — AI tool reads the file automatically on the next turn — no injection, no agent required

## Documentation

- [Quickstart](docs/quickstart.md)
- [Configuration](docs/configuration.md)
- [Graph schema](docs/graph-schema.md)
- [Compiler modes](docs/compiler-modes.md)
- [Ingestors](docs/ingestors.md)

## License

MIT — see [LICENSE](LICENSE)

