# fish_bridge

**Session-scoped knowledge graph engine for AI chat context compression.**

Converts raw AI chat (40k+ tokens) into a compact typed knowledge graph (~300 tokens) and writes it to `.github/copilot-instructions.md` / `CLAUDE.md` — automatically included in every AI turn.

## Quickstart

```bash
pip install fish-bridge-mcp[claude]

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Initialize for your project
fish-bridge init --tool copilot --project ./

# Ingest the latest Copilot session
fish-bridge ingest --source copilot

# Show the active graph
fish-bridge show
```

## Phase 0 CLI Commands

```bash
fish-bridge init                    # initialize for a project
fish-bridge ingest --source copilot # auto-discover & ingest latest Copilot session
fish-bridge ingest --paste          # paste any chat text
fish-bridge compile                 # write compressed graph to instruction file
fish-bridge show                    # display active thread with Rich
fish-bridge watch --source copilot  # tail JSONL, auto-update on new turns
```

## Backends

| Backend | Model | Quality | Cost/Turn |
|---|---|---|---|
| claude | claude-sonnet-4-6 | ~97% | ~$0.002 |
| openai | gpt-4.1-mini | ~93% | ~$0.0003 |
| local (Phase 1) | qwen2.5:7b | ~85% | $0 |

## License

MIT
