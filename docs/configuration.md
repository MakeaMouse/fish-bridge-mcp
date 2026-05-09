# Configuration

fish_bridge stores its configuration at `~/.fish_bridge/config.yaml`.

Run `fish-bridge config show` to view current values.
Run `fish-bridge config set KEY VALUE` to change a value.

## All options

```yaml
# ~/.fish_bridge/config.yaml

# --- Backend ---
# One of: claude | openai | gemini | local | hybrid
extraction:
  backend: "gemini"
  exclude_patterns: []   # strings to strip from turns before extraction (e.g. PII)

  claude:
    model: "claude-opus-4-7"
    api_key_env: "ANTHROPIC_API_KEY"   # env var name (not the key itself)

  openai:
    model: "gpt-4.1-mini"
    api_key_env: "OPENAI_API_KEY"
    base_url: null   # override for OpenAI-compatible endpoints (LM Studio, Groq, etc.)

  gemini:
    model: "models/gemini-2.5-flash"   # "models/" prefix required
    api_key_env: "GEMINI_API_KEY"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

  local:
    provider: "ollama"
    base_url: "http://localhost:11434"
    model: "qwen2.5:7b"
    embed_model: "nomic-embed-text"    # also used for semantic dedup

  hybrid:
    realtime_backend: "local"          # used every turn (fast, free)
    consolidation_backend: "gemini"    # used every N turns (higher quality)
    consolidation_every_n: 10

# --- Output ---
output:
  delivery_mode: "local"                             # local | shared
  local_context_file: ".fish_bridge/context.md"      # used when delivery_mode=local
  shared_context_file: ".github/copilot-instructions.md"  # used when delivery_mode=shared
  token_budget: 800                                  # soft limit for Mode A compilation

# --- Storage ---
data_dir: "~/.fish_bridge/sessions"   # SQLite session files stored here
```

## Environment variables

API keys can be set as environment variables and take precedence over the config file:

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY` | Gemini backend |
| `ANTHROPIC_API_KEY` | Claude backend |
| `OPENAI_API_KEY` | OpenAI backend |
| `FISH_BRIDGE_BACKEND` | Override `extraction.backend` |
| `FISH_BRIDGE_PROJECT` | Override the project directory (used by MCP server) |
| `FISH_BRIDGE_SESSION` | Override the active session name |

## Quick config changes

```bash
fish-bridge config set backend gemini
fish-bridge config set backend local
fish-bridge config show
```

## Using an OpenAI-compatible endpoint (LM Studio, Groq, Together, etc.)

```yaml
extraction:
  backend: "openai"
  openai:
    model: "your-model-name"
    api_key: "not-needed"
    base_url: "http://localhost:1234/v1"   # LM Studio default
```

## Gemini note

The `models/` prefix is required for Gemini model names. `gemini-2.0-flash` returns 404 for new API keys — use `models/gemini-2.5-flash`.
