# Contributing to fish_bridge

Thank you for your interest in contributing.

## Setup

```bash
git clone https://github.com/MakeaMouse/fish-bridge-mcp
cd fish-bridge-mcp
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -q                    # all tests
pytest tests/test_extraction.py -v  # specific file
pytest -k "test_jest"               # specific test pattern
```

All 149 tests must pass before submitting a PR. The test suite runs without any API key — all extraction tests use mock backends.

## Adding a new ingestor

1. Create `fish_bridge/ingestors/<name>.py` with a class extending `AbstractIngestor`
2. The `ingest()` method must return either `list[RawTurn]` (for LLM extraction) or `tuple[list[GraphNode], list[GraphEdge]]` (for direct graph insertion)
3. Add a `--source <name>` branch in `cli.py:merge_cmd()`
4. Update `docs/ingestors.md` with format details
5. Add tests in `tests/test_ingestors.py` or a new `tests/test_phase*.py`

## Adding a new extraction backend

1. Create `fish_bridge/extraction/<name>.py` extending `AbstractExtractionBackend`
2. Implement `_call_llm(user_message, assistant_message) -> dict`
3. Register in `fish_bridge/config.py:build_backend()` under the new backend name
4. Add to the backend table in `README.md` and `docs/configuration.md`
5. Add tests with a fixture in `tests/test_extraction.py`

## Code style

- Python 3.11+
- `ruff check .` must pass (configured in `pyproject.toml`)
- No new hard dependencies without discussion — keep the base install minimal
- `anthropic` and `openai` are optional extras; new cloud dependencies must be optional too

## Commit messages

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`. 
Breaking changes: add `!` after the type (`feat!:`).

## Pull requests

- One feature or fix per PR
- Include tests
- Update `CHANGELOG.md` under `## [Unreleased]`
- Ensure `pytest tests/ -q` passes locally before opening the PR
