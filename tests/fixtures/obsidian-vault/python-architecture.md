---
tags: [python, architecture]
status: active
---
# Python Project Architecture

This note describes best practices for structuring Python projects.

Key decisions:
- Use `src/` layout for installable packages to avoid import confusion
- Always include a `pyproject.toml` as the single source of truth
- Separate `tests/` directory at the project root
- Use `conftest.py` for shared pytest fixtures

Related tools: [[pytest]], [[ruff]], [[mypy]]
