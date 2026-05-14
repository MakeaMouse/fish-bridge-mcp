# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability, please report it privately using GitHub's [Security Advisory](https://github.com/MakeaMouse/fish-bridge-mcp/security/advisories/new) feature.

Include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fix (optional)

You will receive a response within 48 hours and a patch within 7 days for confirmed vulnerabilities.

## Scope

Areas of particular security relevance for this project:

- **API key handling** — fish-bridge-mcp accepts API keys via environment variables. Keys are never written to disk or included in the knowledge graph output.
- **PII in graph output** — the default `_DEFAULT_EXCLUDE_PATTERNS` in `config.py` masks passwords, bearer tokens, AWS keys, and database connection strings. If you find a pattern that leaks sensitive data, please report it.
- **SQLite session files** — session databases are stored in `.fish_bridge/` which is gitignored. If you find a path where session data could be exposed, please report it.
- **MCP server** — the server runs locally and binds to stdio only (no network port). Any bypass of this isolation is in scope.
