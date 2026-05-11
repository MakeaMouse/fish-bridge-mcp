# Ingestors

Ingestors convert external knowledge sources into `RawTurn` objects (or directly
into `GraphNode`/`GraphEdge` lists) for storage in the session graph.

## Chat ingestors

### VS Code Copilot JSONL (`copilot`)
```bash
fish-bridge ingest --source copilot
```
Auto-discovers the latest Copilot session JSONL from VS Code's `workspaceStorage` directory
(probes `transcripts/`, `debug-logs/`, and `sessions/` subdirectories in order).
Each user/assistant pair becomes one `RawTurn`.

### JetBrains Copilot Chat (`jetbrains`)
```bash
fish-bridge ingest --source jetbrains
# aliases: --source idea, --source pycharm, --source webstorm
```
JetBrains Copilot Chat does **not** write transcript files to disk (unlike VS Code).
This source opens `$EDITOR` with step-by-step copy instructions:
1. Select-all (`⌘A`) and copy (`⌘C`) from the JetBrains Chat panel
2. Paste into the editor, save and quit

The parser recognises the JetBrains `Me:` / `GitHub Copilot:` prefix format
as well as the standard `User:` / `Assistant:` formats.

### Paste fallback (any tool)
```bash
fish-bridge ingest --source paste
```
Opens `$EDITOR` — paste any chat text, save and quit. Recognised formats:
`You:` / `User:` / `Me:` (user) and `Assistant:` / `Claude:` / `Copilot:` /
`GitHub Copilot:` / `GPT:` / `AI:` (assistant).

### File (`file`)
```bash
fish-bridge ingest --source file --file session.jsonl
fish-bridge ingest --source claude --file export.json   # Claude JSON export
```
Parses a local file. Supports Claude JSON export format and plain-text delimiter format.

---

## Document ingestor (`document`)
```bash
fish-bridge merge --source document --file HANDOVER.md
```
Reads plain text or Markdown files and splits them into ~1500-character turns.

---

## Codebase ingestor (`codebase`)
```bash
fish-bridge merge --source codebase --path . --commits 20
```
Ingests:
- The last N git commits (summary + diff)
- README.md and any `*HANDOVER*` docs in the repo root

Use `--no-docs` to skip documentation files.

---

## Obsidian vault ingestor (`obsidian`)
```bash
fish-bridge merge --source obsidian --vault ~/notes --tag project/my-app
```
Walks an Obsidian vault and ingests `.md` files. Filter by:
- `--tag <tag>` — only notes with that frontmatter tag
- `--folder <folder>` — only notes under that relative path

---

## Session file ingestor (`session`)
```bash
fish-bridge merge --source session --file export.chatgraph.json
```
Imports a portable `.chatgraph.json` export (produced by `fish-bridge export`).
No LLM extraction — nodes/edges are inserted directly.

---

## Dependency ingestor (`deps`)
```bash
fish-bridge merge --source deps --path .
```
Scans the project root for package manifests and creates **skill nodes** for each dependency.

Supported manifests:

| File | Registry |
|---|---|
| `package.json` | npm |
| `pyproject.toml` | PyPI (PEP 621 + Poetry) |
| `requirements.txt` | PyPI |
| `Cargo.toml` | crates.io |
| `go.mod` | pkg.go.dev |
| `Gemfile` | RubyGems |
| `pom.xml` | Maven |

Dev dependencies are tagged `metadata.is_dev: true`.

---

## Test output ingestor (`testout`)
```bash
fish-bridge merge --source testout --file results.json
fish-bridge merge --source testout --file junit.xml
```
Parses structured test output and creates **error nodes** for failing tests.

Supported formats:

| Format | How to generate |
|---|---|
| Jest JSON | `jest --json > results.json` |
| pytest JSON | `pytest --json-report --json-report-file results.json` |
| JUnit XML | standard CI output (`<testsuites>` or `<testsuite>` root) |

Each failing test becomes a `GraphNode(type=ERROR, subtype="test_failure")` linked
to its test file via a `tested_by` edge.
