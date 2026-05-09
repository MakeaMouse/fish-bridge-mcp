# Changelog

All notable changes to fish_bridge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026-05-04

### Added

**Core pipeline (Phase 0)**
- `CopilotTranscriptIngestor` — auto-discovers and parses VS Code Copilot JSONL session files on macOS, Linux, and Windows
- `ChatTurnIngestor` — universal paste/file fallback for any chat tool
- Extraction backends: `ClaudeBackend` (tool_use schema enforcement), `OpenAIBackend` (json_schema enforcement), `GeminiBackend` (OpenAI-compatible endpoint)
- `SessionGraph` — SQLite WAL-mode storage with full CRUD
- `ActiveThreadCompiler` — Mode A XML output (~300–800 tokens); writes to `.github/copilot-instructions.md` or `CLAUDE.md`
- CLI: `init`, `ingest`, `compile`, `show`, `watch`, `config`, `export`, `import`

**Semantic dedup + local backend (Phase 1)**
- `OllamaBackend` — local extraction with `format=` JSON Schema enforcement at sampler level
- `HybridBackend` — local for real-time, cloud for consolidation every N turns
- `dedup.py` — embedding + cosine similarity dedup (merge >0.88, relate 0.70–0.88)
- Session identity: `<workspace>-<YYYY-MM-DD>` default, persisted to `session.lock`
- CLI: `merge --source session`

**Multi-source ingestion (Phase 2)**
- `ObsidianIngestor` — wikilinks, YAML frontmatter, folder namespacing
- `CodebaseIngestor` — git log + README/HANDOVER/CHANGELOG docs
- `DocumentIngestor` — markdown, JSON, YAML spec files
- `DependencyFileIngestor` — package.json, pyproject.toml, requirements.txt, Cargo.toml, go.mod, Gemfile, pom.xml
- `TestOutputIngestor` — Jest JSON, pytest JSON report, JUnit XML → error nodes per failing test
- CLI: `merge --source document|codebase|obsidian|deps|testout`

**Advanced compilation (Phase 3)**
- `FocusCompiler` — Mode B: query-driven subgraph via semantic similarity
- `DigestCompiler` — Mode C: community detection (Louvain) → cluster summaries
- `algorithms.py` — community detection, BFS traversal, semantic search, shortest path
- CLI: `digest`

**Phase 4 — Polish & launch**
- `IaCIngestor` — Terraform HCL, CDK synth JSON, CloudFormation YAML, docker-compose → decision/concept nodes
- `OpenAPIIngestor` — OpenAPI 3.x / Swagger 2.0 → skill (endpoints), concept (schemas), decision (auth)
- `fish-bridge serve` — Cytoscape.js graph viewer (self-contained HTML, bundled JS)
- `fish-bridge serve-mcp` — FastMCP server with 9 tools
- Content-zone pre-processor (`preprocessor.py`) — code blocks, stack traces, URLs, CLI output
- Conflict detection in merge pipeline — status reversal → `CONFLICTED` node + `contradicts` edge
- Node management CLI: `resolve`, `defer`, `add`, `conflict show`, `conflict resolve`
- `fish-bridge diff` — compare two `.chatgraph.json` exports
- 141 passing tests

### Acceptance test (May 2, 2026)
- Input: 119-turn Copilot session, 51 substantive turns
- Output: 89 unique nodes, 62 edges, 0 extraction errors
- 95% high-confidence nodes (≥0.7)
- Compression: ~28,994 raw tokens → ~2,747 compiled tokens (10× reduction)
