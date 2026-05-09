"""fish_bridge Python library usage demo.

This script demonstrates the public API for use in custom tools and integrations.
Run from the repo root after installing:
    pip install -e ".[claude]"
    export ANTHROPIC_API_KEY=sk-ant-...
"""
from pathlib import Path
from fish_bridge import SessionGraph
from fish_bridge.config import build_backend, load_config
from fish_bridge.ingestors.document import DocumentIngestor
from fish_bridge.ingestors.codebase import CodebaseIngestor
from fish_bridge.compiler.active_thread import ActiveThreadCompiler
from fish_bridge.compiler.digest import DigestCompiler
from fish_bridge.compiler.focus import FocusCompiler
from fish_bridge.graph.schema import RawTurn

# ---------------------------------------------------------------------------
# 1. Open (or create) a session
# ---------------------------------------------------------------------------

DATA_DIR = Path.home() / ".fish_bridge" / "sessions"
SESSION_ID = "demo-session"

sg = SessionGraph.open(SESSION_ID, DATA_DIR)
print(f"Opened session '{SESSION_ID}'")

# ---------------------------------------------------------------------------
# 2. Manually add a turn via extraction
# ---------------------------------------------------------------------------

cfg = load_config()
backend = build_backend(cfg)

turn = RawTurn(
    session_id=SESSION_ID,
    turn_number=1,
    role_user="How should I handle DNC list caching to avoid Lambda cold-start latency?",
    role_assistant=(
        "I recommend Redis ElastiCache with a 24-hour TTL. Cache the full DNC list "
        "on Lambda cold start and refresh asynchronously. This avoids per-request "
        "DynamoDB lookups and reduces p99 latency from ~200ms to ~5ms."
    ),
)
nodes, edges = backend.extract(turn, cfg.extraction.exclude_patterns)
stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
print(f"Extracted: {len(stored_nodes)} nodes, {len(stored_edges)} edges")

# ---------------------------------------------------------------------------
# 3. Merge a document (e.g. a spec file or HANDOVER.md)
# ---------------------------------------------------------------------------

# doc_turns = DocumentIngestor().ingest(
#     file_path=Path("HANDOVER.md"),
#     session_id=SESSION_ID,
# )
# for t in doc_turns:
#     n, e = backend.extract(t, cfg.extraction.exclude_patterns)
#     sg.merge_extraction(n, e)

# ---------------------------------------------------------------------------
# 4. Merge the codebase (git log + README)
# ---------------------------------------------------------------------------

# code_turns = CodebaseIngestor().ingest(
#     path=Path("."),
#     session_id=SESSION_ID,
#     n_commits=10,
# )
# for t in code_turns:
#     n, e = backend.extract(t, cfg.extraction.exclude_patterns)
#     sg.merge_extraction(n, e)

# ---------------------------------------------------------------------------
# 5. Compile to different output formats
# ---------------------------------------------------------------------------

all_nodes = sg.all_nodes()
all_edges = sg.all_edges()

# Mode A — active thread XML (~300 tokens, replaces raw history)
xml = ActiveThreadCompiler(SESSION_ID).compile(all_nodes, all_edges, turn_count=1)
print("\n--- Mode A: Active Thread ---")
print(xml[:500], "...")

# Mode B — focus subgraph (query-driven)
focus_xml = FocusCompiler(SESSION_ID, max_nodes=10).compile(
    all_nodes, all_edges, query="DNC caching Redis"
)
print("\n--- Mode B: Focus Subgraph ---")
print(focus_xml[:300], "...")

# Mode C — full digest (community clusters)
digest_md = DigestCompiler(SESSION_ID).compile(all_nodes, all_edges)
print("\n--- Mode C: Full Digest ---")
print(digest_md[:400], "...")

# ---------------------------------------------------------------------------
# 6. Write to an instruction file (primary delivery mechanism)
# ---------------------------------------------------------------------------

output_file = Path(".github") / "copilot-instructions.md"
# ActiveThreadCompiler(SESSION_ID).write(all_nodes, all_edges, output_file, turn_count=1)
# print(f"\nContext written to {output_file}")

# ---------------------------------------------------------------------------
# 7. Export portable session file
# ---------------------------------------------------------------------------

export_path = Path(f"{SESSION_ID}.chatgraph.json")
sg.export_json(export_path)
print(f"\nSession exported to {export_path}")

sg.close()
