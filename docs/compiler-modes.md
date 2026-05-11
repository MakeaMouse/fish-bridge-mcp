# Compiler Modes

fish_bridge has three compiler modes. Choose based on your IDE's context budget
and whether you need a full digest or a targeted slice.

## Mode A — Active Thread (`active_thread`)

**Best for**: Real-time coding sessions with a small context window.

Outputs an XML block summarising:
- All `ACTIVE` / `IN_PROGRESS` nodes
- Recently updated nodes (last N turns)
- Open questions and blocked tasks
- Key decisions

Output size: **~300–600 tokens**

```bash
fish-bridge compile --mode active --output .github/copilot-instructions.md
```

## Mode B — Focus Subgraph (`focus`)

**Best for**: Query-driven context injection ("what do we know about Redis?").

Uses BFS from the highest-scoring node matching the query, limited to `--max-nodes` hops.

```bash
fish-bridge compile --mode focus --query "DNC list caching"
```

Output size: **~200–800 tokens** depending on subgraph density.

## Mode C — Full Digest (`digest`)

**Best for**: Handover documents, onboarding, or periodic project summaries.

Runs community detection on the full graph and emits a Markdown summary
with one section per cluster.

```bash
fish-bridge compile --mode digest --output HANDOVER.md
```

Output size: **~1000–5000 tokens** depending on graph size.

## Automatic compilation

By default, `ingest` and `merge` run the compiler after every change
(mode = `active_thread`). Disable with `--no-compile`.

## Output format

The compiler writes to the file specified by `--output`, or prints to stdout.
The output is structured for direct inclusion in a copilot instructions file
or system prompt.
