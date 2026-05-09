"""fish_bridge MCP server — optional real-time per-turn capture.

Exposes MCP tools for VS Code Copilot agent mode and other MCP clients.
The file-based path (.github/copilot-instructions.md) is the primary delivery
mechanism and works without this server running.

Setup:
    uvx fish-bridge-mcp
    or add to .vscode/mcp.json / ~/.claude/claude_desktop_config.json

To avoid approval prompts on every turn in VS Code agent mode, add fish-bridge
to chat.tools.autoApproveList in VS Code settings.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from fish_bridge.config import (
    FishBridgeConfig,
    build_backend,
    get_active_session_id,
    get_data_dir,
    load_config,
)
from fish_bridge.compiler.active_thread import ActiveThreadCompiler
from fish_bridge.compiler.focus import FocusCompiler
from fish_bridge.graph.schema import GraphEdge, GraphNode, NodeStatus, NodeType, RawTurn
from fish_bridge.graph.session import SessionGraph


# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "fish-bridge",
    instructions=(
        "fish_bridge manages a session-scoped knowledge graph for AI chat compression. "
        "After each response, call record_turn with the user message and your response. "
        "Use get_context to retrieve the current compressed session context. "
        "Use mark_resolved to close questions or tasks as they are completed."
    ),
)

# Lazy-loaded singletons — initialised on first tool call
_cfg:     FishBridgeConfig | None = None
_sg:      SessionGraph      | None = None
_project: Path                     = Path(".")


def _get_cfg() -> FishBridgeConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _get_sg() -> SessionGraph:
    global _sg
    if _sg is None:
        cfg        = _get_cfg()
        session_id = get_active_session_id(_project)
        data_dir   = get_data_dir(cfg)
        _sg        = SessionGraph.open(session_id, data_dir, dedup_config=cfg.dedup)
    return _sg


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def record_turn(user_message: str, assistant_message: str) -> str:
    """Ingest one user/assistant exchange into the session graph.

    Call this after each response in agent mode.  The graph is updated
    incrementally — no full re-extraction required.

    Returns a brief acknowledgement with node/edge counts.
    """
    sg  = _get_sg()
    cfg = _get_cfg()

    turn = RawTurn(
        session_id=   sg.session_id,
        turn_number=  len(sg.all_nodes()) + 1,  # approximate
        role_user=    user_message,
        role_assistant=assistant_message,
        source=       "mcp_record_turn",
    )

    try:
        backend = build_backend(cfg)
        nodes, edges = await asyncio.to_thread(
            backend.extract, turn, cfg.extraction.exclude_patterns
        )
        stored_nodes, stored_edges = await asyncio.to_thread(
            sg.merge_extraction, nodes, edges
        )
    except Exception as exc:
        return f"queued (extraction deferred: {exc})"

    # Update the instruction file if it exists
    _maybe_update_output(sg, cfg)

    return (
        f"recorded: +{len(stored_nodes)} nodes, +{len(stored_edges)} edges "
        f"(session total: {len(sg.all_nodes())} nodes)"
    )


@mcp.tool()
def get_context() -> str:
    """Return the current compressed session context as XML.

    This is the same content written to .github/copilot-instructions.md.
    Use this when you need to recall the session state mid-conversation.
    """
    sg  = _get_sg()
    cfg = _get_cfg()
    compiler = ActiveThreadCompiler(sg.session_id, cfg.output.token_budget)
    return compiler.compile(sg.all_nodes(), sg.all_edges(), turn_count=len(sg.all_nodes()))


@mcp.tool()
def get_focus(query: str) -> str:
    """Return a query-scoped subgraph context XML.

    Finds the nodes most relevant to *query* and their graph neighborhood.
    Use this for targeted technical questions where full session context is too broad.

    Args:
        query: Natural language question or topic (e.g. "CORS headers configuration")
    """
    sg       = _get_sg()
    compiler = FocusCompiler(sg.session_id)
    return compiler.compile(sg.all_nodes(), sg.all_edges(), query=query)


@mcp.tool()
def mark_resolved(label: str, note: str = "") -> str:
    """Mark a question, task, or error as resolved.

    Args:
        label: The label of the node to resolve (partial match, case-insensitive).
        note:  Optional note explaining how it was resolved.
    """
    sg = _get_sg()
    nodes = sg.all_nodes()
    label_lower = label.lower()

    matches = [n for n in nodes if label_lower in n.label.lower()]
    if not matches:
        return f"No node found matching '{label}'."

    updated = []
    for node in matches[:1]:  # resolve only the best match
        new_status: NodeStatus
        node_type = str(node.type)
        if node_type == "question":
            new_status = NodeStatus.RESOLVED
        elif node_type in ("task",):
            new_status = NodeStatus.DONE
        elif node_type == "error":
            new_status = NodeStatus.FIXED
        else:
            new_status = NodeStatus.RESOLVED
        sg.set_status(node.id, new_status, note=note or None)
        updated.append(node.label)

    _maybe_update_output(sg, _get_cfg())
    return f"Resolved: {', '.join(updated)}"


@mcp.tool()
def add_node(
    label: str,
    node_type: str = "task",
    summary: str = "",
    status: str = "",
) -> str:
    """Manually add a node to the session graph.

    Args:
        label:     Short descriptive label (≤8 words).
        node_type: One of: question | decision | concept | skill | file | error | task
        summary:   Optional 1-2 sentence description.
        status:    Optional status (default depends on type).
    """
    sg = _get_sg()
    try:
        nt = NodeType(node_type)
    except ValueError:
        return f"Invalid node_type '{node_type}'. Use: question|decision|concept|skill|file|error|task"

    _default_status = {
        "question": "active",
        "decision": "proposed",
        "task":     "pending",
        "error":    "active",
    }
    resolved_status_str = status or _default_status.get(node_type, "active")
    try:
        ns = NodeStatus(resolved_status_str)
    except ValueError:
        ns = NodeStatus.ACTIVE

    node = GraphNode(type=nt, label=label, summary=summary, status=ns)
    sg.add_node(node)
    _maybe_update_output(sg, _get_cfg())
    return f"Added {node_type}: '{label}' [{ns}]"


@mcp.tool()
def export_session() -> str:
    """Return the full session graph as a JSON string.

    Useful for saving a portable snapshot or importing into another session.
    The JSON can be saved as a .chatgraph.json file and imported with
    `fish-bridge import <file>`.
    """
    sg = _get_sg()
    return sg.to_model().model_dump_json(indent=2)


@mcp.tool()
def import_session(json_str: str) -> str:
    """Load a prior session graph and merge it into the current session.

    Pass the JSON content of a .chatgraph.json file (from export_session or
    `fish-bridge export`). Deferred nodes from the prior session become active;
    resolved and adopted decisions carry forward.

    Returns a summary of how many nodes and edges were merged.
    """
    import json as _json
    from fish_bridge.ingestors.session_file import SessionFileIngestor

    try:
        data = _json.loads(json_str)
    except _json.JSONDecodeError as exc:
        return f"Error: invalid JSON — {exc}"

    try:
        nodes, edges = SessionFileIngestor.load_from_dict(data)
    except Exception as exc:
        return f"Error loading session data: {exc}"

    sg = _get_sg()
    stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
    _maybe_update_output(sg, _get_cfg())
    return f"Imported {len(stored_nodes)} nodes, {len(stored_edges)} edges from prior session."


@mcp.tool()
def list_deferred() -> str:
    """Return all deferred items in the current session as plain text.

    Shows questions, tasks, decisions, and errors that have been parked.
    Use `mark_resolved` or `fish-bridge resolve` to act on them.
    """
    sg = _get_sg()
    nodes = sg.all_nodes()
    deferred = [n for n in nodes if str(n.status) in {"deferred", "NodeStatus.DEFERRED"}]

    if not deferred:
        return "No deferred items in this session."

    lines = [f"Deferred items ({len(deferred)}):", ""]
    for n in sorted(deferred, key=lambda x: str(x.type)):
        ntype = n.type if isinstance(n.type, str) else n.type.value
        lines.append(f"  [{ntype}] {n.label}")
        if n.summary:
            lines.append(f"    {n.summary}")
        lines.append(f"    id: {n.id}")
    return "\n".join(lines)


@mcp.tool()
def show_active() -> str:
    """Return the active session thread as plain text (not XML).

    Lists open questions, pending tasks, active errors, and key decisions.
    """
    sg = _get_sg()
    nodes = sg.active_nodes()

    lines: list[str] = [f"fish_bridge — {sg.session_id}  ({len(nodes)} active nodes)", ""]

    _sections: list[tuple[str, set[str]]] = [
        ("Open Questions",  {"question"}),
        ("Open Errors",     {"error"}),
        ("Decisions",       {"decision"}),
        ("Tasks",           {"task"}),
        ("Context",         {"skill", "concept", "file"}),
    ]
    for header, types in _sections:
        group = [n for n in nodes if str(n.type) in types]
        if not group:
            continue
        lines.append(f"### {header}")
        for n in group:
            lines.append(f"  [{n.status}] {n.label}: {n.summary}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_update_output(sg: SessionGraph, cfg: FishBridgeConfig) -> None:
    """Re-compile and write to the instruction file if it already exists."""
    out_path = _project / cfg.output.default_file
    if not out_path.exists():
        return
    compiler = ActiveThreadCompiler(sg.session_id, cfg.output.token_budget)
    compiler.write(sg.all_nodes(), sg.all_edges(), out_path, turn_count=len(sg.all_nodes()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server (called by `uvx fish-bridge-mcp` / `fish-bridge serve-mcp`)."""
    global _project
    # Use CWD as the project root when launched as a server
    _project = Path.cwd()
    mcp.run()


if __name__ == "__main__":
    main()
