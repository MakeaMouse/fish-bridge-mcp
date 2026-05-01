"""fish_bridge CLI — Typer-based command interface.

Commands:
  init      — initialize for a project (creates managed block in instruction file)
  ingest    — ingest turns from copilot JSONL or paste/file
  compile   — write compressed graph to instruction file
  show      — display active thread in terminal with Rich
  watch     — tail Copilot JSONL and auto-update on new turns
  config    — show or switch configuration
"""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from fish_bridge.config import build_backend, get_data_dir, load_config
from fish_bridge.graph.schema import NodeStatus, NodeType, RawTurn
from fish_bridge.graph.session import SessionGraph

app  = typer.Typer(help="fish_bridge — session-scoped knowledge graph for AI chat compression.")
console = Console()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_session_id(session_name: str | None, project: Path) -> str:
    if session_name:
        return session_name
    basename = project.resolve().name.lower().replace(" ", "-")
    today    = date.today().isoformat()
    return f"{basename}-{today}"


def _open_session(session_id: str, config_path: Path | None = None) -> tuple[SessionGraph, object]:
    cfg      = load_config(config_path)
    data_dir = get_data_dir(cfg)
    sg       = SessionGraph.open(session_id, data_dir)
    return sg, cfg


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    project:     Path = typer.Option(Path("."), "--project",      help="Project root directory."),
    tool:        str  = typer.Option("copilot",  "--tool",        help="Target AI tool: copilot | claude."),
    output:      Optional[Path] = typer.Option(None, "--output",  help="Custom output file path."),
    session_name: Optional[str] = typer.Option(None, "--session-name", help="Session identifier."),
) -> None:
    """Initialize fish_bridge for a project."""
    cfg = load_config()
    session_id = _resolve_session_id(session_name, project)

    if output:
        out_file = output
    elif tool == "claude":
        out_file = project / "CLAUDE.md"
    else:
        out_file = project / ".github" / "copilot-instructions.md"

    # Create the file with managed block if it doesn't exist
    from fish_bridge.compiler.active_thread import ActiveThreadCompiler, _BLOCK_START, _BLOCK_END
    compiler = ActiveThreadCompiler(session_id, cfg.output.token_budget)

    if not out_file.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            "# Project Instructions\n\n"
            "<!-- Add your project-specific AI instructions above this line. -->\n\n",
            encoding="utf-8",
        )

    content = out_file.read_text(encoding="utf-8")
    if _BLOCK_START not in content:
        compiler.write([], [], out_file, turn_count=0)

    console.print(f"[green]✓[/green] Initialized session [bold]{session_id}[/bold]")
    console.print(f"[dim]  Output file: {out_file}[/dim]")
    console.print(f"[dim]  Backend:     {cfg.extraction.backend}[/dim]")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    source:       Optional[str]  = typer.Option(None,   "--source",       help="Source type: copilot | paste | file."),
    file_path:    Optional[Path] = typer.Option(None,   "--file",         help="Path to export file (Claude JSON, text)."),
    session_name: Optional[str]  = typer.Option(None,   "--session",      help="Copilot session ID (JSONL file stem)."),
    workspace:    Optional[Path] = typer.Option(None,   "--workspace",    help="VS Code workspace path for Copilot auto-discovery."),
    project:      Path           = typer.Option(Path("."), "--project",   help="Project root."),
    output:       Optional[Path] = typer.Option(None,   "--output",       help="Instruction file to update after compile."),
    no_compile:   bool           = typer.Option(False,  "--no-compile",   help="Ingest only; skip compile step."),
    config_file:  Optional[Path] = typer.Option(None,   "--config",       help="Config file path."),
) -> None:
    """Ingest chat turns and update the session graph."""
    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    turns: list[RawTurn] = []

    effective_source = source or ("copilot" if file_path is None else "file")

    if effective_source == "copilot":
        from fish_bridge.ingestors.copilot import CopilotTranscriptIngestor
        ingestor = CopilotTranscriptIngestor()
        ws = str(workspace) if workspace else str(project.resolve())
        try:
            turns = ingestor.ingest(workspace_path=ws, session_id=session_name)
        except RuntimeError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)

    elif effective_source in ("file", "claude"):
        from fish_bridge.ingestors.chat import ChatTurnIngestor
        if not file_path:
            console.print("[red]Error:[/red] --file is required for --source file.")
            raise typer.Exit(1)
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(file_path=file_path, session_id=session_id)

    elif effective_source == "paste":
        from fish_bridge.ingestors.chat import ChatTurnIngestor
        console.print("[dim]Opening editor — paste your chat exchange, save and quit.[/dim]")
        text = ChatTurnIngestor.open_editor()
        if not text:
            console.print("[yellow]No text provided — nothing ingested.[/yellow]")
            raise typer.Exit(0)
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(text=text, session_id=session_id)

    else:
        console.print(f"[red]Unknown source:[/red] {effective_source!r}. Use: copilot | paste | file.")
        raise typer.Exit(1)

    if not turns:
        console.print("[yellow]No turns found to ingest.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[dim]Ingesting {len(turns)} turn(s) via [bold]{cfg.extraction.backend}[/bold] backend...[/dim]")

    backend = build_backend(cfg)
    total_nodes = total_edges = 0

    for turn in turns:
        try:
            nodes, edges = backend.extract(turn, cfg.extraction.exclude_patterns)
            stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
            total_nodes += len(stored_nodes)
            total_edges += len(stored_edges)
        except Exception as exc:
            console.print(f"[yellow]  ⚠ Turn {turn.turn_number} skipped: {exc}[/yellow]")
            continue

    console.print(f"[green]✓[/green] Ingested {len(turns)} turns → {total_nodes} nodes, {total_edges} edges")

    if not no_compile:
        _do_compile(sg, cfg, project, output, session_id)

    sg.close()


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

@app.command()
def compile(
    project:     Path           = typer.Option(Path("."), "--project",  help="Project root."),
    output:      Optional[Path] = typer.Option(None,      "--output",  help="Output file path."),
    session_name: Optional[str] = typer.Option(None,      "--session", help="Session identifier."),
    config_file: Optional[Path] = typer.Option(None,      "--config",  help="Config file path."),
) -> None:
    """Write compressed active graph to the instruction file."""
    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    _do_compile(sg, cfg, project, output, session_id)
    sg.close()


def _do_compile(sg: SessionGraph, cfg, project: Path, output: Path | None, session_id: str) -> None:
    from fish_bridge.compiler.active_thread import ActiveThreadCompiler

    out_file = output or (project / cfg.output.default_file)
    compiler = ActiveThreadCompiler(session_id, cfg.output.token_budget)
    all_nodes = sg.all_nodes()
    all_edges = sg.all_edges()
    compiler.write(all_nodes, all_edges, out_file, turn_count=len(all_nodes))
    console.print(f"[green]✓[/green] Compiled → [bold]{out_file}[/bold]  ({len(all_nodes)} nodes)")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@app.command()
def show(
    project:      Path          = typer.Option(Path("."), "--project",    help="Project root."),
    session_name: Optional[str] = typer.Option(None,     "--session",    help="Session identifier."),
    all_nodes:    bool          = typer.Option(False,    "--all",        help="Show all nodes including resolved."),
    node_type:    Optional[str] = typer.Option(None,     "--type",       help="Filter by node type."),
    unconfirmed:  bool          = typer.Option(False,    "--unconfirmed",help="Show only unconfirmed nodes."),
    config_file:  Optional[Path]= typer.Option(None,     "--config",     help="Config file path."),
) -> None:
    """Display the active session graph in the terminal."""
    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    nodes = sg.all_nodes() if all_nodes else sg.active_nodes()

    if unconfirmed:
        nodes = [n for n in nodes if NodeStatus(n.status) == NodeStatus.UNCONFIRMED]

    if node_type:
        try:
            nt = NodeType(node_type)
            nodes = [n for n in nodes if n.type == nt]
        except ValueError:
            console.print(f"[red]Unknown node type:[/red] {node_type!r}")
            raise typer.Exit(1)

    if not nodes:
        console.print("[dim]No nodes to display.[/dim]")
        sg.close()
        return

    table = Table(
        title=f"[bold]fish_bridge — {session_id}[/bold]  ({len(nodes)} nodes)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Type",       style="dim",       width=10)
    table.add_column("Status",     style="dim",       width=12)
    table.add_column("Label",      style="bold",      width=35)
    table.add_column("Summary",    style="",          width=55)
    table.add_column("Conf",       style="dim",       width=6,  justify="right")

    _STATUS_COLOUR = {
        "active":      "green",
        "proposed":    "yellow",
        "adopted":     "blue",
        "rejected":    "red",
        "resolved":    "dim",
        "fixed":       "dim",
        "done":        "dim",
        "deferred":    "dim yellow",
        "pending":     "cyan",
        "in_progress": "green",
        "blocked":     "red",
        "unconfirmed": "dim red",
        "conflicted":  "bold red",
    }
    _TYPE_COLOUR = {
        "question": "orange1",
        "decision": "blue",
        "concept":  "grey70",
        "skill":    "purple",
        "file":     "cyan",
        "error":    "red",
        "task":     "green",
    }

    for n in nodes:
        t_col = _TYPE_COLOUR.get(str(n.type), "")
        s_col = _STATUS_COLOUR.get(str(n.status), "")
        table.add_row(
            f"[{t_col}]{n.type}[/{t_col}]",
            f"[{s_col}]{n.status}[/{s_col}]",
            n.label,
            n.summary[:120],
            f"{n.confidence:.2f}",
        )

    console.print(table)
    sg.close()


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

@app.command()
def watch(
    source:       str           = typer.Option("copilot", "--source",    help="Source type (currently: copilot)."),
    workspace:    Optional[Path]= typer.Option(None,      "--workspace", help="VS Code workspace path."),
    project:      Path          = typer.Option(Path("."), "--project",   help="Project root."),
    output:       Optional[Path]= typer.Option(None,      "--output",   help="Instruction file to update."),
    session_name: Optional[str] = typer.Option(None,      "--session",  help="Session identifier."),
    poll:         float         = typer.Option(2.0,        "--poll",     help="Poll interval in seconds."),
    config_file:  Optional[Path]= typer.Option(None,      "--config",   help="Config file path."),
) -> None:
    """Tail Copilot JSONL and update the instruction file on new turns."""
    if source != "copilot":
        console.print(f"[red]watch only supports --source copilot (got {source!r})[/red]")
        raise typer.Exit(1)

    from fish_bridge.ingestors.copilot import CopilotTranscriptIngestor

    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    backend    = build_backend(cfg)
    ingestor   = CopilotTranscriptIngestor()
    ws         = str(workspace) if workspace else str(project.resolve())

    console.print(f"[green]Watching[/green] Copilot session for [bold]{session_id}[/bold] (Ctrl+C to stop)")

    try:
        for turn in ingestor.watch(workspace_path=ws, session_id=session_name, poll_interval=poll):
            try:
                nodes, edges = backend.extract(turn, cfg.extraction.exclude_patterns)
                sg.merge_extraction(nodes, edges)
                _do_compile(sg, cfg, project, output, session_id)
                console.print(f"  [dim]turn {turn.turn_number} → {len(nodes)} nodes[/dim]")
            except Exception as exc:
                console.print(f"  [yellow]⚠ Turn {turn.turn_number}: {exc}[/yellow]")
    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/dim]")
    finally:
        sg.close()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@app.command("config")
def config_cmd(
    show:    bool         = typer.Option(False, "--show",    help="Display current config."),
    backend: Optional[str]= typer.Option(None,  "--backend", help="Switch backend: claude | openai | openai-compatible."),
) -> None:
    """Show or update fish_bridge configuration."""
    cfg = load_config()

    if backend:
        _write_config_backend(backend)
        console.print(f"[green]✓[/green] Backend set to [bold]{backend}[/bold]")
        return

    console.print(f"[bold]Backend:[/bold]          {cfg.extraction.backend}")
    console.print(f"[bold]Claude model:[/bold]     {cfg.extraction.claude.model}")
    console.print(f"[bold]OpenAI model:[/bold]     {cfg.extraction.openai.model}")
    console.print(f"[bold]Token budget:[/bold]     {cfg.output.token_budget}")
    console.print(f"[bold]Default output:[/bold]   {cfg.output.default_file}")
    console.print(f"[bold]Data dir:[/bold]         {cfg.data_dir}")
    console.print(f"[bold]Exclude patterns:[/bold] {cfg.extraction.exclude_patterns or '(none)'}")


def _write_config_backend(backend: str) -> None:
    import yaml
    from fish_bridge.config import _default_config_path

    config_path = _default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            raw = {}
    else:
        raw = {}

    raw.setdefault("extraction", {})["backend"] = backend
    config_path.write_text(yaml.dump(raw, default_flow_style=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
