"""fish_bridge CLI — Typer-based command interface.

Commands:
  init      — initialize for a project (creates managed block in instruction file)
  ingest    — ingest turns from copilot JSONL or paste/file
  compile   — write compressed graph to instruction file (Mode A, B, or C)
  digest    — generate HANDOVER.md digest (Mode C)
  show      — display active thread in terminal with Rich
  watch     — tail Copilot JSONL and auto-update on new turns
  context   — print (or copy) compiled context to stdout / clipboard
  verify    — health-check fish_bridge setup for the current project
  serve     — start local web graph viewer (Cytoscape.js)
  serve-mcp — start the MCP server for agent-mode real-time capture
  session   — manage sessions (list / status / switch / new / rename)
  export    — export session as .chatgraph.json
  import    — import a prior session graph
  merge     — merge an external knowledge source
  config    — show or switch configuration
  resolve   — mark a node as resolved/done/fixed
  defer     — park a node (removes from active compilation)
  add       — manually add a node to the session graph
  conflict  — show or resolve conflicted nodes
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from fish_bridge.config import (
    build_backend,
    check_file_path_deprecations,
    get_active_session_id,
    get_data_dir,
    load_config,
    write_config_output,
)
from fish_bridge.graph.schema import NodeStatus, NodeType, RawTurn
from fish_bridge.graph.session import SessionGraph
from fish_bridge.platforms import PLATFORM_ADAPTERS, list_tools, resolve_tool

app  = typer.Typer(help="fish_bridge — session-scoped knowledge graph for AI chat compression.")
console = Console()

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_session_id(session_name: str | None, project: Path) -> str:
    if session_name:
        return session_name
    return get_active_session_id(project.resolve())


def _open_session(session_id: str, config_path: Path | None = None) -> tuple[SessionGraph, object]:
    cfg      = load_config(config_path)
    data_dir = get_data_dir(cfg)
    sg       = SessionGraph.open(session_id, data_dir, dedup_config=cfg.dedup)
    return sg, cfg


def _warn_stale_models(cfg) -> None:
    """Print warnings for deprecated models and stale file paths in the active config."""
    from fish_bridge.config import check_model_staleness
    stale = check_model_staleness(cfg)
    for backend_name, model_name, fix in stale:
        console.print(
            f"[yellow]⚠  Deprecated model:[/yellow] [bold]{backend_name}[/bold] backend uses "
            f"[red]{model_name!r}[/red].\n"
            f"   {fix}\n"
            f"   Update with: [bold]fish-bridge config --backend {backend_name}[/bold] "
            f"or edit [bold]~/.fish_bridge/config.yaml[/bold]"
        )
    # C2: file-path deprecation warnings
    stale_paths = check_file_path_deprecations(cfg)
    for field_name, msg in stale_paths:
        console.print(
            f"[yellow]⚠  Config warning:[/yellow] [bold]{field_name}[/bold]\n"
            f"   {msg}"
        )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    project:     Path = typer.Option(Path("."), "--project",      help="Project root directory."),
    tool:        str  = typer.Option("copilot",  "--tool",        help=(
        "Target AI tool. One of: "
        + ", ".join(list_tools())
        + ".  Aliases like 'idea', 'pycharm', 'claude-code' are also accepted."
    )),
    output:      Optional[Path] = typer.Option(None, "--output",  help="Custom output file path (overrides tool default)."),
    session_name: Optional[str] = typer.Option(None, "--session-name", help="Session identifier."),
) -> None:
    """Initialize fish_bridge for a project."""
    cfg = load_config()
    session_id = _resolve_session_id(session_name, project)

    # Resolve tool via platform registry (A2)
    canonical = resolve_tool(tool)
    if canonical is None:
        console.print(
            f"[red]Unknown tool:[/red] {tool!r}\n"
            f"Known tools: {', '.join(list_tools())}"
        )
        raise typer.Exit(1)
    adapter = PLATFORM_ADAPTERS[canonical]

    # Determine primary output file
    if output:
        out_file = output
        extra_targets: list[str] = []
    else:
        # Use platform-preferred targets
        targets = list(adapter.output_targets)
        out_file = project / targets[0]
        extra_targets = [t for t in targets[1:]]

    # Update config extra_targets if the adapter has multiple targets (A1)
    if extra_targets:
        cfg.output.extra_targets = extra_targets
        # A1-fix: persist to YAML so future compile/watch/serve use the same targets
        write_config_output(extra_targets=extra_targets)

    # Ensure .fish_bridge/ is gitignored so session-specific runtime data
    # (bugs, internal API names) never leaks into the team repo accidentally.
    fish_bridge_dir = project / ".fish_bridge"
    gitignore_path  = fish_bridge_dir / ".gitignore"
    if not gitignore_path.exists():
        fish_bridge_dir.mkdir(parents=True, exist_ok=True)
        gitignore_path.write_text(
            "# fish_bridge local session files — do not commit\n"
            "context.md\n"
            "context.prompt.md\n"
            "*.db\n"
            "session.lock\n"
            ".cloud_backend_warned\n"
            ".env\n",
            encoding="utf-8",
        )

    # B3: write session env file for cross-IDE sharing
    env_file = fish_bridge_dir / ".env"
    fish_bridge_dir.mkdir(parents=True, exist_ok=True)
    env_file.write_text(
        f"# fish_bridge session env — source this in your shell or IDE terminal\n"
        f"FISH_BRIDGE_SESSION={session_id}\n"
        f"FISH_BRIDGE_PROJECT={project.resolve()}\n",
        encoding="utf-8",
    )

    # For VS Code / copilot tool: add nested link reference in
    # .github/copilot-instructions.md so Copilot still picks up the context.
    if adapter.nested_link_supported and not output:
        shared_ref = project / ".github" / "copilot-instructions.md"
        shared_ref.parent.mkdir(parents=True, exist_ok=True)
        rel_path = Path(adapter.output_targets[0])
        ref_line = f"\n[//]: # (fish_bridge context — VS Code 1.99+ nested link resolution)\n[fish_bridge context]({rel_path})\n"
        if shared_ref.exists():
            existing = shared_ref.read_text(encoding="utf-8")
            if str(rel_path) not in existing:
                shared_ref.write_text(existing.rstrip() + ref_line, encoding="utf-8")
        else:
            shared_ref.write_text(
                "# Project Instructions\n\n"
                "<!-- Add your project-specific AI instructions above this line. -->\n"
                + ref_line,
                encoding="utf-8",
            )

    # For non-local-mode platforms that use .github/copilot-instructions.md directly,
    # set delivery_mode=shared so _do_compile writes there (not via nested link).
    if not adapter.local_mode_safe and not output:
        cfg.output.delivery_mode = "shared"
        # Override shared_context_file to point to the platform's primary target
        cfg.output.shared_context_file = adapter.output_targets[0]
        # A1-fix: persist delivery_mode and shared_context_file changes
        write_config_output(
            delivery_mode="shared",
            shared_context_file=adapter.output_targets[0],
        )

    # Create the instruction file with managed block if not yet present
    from fish_bridge.compiler.active_thread import ActiveThreadCompiler, _BLOCK_START
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

    # Also init any extra targets
    for extra_path in extra_targets:
        extra_file = project / extra_path
        if not extra_file.exists():
            extra_file.parent.mkdir(parents=True, exist_ok=True)
            extra_file.write_text(
                "# Project Instructions\n\n"
                "<!-- Add your project-specific AI instructions above this line. -->\n\n",
                encoding="utf-8",
            )
        extra_content = extra_file.read_text(encoding="utf-8")
        if _BLOCK_START not in extra_content:
            compiler.write([], [], extra_file, turn_count=0)

    # Print summary
    console.print(f"[green]✓[/green] Initialized session [bold]{session_id}[/bold] for [bold]{adapter.display_name}[/bold]")
    console.print(f"[dim]  Primary output: {out_file}[/dim]")
    if extra_targets:
        for t in extra_targets:
            console.print(f"[dim]  Also writes to: {project / t}[/dim]")
    console.print(f"[dim]  Backend: {cfg.extraction.backend}[/dim]")
    if adapter.mcp_supported:
        console.print("[dim]  MCP server (optional): fish-bridge serve-mcp[/dim]")
    if adapter.setup_note:
        for line in adapter.setup_note.splitlines():
            console.print(f"[dim]  {line}[/dim]")
    if not adapter.local_mode_safe:
        console.print(
            "[yellow]  ⚠  Commit the output file(s) to your repo so "
            f"{adapter.display_name} can read them.[/yellow]"
        )
    # B3: cross-IDE hint
    if adapter.mcp_supported:
        console.print(
            f"[dim]  Cross-IDE session ID: [bold]{session_id}[/bold]  "
            f"(sourced from .fish_bridge/.env)[/dim]"
        )


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
    topic:        Optional[str]  = typer.Option(None,   "--topic",        help="Tag all extracted nodes with this topic label."),
) -> None:
    """Ingest chat turns and update the session graph."""
    cfg        = load_config(config_file)
    _warn_stale_models(cfg)
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

    elif effective_source in ("jetbrains", "idea", "pycharm", "webstorm"):
        # C3: JetBrains Copilot Chat has no transcript file — guide the user through
        # manually copying the conversation from the chat panel.
        from fish_bridge.ingestors.chat import ChatTurnIngestor
        console.print(
            "\n[bold]JetBrains Copilot Chat — manual copy instructions:[/bold]\n"
            "  1. In your JetBrains IDE, open the [bold]GitHub Copilot Chat[/bold] panel.\n"
            "  2. Click inside the chat panel, then press [bold]Ctrl+A[/bold] (⌘A on macOS)\n"
            "     to select all text in the panel.\n"
            "  3. Press [bold]Ctrl+C[/bold] (⌘C) to copy.\n"
            "  4. When the editor opens below, paste with [bold]Ctrl+V[/bold] (⌘V),\n"
            "     save (usually :wq or Ctrl+S), then quit.\n"
            "\n"
            "[dim]Tip: JetBrains formats turns as 'Me:' / 'GitHub Copilot:' prefixes.\n"
            "Both prefixes are recognised automatically.[/dim]\n"
        )
        text = ChatTurnIngestor.open_editor(
            prompt="# Paste your JetBrains Copilot Chat conversation below, then save and quit.\n"
                   "# Expected format:\n"
                   "#   Me: <your message>\n"
                   "#   GitHub Copilot: <response>\n\n"
        )
        if not text:
            console.print("[yellow]No text provided — nothing ingested.[/yellow]")
            raise typer.Exit(0)
        ingestor = ChatTurnIngestor()
        turns = ingestor.ingest(text=text, session_id=session_id, source="jetbrains")

    else:
        console.print(
            f"[red]Unknown source:[/red] {effective_source!r}. "
            "Use: copilot | paste | file | jetbrains."
        )
        raise typer.Exit(1)

    if not turns:
        console.print("[yellow]No turns found to ingest.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[dim]Ingesting {len(turns)} turn(s) via [bold]{cfg.extraction.backend}[/bold] backend...[/dim]")

    backend = build_backend(cfg)
    total_nodes = total_edges = 0
    skipped_turns = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} turns"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(turns))
        for turn in turns:
            try:
                nodes, edges = backend.extract(turn, cfg.extraction.exclude_patterns)
                # B2: tag nodes with topic if --topic was supplied
                if topic:
                    for n in nodes:
                        n.metadata["topic"] = topic
                stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
                total_nodes += len(stored_nodes)
                total_edges += len(stored_edges)
            except Exception as exc:
                console.print(f"[yellow]  ⚠ Turn {turn.turn_number} skipped: {exc}[/yellow]")
                skipped_turns += 1
            finally:
                progress.advance(task)

    # C4: quality summary
    ok_turns = len(turns) - skipped_turns
    console.print(
        f"[green]✓[/green] Ingested [bold]{ok_turns}/{len(turns)}[/bold] turns"
        + (f" ([yellow]{skipped_turns} skipped[/yellow])" if skipped_turns else "")
        + f" → [bold]{total_nodes}[/bold] nodes, [bold]{total_edges}[/bold] edges"
    )
    if topic:
        console.print(f"[dim]  ↳ All nodes tagged with topic: [bold]{topic}[/bold][/dim]")

    # Node-type breakdown
    all_stored = sg.all_nodes()
    if all_stored:
        from collections import Counter
        type_counts = Counter(
            (n.type if isinstance(n.type, str) else n.type.value) for n in all_stored
        )
        breakdown = "  ".join(
            f"[dim]{k}:{v}[/dim]"
            for k, v in sorted(type_counts.items())
        )
        console.print(f"[dim]  ↳ Session totals: {breakdown}[/dim]")

        unconfirmed_count = sum(
            1 for n in all_stored
            if (n.status if isinstance(n.status, str) else n.status.value) == "unconfirmed"
        )
        conflict_count = sum(
            1 for n in all_stored
            if (n.status if isinstance(n.status, str) else n.status.value) == "conflicted"
        )
        if unconfirmed_count:
            console.print(
                f"[yellow]  ↳ {unconfirmed_count} unconfirmed node(s) — "
                "run: [bold]fish-bridge show --unconfirmed[/bold][/yellow]"
            )
        if conflict_count:
            console.print(
                f"[red]  ↳ {conflict_count} conflicted node(s) need resolution — "
                "run: [bold]fish-bridge conflict show[/bold][/red]"
            )

    # Fix 5: Remediate any existing low-confidence nodes (e.g. from older sessions)
    remediated = sg.remediate_low_confidence()
    if remediated:
        console.print(f"[dim]  ↳ {remediated} low-confidence node(s) marked unconfirmed[/dim]")

    # Post-ingest graph quality passes (offline, no LLM)
    new_edges = sg.infer_cross_session_edges()
    if new_edges:
        console.print(f"[dim]  ↳ {new_edges} cross-session edge(s) inferred[/dim]")
    upgraded = sg.upgrade_relates_to_edges()
    if upgraded:
        console.print(f"[dim]  ↳ {upgraded} relates-to edge(s) upgraded to typed relations[/dim]")
    healed = sg.heal_orphan_edges()
    if healed:
        console.print(f"[dim]  ↳ {healed} orphan node(s) connected[/dim]")

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
    mode:        str            = typer.Option("active",  "--mode",    help="Compiler mode: active | focus | digest."),
    query:       Optional[str]  = typer.Option(None,      "--query",   help="Query for --mode focus."),
    topic:       Optional[str]  = typer.Option(None,      "--topic",   help="Filter to nodes tagged with this topic (B2)."),
    config_file: Optional[Path] = typer.Option(None,      "--config",  help="Config file path."),
) -> None:
    """Write compressed active graph to the instruction file."""
    cfg        = load_config(config_file)
    _warn_stale_models(cfg)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    # B2: apply topic filter before compilation
    def _topic_filter(nodes):
        if not topic:
            return nodes
        filtered = [n for n in nodes if n.metadata.get("topic") == topic]
        if not filtered:
            console.print(
                f"[yellow]No nodes tagged with topic '{topic}'. "
                "Showing all nodes.[/yellow]"
            )
            return nodes
        console.print(f"[dim]Topic filter '{topic}': {len(filtered)} of {len(nodes)} nodes[/dim]")
        return filtered

    if mode == "focus":
        from fish_bridge.compiler.focus import FocusCompiler
        out_file = output or (project / cfg.output.default_file)
        compiler = FocusCompiler(session_id)
        xml = compiler.compile(_topic_filter(sg.all_nodes()), sg.all_edges(), query=query or "")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(xml, encoding="utf-8")
        console.print(f"[green]✓[/green] Focus compiled → [bold]{out_file}[/bold]")
    elif mode == "digest":
        from fish_bridge.compiler.digest import DigestCompiler
        out_file = output or (project / "HANDOVER.md")
        compiler = DigestCompiler(session_id)
        compiler.write(_topic_filter(sg.all_nodes()), sg.all_edges(), out_file)
        console.print(f"[green]✓[/green] Digest compiled → [bold]{out_file}[/bold]")
    else:
        _do_compile(sg, cfg, project, output, session_id, topic_filter=_topic_filter)
    sg.close()


def _do_compile(
    sg: SessionGraph,
    cfg,
    project: Path,
    output: Path | None,
    session_id: str,
    topic_filter=None,
) -> None:
    from fish_bridge.compiler.active_thread import ActiveThreadCompiler

    raw_nodes = sg.all_nodes()
    all_nodes = topic_filter(raw_nodes) if topic_filter else raw_nodes
    all_edges = sg.all_edges()
    compiler  = ActiveThreadCompiler(session_id, cfg.output.token_budget)

    if output:
        # Explicit --output overrides everything
        compiler.write(all_nodes, all_edges, output, turn_count=len(all_nodes))
        console.print(f"[green]✓[/green] Compiled → [bold]{output}[/bold]  ({len(all_nodes)} nodes)")
        return

    # Write to every configured output target (A1 multi-target)
    targets = [project / p for p in cfg.output.all_output_files]
    for out_file in targets:
        compiler.write(all_nodes, all_edges, out_file, turn_count=len(all_nodes))

    if len(targets) == 1:
        console.print(f"[green]✓[/green] Compiled → [bold]{targets[0]}[/bold]  ({len(all_nodes)} nodes)")
    else:
        console.print(f"[green]✓[/green] Compiled → {len(targets)} target(s)  ({len(all_nodes)} nodes)")
        for t in targets:
            console.print(f"           [dim]{t}[/dim]")


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
    topic:        Optional[str] = typer.Option(None,     "--topic",      help="Filter to nodes tagged with this topic."),
    config_file:  Optional[Path]= typer.Option(None,     "--config",     help="Config file path."),
) -> None:
    """Display the active session graph in the terminal."""
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    nodes = sg.all_nodes() if all_nodes else sg.active_nodes()

    if unconfirmed:
        nodes = [n for n in nodes if NodeStatus(n.status) == NodeStatus.UNCONFIRMED]

    if topic:
        nodes = [n for n in nodes if n.metadata.get("topic") == topic]
        if not nodes:
            console.print(f"[yellow]No nodes tagged with topic '{topic}'.[/yellow]")
            sg.close()
            return

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
    topic:        Optional[str] = typer.Option(None,       "--topic",    help="Tag all auto-ingested turns with this topic label."),
    config_file:  Optional[Path]= typer.Option(None,      "--config",   help="Config file path."),
) -> None:
    """Tail Copilot JSONL and update the instruction file on new turns."""
    if source != "copilot":
        console.print(f"[red]watch only supports --source copilot (got {source!r})[/red]")
        raise typer.Exit(1)

    from fish_bridge.ingestors.copilot import CopilotTranscriptIngestor

    cfg        = load_config(config_file)
    _warn_stale_models(cfg)
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
                # B2/watch: tag nodes with topic if --topic was supplied
                if topic:
                    for n in nodes:
                        n.metadata["topic"] = topic
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

    from fish_bridge.config import check_model_staleness, MODEL_DEFAULTS
    active = cfg.extraction.backend
    active_model = {
        "claude":  cfg.extraction.claude.model,
        "openai":  cfg.extraction.openai.model,
        "gemini":  cfg.extraction.gemini.model,
        "local":   cfg.extraction.local.model,
        "hybrid":  f"{cfg.extraction.local.model} + {cfg.extraction.gemini.model}",
    }.get(active, active)

    console.print(f"[bold]Backend:[/bold]          {active}")
    console.print(f"[bold]Active model:[/bold]     {active_model}  [dim](current default: {MODEL_DEFAULTS.get(active, 'n/a')})[/dim]")
    console.print(f"[bold]Claude model:[/bold]     {cfg.extraction.claude.model}")
    console.print(f"[bold]OpenAI model:[/bold]     {cfg.extraction.openai.model}")
    console.print(f"[bold]Gemini model:[/bold]     {cfg.extraction.gemini.model}")
    console.print(f"[bold]Local model:[/bold]      {cfg.extraction.local.model}")
    console.print(f"[bold]Embed model:[/bold]      {cfg.extraction.local.embed_model}")
    console.print(f"[bold]Token budget:[/bold]     {cfg.output.token_budget}")
    console.print(f"[bold]Default output:[/bold]   {cfg.output.default_file}")
    if cfg.output.extra_targets:
        console.print(f"[bold]Extra targets:[/bold]    {', '.join(cfg.output.extra_targets)}")
    console.print(f"[bold]Data dir:[/bold]         {cfg.data_dir}")
    console.print(f"[bold]Exclude patterns:[/bold] {len(cfg.extraction.exclude_patterns)} pattern(s) active")

    stale = check_model_staleness(cfg)
    if stale:
        console.print()
        for bname, model_name, fix in stale:
            console.print(f"[yellow]⚠  {bname}[/yellow]: [red]{model_name!r}[/red] is deprecated. {fix}")


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
# export
# ---------------------------------------------------------------------------

@app.command("export")
def export_cmd(
    project:      Path           = typer.Option(Path("."), "--project",  help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session",  help="Session identifier."),
    output:       Optional[Path] = typer.Option(None,     "--output",   help="Output .chatgraph.json path."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",   help="Config file path."),
) -> None:
    """Export the session graph to a portable .chatgraph.json file."""
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    out_path = output or (project / f"{session_id}.chatgraph.json")
    sg.export_json(out_path)
    console.print(f"[green]✓[/green] Exported → [bold]{out_path}[/bold]  ({len(sg.all_nodes())} nodes, {len(sg.all_edges())} edges)")
    sg.close()


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------

@app.command("import")
def import_cmd(
    file_path:    Path           = typer.Argument(...,     help="Path to .chatgraph.json file to import."),
    project:      Path           = typer.Option(Path("."), "--project",  help="Project root (target session)."),
    session_name: Optional[str]  = typer.Option(None,     "--session",  help="Target session identifier."),
    no_compile:   bool           = typer.Option(False,    "--no-compile", help="Skip compile after import."),
    output:       Optional[Path] = typer.Option(None,     "--output",   help="Instruction file to update."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",   help="Config file path."),
) -> None:
    """Import a prior session graph into the current session."""
    from fish_bridge.ingestors.session_file import SessionFileIngestor

    if not file_path.exists():
        console.print(f"[red]Error:[/red] File not found: {file_path}")
        raise typer.Exit(1)

    # Show summary before importing
    try:
        summary = SessionFileIngestor.summary(file_path)
        console.print(
            f"[dim]Importing session [bold]{summary['session_id']}[/bold] "
            f"({summary['node_count']} nodes, {summary['edge_count']} edges)[/dim]"
        )
    except Exception:
        pass

    nodes, edges = SessionFileIngestor.load(file_path)
    if not nodes:
        console.print("[yellow]No nodes found in file.[/yellow]")
        raise typer.Exit(0)

    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)

    console.print(
        f"[green]✓[/green] Imported → session [bold]{session_id}[/bold]  "
        f"({len(stored_nodes)} nodes merged, {len(stored_edges)} edges)"
    )

    if not no_compile:
        _do_compile(sg, cfg, project, output, session_id)

    sg.close()


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

@app.command("merge")
def merge_cmd(
    source:       str            = typer.Option(...,       "--source",   help="Source: document | codebase | obsidian | session | deps | testout | iac | openapi | pdf."),
    file_path:    Optional[Path] = typer.Option(None,     "--file",     help="File path (for document/session/testout/openapi sources)."),
    vault:        Optional[Path] = typer.Option(None,     "--vault",    help="Obsidian vault path."),
    path:         Optional[Path] = typer.Option(None,     "--path",     help="Directory path (for codebase/deps sources)."),
    tag:          Optional[str]  = typer.Option(None,     "--tag",      help="Tag filter (Obsidian)."),
    folder:       Optional[str]  = typer.Option(None,     "--folder",   help="Folder filter (Obsidian)."),
    commits:      int            = typer.Option(20,        "--commits",  help="Number of git commits (codebase)."),
    no_docs:      bool           = typer.Option(False,    "--no-docs",  help="Skip README/HANDOVER docs (codebase)."),
    project:      Path           = typer.Option(Path("."), "--project",  help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session",  help="Session identifier."),
    no_compile:   bool           = typer.Option(False,    "--no-compile", help="Skip compile after merge."),
    output:       Optional[Path] = typer.Option(None,     "--output",   help="Instruction file to update."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",   help="Config file path."),
) -> None:
    """Merge an external knowledge source into the current session graph."""
    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    if source == "session":
        # Direct node/edge import — no LLM needed
        if not file_path:
            console.print("[red]Error:[/red] --file required for --source session.")
            raise typer.Exit(1)
        from fish_bridge.ingestors.session_file import SessionFileIngestor
        nodes, edges = SessionFileIngestor.load(file_path)
        stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
        console.print(f"[green]✓[/green] Merged session: {len(stored_nodes)} nodes, {len(stored_edges)} edges")

    elif source == "document":
        if not file_path:
            console.print("[red]Error:[/red] --file required for --source document.")
            raise typer.Exit(1)
        from fish_bridge.ingestors.document import DocumentIngestor
        turns = DocumentIngestor().ingest(file_path=file_path, session_id=session_id)
        _extract_and_merge(turns, sg, cfg, console)

    elif source == "codebase":
        repo_path = path or project
        from fish_bridge.ingestors.codebase import CodebaseIngestor
        turns = CodebaseIngestor().ingest(
            path=repo_path,
            session_id=session_id,
            n_commits=commits,
            include_docs=not no_docs,
        )
        _extract_and_merge(turns, sg, cfg, console)

    elif source == "obsidian":
        if not vault:
            console.print("[red]Error:[/red] --vault required for --source obsidian.")
            raise typer.Exit(1)
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        turns = ObsidianIngestor().ingest(
            vault_path=vault,
            session_id=session_id,
            tag_filter=tag,
            folder_filter=folder,
        )
        _extract_and_merge(turns, sg, cfg, console)

    elif source == "deps":
        scan_path = path or project
        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest_project(scan_path)
        if not nodes:
            console.print(f"[yellow]No supported manifest files found in {scan_path}[/yellow]")
        else:
            stored_nodes, stored_edges = sg.merge_extraction(nodes, [])
            console.print(f"[green]✓[/green] Merged {len(stored_nodes)} dependency skill nodes")

    elif source == "testout":
        if not file_path:
            console.print("[red]Error:[/red] --file required for --source testout.")
            raise typer.Exit(1)
        from fish_bridge.ingestors.testout import TestOutputIngestor
        nodes, edges = TestOutputIngestor().ingest(file_path)
        if not nodes:
            console.print("[yellow]No test failures found (or unrecognised format).[/yellow]")
        else:
            stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
            failures = sum(1 for n in stored_nodes if getattr(n, 'subtype', None) == 'test_failure')
            console.print(f"[green]✓[/green] Merged {failures} test failure node(s)")

    elif source == "iac":
        scan_path = path or project
        from fish_bridge.ingestors.iac import IaCIngestor
        nodes, edges = IaCIngestor().ingest_project(scan_path)
        if not nodes:
            console.print(f"[yellow]No IaC files found in {scan_path}[/yellow]")
        else:
            stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
            console.print(f"[green]✓[/green] Merged {len(stored_nodes)} IaC resource node(s)")

    elif source == "openapi":
        if not file_path:
            console.print("[red]Error:[/red] --file required for --source openapi.")
            raise typer.Exit(1)
        from fish_bridge.ingestors.openapi import OpenAPIIngestor
        nodes, edges = OpenAPIIngestor().ingest(file_path)
        if not nodes:
            console.print("[yellow]No endpoints or schemas found (unrecognised format?).[/yellow]")
        else:
            stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
            endpoints = sum(1 for n in stored_nodes if getattr(n, 'subtype', None) == 'endpoint')
            console.print(f"[green]✓[/green] Merged {endpoints} endpoint node(s) from OpenAPI spec")

    elif source == "pdf":
        if not file_path:
            console.print("[red]Error:[/red] --file required for --source pdf.")
            raise typer.Exit(1)
        try:
            from fish_bridge.ingestors.pdf import PDFIngestor
        except ImportError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1)
        turns = PDFIngestor().ingest(file_path=file_path, session_id=session_id)
        if not turns:
            console.print("[yellow]No text extracted from PDF.[/yellow]")
        else:
            _extract_and_merge(turns, sg, cfg, console)

    else:
        console.print(f"[red]Unknown source:[/red] {source!r}. Use: document | codebase | obsidian | session | deps | testout | iac | openapi | pdf.")
        sg.close()
        raise typer.Exit(1)

    # Fix 2: After any merge, run cross-session edge inference to connect nodes
    # that originated in different source sessions.  This is offline (no LLM /
    # no GPU required) so it is safe for all environments.
    new_edges = sg.infer_cross_session_edges()
    if new_edges:
        console.print(f"[dim]  ↳ {new_edges} cross-session edge(s) inferred[/dim]")

    # Upgrade cross-type relates-to to specific typed relations.
    upgraded = sg.upgrade_relates_to_edges()
    if upgraded:
        console.print(f"[dim]  ↳ {upgraded} relates-to edge(s) upgraded to typed relations[/dim]")

    # Heal orphan nodes using embedding similarity (or token overlap).
    healed = sg.heal_orphan_edges()
    if healed:
        console.print(f"[dim]  ↳ {healed} orphan node(s) connected[/dim]")

    # Fix 5: Mark any low-confidence nodes that slipped through as unconfirmed
    remediated = sg.remediate_low_confidence()
    if remediated:
        console.print(f"[dim]  ↳ {remediated} low-confidence node(s) marked unconfirmed[/dim]")

    if not no_compile:
        _do_compile(sg, cfg, project, output, session_id)

    sg.close()


def _extract_and_merge(turns, sg, cfg, console_obj) -> None:
    """Run LLM extraction on turns and merge into session graph."""
    if not turns:
        console_obj.print("[yellow]No turns to extract.[/yellow]")
        return
    console_obj.print(f"[dim]Extracting {len(turns)} chunk(s) via [bold]{cfg.extraction.backend}[/bold]...[/dim]")
    backend = build_backend(cfg)
    total_nodes = total_edges = 0
    for turn in turns:
        try:
            nodes, edges = backend.extract(turn, cfg.extraction.exclude_patterns)
            stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
            total_nodes += len(stored_nodes)
            total_edges += len(stored_edges)
        except Exception as exc:
            console_obj.print(f"  [yellow]⚠ chunk {turn.turn_number} skipped: {exc}[/yellow]")
    console_obj.print(f"[green]✓[/green] Merged → {total_nodes} nodes, {total_edges} edges")


# ---------------------------------------------------------------------------
# digest — generate HANDOVER.md
# ---------------------------------------------------------------------------

@app.command()
def digest(
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    output:       Optional[Path] = typer.Option(None,     "--output",  help="Output file (default: HANDOVER.md)."),
    topic:        Optional[str]  = typer.Option(None,     "--topic",   help="Filter to nodes tagged with this topic."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Generate a full session digest (HANDOVER.md)."""
    from fish_bridge.compiler.digest import DigestCompiler

    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    all_nodes = sg.all_nodes()
    if topic:
        filtered = [n for n in all_nodes if n.metadata.get("topic") == topic]
        if filtered:
            console.print(f"[dim]Topic filter '{topic}': {len(filtered)} of {len(all_nodes)} nodes[/dim]")
            all_nodes = filtered
        else:
            console.print(f"[yellow]No nodes tagged with topic '{topic}'. Showing all nodes.[/yellow]")

    out_file = output or (project / "HANDOVER.md")
    compiler = DigestCompiler(session_id)
    compiler.write(all_nodes, sg.all_edges(), out_file)
    sg.close()

    console.print(f"[green]✓[/green] Digest written → [bold]{out_file}[/bold]")


# ---------------------------------------------------------------------------
# serve — local web graph viewer
# ---------------------------------------------------------------------------

@app.command()
def serve(
    port:         int            = typer.Option(8080,      "--port",    help="HTTP port (bound to 127.0.0.1 only)."),
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    no_open:      bool           = typer.Option(False,     "--no-open", help="Don't open browser automatically."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Launch the local web graph viewer (Cytoscape.js)."""
    import json as _json
    from fish_bridge.viewer.server import run_viewer
    from fish_bridge.config import get_data_dir

    session_id = _resolve_session_id(session_name, project)
    cfg        = load_config(config_file)
    sg, _      = _open_session(session_id, config_file)
    nodes      = sg.all_nodes()
    edges      = sg.all_edges()
    sg.close()

    # Resolve db / lock paths for live node editing
    data_dir  = get_data_dir(cfg)
    db_path   = data_dir / f"{session_id}.db"
    lock_path = data_dir / "session.lock"

    # Serialise graph to JSON for the viewer
    graph_data = {
        "session_id": session_id,
        "nodes": [
            {
                "id":         n.id,
                "label":      n.label,
                "type":       n.type if isinstance(n.type, str) else n.type.value,
                "status":     n.status if isinstance(n.status, str) else n.status.value,
                "summary":    n.summary or "",
                "confidence": n.confidence,
            }
            for n in nodes
        ],
        "edges": [
            {
                "id":       e.id,
                "from_id":  e.from_id,
                "to_id":    e.to_id,
                "relation": e.relation if isinstance(e.relation, str) else e.relation.value,
                "weight":   e.weight,
            }
            for e in edges
        ],
    }

    console.print(f"[green]✓[/green] Loaded {len(nodes)} nodes, {len(edges)} edges")
    run_viewer(
        _json.dumps(graph_data),
        port=port,
        open_browser=not no_open,
        db_path=db_path if db_path.exists() else None,
        lock_path=lock_path,
        data_dir=data_dir,
    )


# ---------------------------------------------------------------------------
# resolve — mark a node as resolved
# ---------------------------------------------------------------------------

@app.command()
def resolve(
    label:        str            = typer.Argument(...,        help="Partial or full node label to resolve."),
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    note:         Optional[str]  = typer.Option(None,     "--note",    help="Optional resolution note."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Mark a node (question/task/error) as resolved/done/fixed."""
    from fish_bridge.graph.schema import NodeStatus
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    nodes      = sg.all_nodes()
    label_lower = label.lower()

    # fuzzy label match
    matches = [n for n in nodes if label_lower in n.label.lower()]
    if not matches:
        console.print(f"[yellow]No nodes matching '{label}'[/yellow]")
        sg.close()
        return
    if len(matches) > 1:
        console.print("[yellow]Multiple matches — pick one:[/yellow]")
        for n in matches:
            console.print(f"  [{n.id[:8]}] {n.label} ({n.type} / {n.status})")
        sg.close()
        return

    node = matches[0]
    # Choose terminal status based on node type
    ntype = node.type if isinstance(node.type, str) else node.type.value
    if ntype == "task":
        new_status = NodeStatus.DONE
    elif ntype == "error":
        new_status = NodeStatus.FIXED
    else:
        new_status = NodeStatus.RESOLVED

    sg.set_status(node.id, new_status, note)
    sg.close()
    console.print(f"[green]✓[/green] [{node.id[:8]}] [bold]{node.label}[/bold] → [bold]{new_status.value}[/bold]")


# ---------------------------------------------------------------------------
# defer — park a node
# ---------------------------------------------------------------------------

@app.command()
def defer(
    label:        str            = typer.Argument(...,        help="Partial or full node label to defer."),
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    note:         Optional[str]  = typer.Option(None,     "--note",    help="Optional deferral note."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Park a node — removes it from active compilation."""
    from fish_bridge.graph.schema import NodeStatus
    session_id  = _resolve_session_id(session_name, project)
    sg, _       = _open_session(session_id, config_file)
    nodes       = sg.all_nodes()
    label_lower = label.lower()

    matches = [n for n in nodes if label_lower in n.label.lower()]
    if not matches:
        console.print(f"[yellow]No nodes matching '{label}'[/yellow]")
        sg.close()
        return
    if len(matches) > 1:
        console.print("[yellow]Multiple matches — pick one:[/yellow]")
        for n in matches:
            console.print(f"  [{n.id[:8]}] {n.label} ({n.type} / {n.status})")
        sg.close()
        return

    node = matches[0]
    sg.set_status(node.id, NodeStatus.DEFERRED, note)
    sg.close()
    console.print(f"[dim]⏸[/dim] [{node.id[:8]}] [bold]{node.label}[/bold] → deferred")


# ---------------------------------------------------------------------------
# add — manually add a node
# ---------------------------------------------------------------------------

@app.command("add")
def add_node_cmd(
    label:        str            = typer.Argument(...,               help="Node label (3–6 words)."),
    node_type:    str            = typer.Option("concept",  "--type",    help="Node type: question|decision|concept|skill|file|error|task."),
    status:       str            = typer.Option("active",   "--status",  help="Initial status."),
    summary:      Optional[str]  = typer.Option(None,       "--summary", help="One-sentence description."),
    project:      Path           = typer.Option(Path("."),  "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,       "--session", help="Session identifier."),
    config_file:  Optional[Path] = typer.Option(None,       "--config",  help="Config file path."),
) -> None:
    """Manually add a node to the session graph."""
    from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType

    try:
        ntype  = NodeType(node_type)
        nstatus = NodeStatus(status)
    except ValueError as exc:
        console.print(f"[red]Invalid type or status: {exc}[/red]")
        raise typer.Exit(1)

    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    node = GraphNode(
        type=ntype,
        label=label,
        status=nstatus,
        summary=summary or "",
        confidence=1.0,
    )
    sg.add_node(node)
    sg.close()
    console.print(f"[green]✓[/green] Added [{node.id[:8]}] [bold]{label}[/bold] ({node_type} / {status})")


# ---------------------------------------------------------------------------
# conflict — show or resolve conflicted nodes
# ---------------------------------------------------------------------------

conflict_app = typer.Typer(help="Manage conflicted nodes (status reversals).")
app.add_typer(conflict_app, name="conflict")


@conflict_app.command("show")
def conflict_show(
    project:      Path          = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str] = typer.Option(None,     "--session", help="Session identifier."),
    config_file:  Optional[Path]= typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """List all conflicted nodes with their status history."""
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    nodes      = [n for n in sg.all_nodes() if
                  (n.status if isinstance(n.status, str) else n.status.value) == "conflicted"]
    sg.close()

    if not nodes:
        console.print("[green]No conflicted nodes.[/green]")
        return

    for n in nodes:
        console.print(f"\n[bold red]{n.label}[/bold red] [{n.id[:8]}]")
        console.print(f"  Type: {n.type}  Summary: {n.summary}")
        history = n.status_history or []
        if history:
            console.print("  Status history:")
            for h in history:
                ts = h.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(h.timestamp, 'strftime') else str(h.timestamp)
                note_str = f" — {h.note}" if h.note else ""
                console.print(f"    {ts}  {h.status}{note_str}")


@conflict_app.command("resolve")
def conflict_resolve(
    node_id:      str           = typer.Argument(...,           help="Node ID (first 8 chars suffice)."),
    keep:         str           = typer.Option(...,  "--keep",  help="Which status to keep: 'old' or 'new', or a specific status value."),
    project:      Path          = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str] = typer.Option(None,     "--session", help="Session identifier."),
    note:         Optional[str] = typer.Option(None,     "--note",    help="Resolution note."),
    config_file:  Optional[Path]= typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Reconcile a conflicted node by choosing which status wins."""
    from fish_bridge.graph.schema import NodeStatus
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    # Support 8-char prefix match
    all_nodes = sg.all_nodes()
    node = next((n for n in all_nodes if n.id == node_id or n.id.startswith(node_id)), None)
    if node is None:
        console.print(f"[red]Node '{node_id}' not found.[/red]")
        sg.close()
        raise typer.Exit(1)

    history = node.status_history or []
    if keep == "old":
        # Restore the status before the conflict
        pre_conflict = next(
            (h.status for h in reversed(history) if
             (h.status if isinstance(h.status, str) else h.status.value) != "conflicted"),
            None,
        )
        if pre_conflict is None:
            console.print("[red]Cannot determine previous status.[/red]")
            sg.close()
            raise typer.Exit(1)
        target_status = NodeStatus(pre_conflict if isinstance(pre_conflict, str) else pre_conflict.value)
    elif keep == "new":
        # Use the most recent non-conflicted history entry
        target_status_str = next(
            (h.status if isinstance(h.status, str) else h.status.value
             for h in history
             if (h.status if isinstance(h.status, str) else h.status.value) != "conflicted"),
            None,
        )
        if target_status_str is None:
            console.print("[red]Cannot determine new status from history.[/red]")
            sg.close()
            raise typer.Exit(1)
        target_status = NodeStatus(target_status_str)
    else:
        try:
            target_status = NodeStatus(keep)
        except ValueError:
            console.print(f"[red]Unknown status '{keep}'. Use 'old', 'new', or a valid status value.[/red]")
            sg.close()
            raise typer.Exit(1)

    sg.set_status(node.id, target_status, note or f"Conflict resolved: kept {keep}")
    sg.close()
    console.print(f"[green]✓[/green] [{node.id[:8]}] [bold]{node.label}[/bold] → [bold]{target_status.value}[/bold]")


# ---------------------------------------------------------------------------
# context — print (or copy) compiled context to stdout  (A3)
# ---------------------------------------------------------------------------

@app.command()
def context(
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    copy:         bool           = typer.Option(False,    "--copy",    help="Copy to clipboard instead of printing."),
    fmt:          str            = typer.Option("xml",    "--format",  help="Output format: xml | md | txt."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Print compiled session context to stdout (or copy to clipboard with --copy).

    Useful for web AI tools (claude.ai, chatgpt.com, gemini.google.com) that
    cannot read local files — paste the output into the chat window.
    """
    import subprocess

    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)

    from fish_bridge.compiler.active_thread import ActiveThreadCompiler
    compiler  = ActiveThreadCompiler(session_id, cfg.output.token_budget)
    all_nodes = sg.all_nodes()
    all_edges = sg.all_edges()
    sg.close()

    xml_text = compiler.compile(all_nodes, all_edges, turn_count=len(all_nodes))

    if fmt == "md":
        output_text = f"```xml\n{xml_text}\n```"
    elif fmt == "txt":
        # Strip XML tags for a plain-text summary
        import re
        output_text = re.sub(r"<[^>]+>", "", xml_text).strip()
    else:
        output_text = xml_text

    if copy:
        # Cross-platform clipboard copy
        platform = sys.platform
        try:
            if platform == "darwin":
                subprocess.run(["pbcopy"], input=output_text.encode(), check=True)
            elif platform.startswith("linux"):
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=output_text.encode(),
                    check=True,
                )
            else:
                # Windows (includes WSL)
                subprocess.run(
                    ["clip.exe"],
                    input=output_text.encode(),
                    check=True,
                )
            console.print(
                f"[green]✓[/green] Compiled context copied to clipboard "
                f"({len(output_text)} chars, {len(all_nodes)} nodes). "
                "Paste into your AI chat window."
            )
        except FileNotFoundError as exc:
            console.print(f"[yellow]Clipboard tool not found ({exc}). Printing to stdout instead.[/yellow]")
            print(output_text)
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]Clipboard copy failed:[/red] {exc}")
            raise typer.Exit(1)
    else:
        print(output_text)


# ---------------------------------------------------------------------------
# session — sub-commands for multi-session management  (B1)
# ---------------------------------------------------------------------------

session_app = typer.Typer(help="Manage fish_bridge sessions.")
app.add_typer(session_app, name="session")


def _sessions_dir(data_dir: str) -> Path:
    """Return the resolved sessions directory from data_dir config."""
    return Path(data_dir).expanduser()


@session_app.command("list")
def session_list(
    config_file: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
) -> None:
    """List all known sessions."""
    from fish_bridge.config import load_session_metadata

    cfg = load_config(config_file)
    base = _sessions_dir(cfg.data_dir)
    if not base.exists():
        console.print("[yellow]No sessions found.[/yellow]")
        return

    db_files = sorted(base.glob("*.db"))
    if not db_files:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    name_map = load_session_metadata(base)

    from rich.table import Table
    table = Table(title="Sessions", show_header=True, header_style="bold magenta")
    table.add_column("Session ID",     style="bold")
    table.add_column("Title",          style="cyan")
    table.add_column("Nodes",          justify="right")
    table.add_column("Last modified",  style="dim")

    for db_path in db_files:
        sid = db_path.stem
        try:
            sg = SessionGraph.open(sid, base)
            n  = len(sg.all_nodes())
            sg.close()
        except Exception:
            n = "?"
        import datetime
        mtime = datetime.datetime.fromtimestamp(db_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        title = name_map.get(sid, {}).get("title", "")
        table.add_row(sid, title, str(n), mtime)

    console.print(table)


@session_app.command("status")
def session_status(
    project:     Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str] = typer.Option(None,     "--session", help="Session identifier."),
    config_file: Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Show summary statistics for the current (or named) session."""
    from collections import Counter

    session_id = _resolve_session_id(session_name, project)
    sg, _      = _open_session(session_id, config_file)
    nodes      = sg.all_nodes()
    edges      = sg.all_edges()
    sg.close()

    console.print(f"[bold]Session:[/bold] {session_id}")
    console.print(f"  Nodes: [bold]{len(nodes)}[/bold]  |  Edges: [bold]{len(edges)}[/bold]")

    if nodes:
        by_type   = Counter(n.type if isinstance(n.type, str) else n.type.value for n in nodes)
        by_status = Counter(n.status if isinstance(n.status, str) else n.status.value for n in nodes)
        console.print(
            "  By type:   " + "  ".join(f"[dim]{k}[/dim]:[bold]{v}[/bold]" for k, v in sorted(by_type.items()))
        )
        console.print(
            "  By status: " + "  ".join(f"[dim]{k}[/dim]:[bold]{v}[/bold]" for k, v in sorted(by_status.items()))
        )
        topics = {n.metadata.get("topic") for n in nodes if n.metadata.get("topic")}
        if topics:
            console.print(f"  Topics:    {', '.join(sorted(topics))}")


@session_app.command("switch")
def session_switch(
    session_name: str  = typer.Argument(..., help="Session ID to switch to."),
    project:      Path = typer.Option(Path("."), "--project", help="Project root."),
) -> None:
    """Switch the active session for this project."""
    lock_path = project / ".fish_bridge" / "session.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(session_name, encoding="utf-8")

    env_file = project / ".fish_bridge" / ".env"
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        replaced = False
        for line in lines:
            if line.startswith("FISH_BRIDGE_SESSION="):
                new_lines.append(f"FISH_BRIDGE_SESSION={session_name}")
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(f"FISH_BRIDGE_SESSION={session_name}")
        env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    console.print(f"[green]✓[/green] Active session → [bold]{session_name}[/bold]")


@session_app.command("new")
def session_new(
    session_name: str            = typer.Argument(..., help="New session identifier."),
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    config_file:  Optional[Path] = typer.Option(None, "--config", help="Config file path."),
) -> None:
    """Create a new session and switch to it."""
    cfg      = load_config(config_file)
    base     = _sessions_dir(cfg.data_dir)
    base.mkdir(parents=True, exist_ok=True)
    db_path  = base / f"{session_name}.db"
    if db_path.exists():
        console.print(f"[yellow]Session '{session_name}' already exists. Use 'session switch' to activate it.[/yellow]")
        raise typer.Exit(1)

    # Create an empty session
    sg = SessionGraph.open(session_name, base)
    sg.close()

    # Switch the active lock
    lock_path = project / ".fish_bridge" / "session.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(session_name, encoding="utf-8")

    console.print(f"[green]✓[/green] Created and switched to session [bold]{session_name}[/bold]")


@session_app.command("rename")
def session_rename(
    old_name:    str            = typer.Argument(..., help="Current session ID."),
    new_name:    str            = typer.Argument(..., help="New session ID."),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
) -> None:
    """Rename a session (renames the underlying database file)."""
    cfg     = load_config(config_file)
    base    = _sessions_dir(cfg.data_dir)
    old_db  = base / f"{old_name}.db"
    new_db  = base / f"{new_name}.db"

    if not old_db.exists():
        console.print(f"[red]Session '{old_name}' not found.[/red]")
        raise typer.Exit(1)
    if new_db.exists():
        console.print(f"[red]Session '{new_name}' already exists.[/red]")
        raise typer.Exit(1)

    old_db.rename(new_db)
    # Also rename WAL / shm sidecar files if present
    for suffix in ("-wal", "-shm"):
        sidecar = base / f"{old_name}.db{suffix}"
        if sidecar.exists():
            sidecar.rename(base / f"{new_name}.db{suffix}")
    # Migrate display-name metadata entry
    from fish_bridge.config import rename_session_metadata
    rename_session_metadata(base, old_name, new_name)

    console.print(f"[green]✓[/green] Session [bold]{old_name}[/bold] → [bold]{new_name}[/bold]")


@session_app.command("title")
def session_title(
    session_name: str            = typer.Argument(..., help="Session ID to label."),
    title:        Optional[str]  = typer.Argument(None, help="New display title. Omit to read current title."),
    config_file:  Optional[Path] = typer.Option(None, "--config", help="Config file path."),
) -> None:
    """Get or set the human-readable display title for a session.

    The title is shown in the viewer session dropdown instead of the raw
    session ID.  It is stored in a sidecar metadata.json file alongside the
    session databases and never modifies the database itself.

    Examples::

        fish-bridge session title optimized-v6 "AI industry analysis and deployment readiness"
        fish-bridge session title optimized-v6
    """
    from fish_bridge.config import load_session_metadata, set_session_title

    cfg  = load_config(config_file)
    base = _sessions_dir(cfg.data_dir)

    if title is None:
        # Read mode
        meta = load_session_metadata(base)
        current = meta.get(session_name, {}).get("title")
        if current:
            console.print(f"[bold]{session_name}[/bold]: {current}")
        else:
            console.print(f"[bold]{session_name}[/bold]: [dim](no title set)[/dim]")
    else:
        # Write mode
        db_path = base / f"{session_name}.db"
        if not db_path.exists():
            console.print(f"[red]Session '{session_name}' not found.[/red]")
            raise typer.Exit(1)
        set_session_title(base, session_name, title)
        console.print(f"[green]✓[/green] [bold]{session_name}[/bold] title → {title!r}")


# ---------------------------------------------------------------------------
# verify — health check for the current project setup  (C5)
# ---------------------------------------------------------------------------

@app.command()
def verify(
    project:      Path           = typer.Option(Path("."), "--project", help="Project root."),
    session_name: Optional[str]  = typer.Option(None,     "--session", help="Session identifier."),
    config_file:  Optional[Path] = typer.Option(None,     "--config",  help="Config file path."),
) -> None:
    """Check fish_bridge setup health for the current project.

    Verifies: session lock, output file freshness, nested link, backend reachability.
    """
    import datetime

    cfg        = load_config(config_file)
    session_id = _resolve_session_id(session_name, project)
    all_ok     = True

    def _ok(msg: str) -> None:
        console.print(f"[green]  ✓[/green] {msg}")

    def _warn(msg: str) -> None:
        nonlocal all_ok
        all_ok = False
        console.print(f"[yellow]  ⚠[/yellow]  {msg}")

    def _fail(msg: str) -> None:
        nonlocal all_ok
        all_ok = False
        console.print(f"[red]  ✗[/red] {msg}")

    console.print(f"[bold]Verifying session:[/bold] {session_id}")

    # 1. Session lock
    lock_path = project / ".fish_bridge" / "session.lock"
    if lock_path.exists():
        _ok(f"Session lock found: {lock_path}")
    else:
        _warn(f"Session lock not found at {lock_path} — run: fish-bridge init")

    # 2. Output file exists and is not stale
    out_file = project / cfg.output.default_file
    if out_file.exists():
        age = datetime.datetime.now() - datetime.datetime.fromtimestamp(out_file.stat().st_mtime)
        if age > datetime.timedelta(hours=4):
            _warn(f"Output file exists but is stale ({int(age.total_seconds()//3600)}h old): {out_file}")
        else:
            _ok(f"Output file fresh ({int(age.total_seconds()//60)}m old): {out_file}")
    else:
        _fail(f"Output file missing: {out_file}  — run: fish-bridge compile")

    # 3. Nested link present (only meaningful for VS Code local mode)
    if cfg.output.delivery_mode == "local":
        shared_path = project / ".github" / "copilot-instructions.md"
        if shared_path.exists():
            content = shared_path.read_text(encoding="utf-8")
            if cfg.output.local_context_file in content:
                _ok("Nested link found in .github/copilot-instructions.md")
            else:
                _warn(
                    "Nested link missing from .github/copilot-instructions.md — "
                    "run: fish-bridge init --tool copilot"
                )
        else:
            _warn(".github/copilot-instructions.md not found — run: fish-bridge init --tool copilot")

    # 4. Backend reachable
    backend = cfg.extraction.backend
    if backend in ("local", "ollama"):
        local_url = cfg.extraction.local.base_url
        try:
            import urllib.request
            urllib.request.urlopen(f"{local_url}/api/tags", timeout=3)  # noqa: S310
            _ok(f"Ollama reachable at {local_url}")
        except Exception as exc:
            _fail(f"Ollama not reachable at {local_url} ({exc}) — is Ollama running?")
    elif backend == "claude":
        if os.environ.get(cfg.extraction.claude.api_key_env):
            _ok(f"Claude API key set ({cfg.extraction.claude.api_key_env})")
        else:
            _fail(f"Claude API key env var '{cfg.extraction.claude.api_key_env}' not set")
    elif backend in ("openai",):
        if os.environ.get(cfg.extraction.openai.api_key_env):
            _ok(f"OpenAI API key set ({cfg.extraction.openai.api_key_env})")
        else:
            _fail(f"OpenAI API key env var '{cfg.extraction.openai.api_key_env}' not set")
    elif backend == "gemini":
        if os.environ.get(cfg.extraction.gemini.api_key_env):
            _ok(f"Gemini API key set ({cfg.extraction.gemini.api_key_env})")
        else:
            _fail(f"Gemini API key env var '{cfg.extraction.gemini.api_key_env}' not set")
    elif backend == "hybrid":
        # Hybrid uses two backends — check both
        rt  = cfg.extraction.hybrid.realtime_backend
        con = cfg.extraction.hybrid.consolidation_backend
        if rt in ("local", "ollama"):
            local_url = cfg.extraction.local.base_url
            try:
                import urllib.request
                urllib.request.urlopen(f"{local_url}/api/tags", timeout=3)  # noqa: S310
                _ok(f"Hybrid realtime (Ollama) reachable at {local_url}")
            except Exception as exc:
                _fail(f"Hybrid realtime Ollama not reachable at {local_url} ({exc})")
        elif rt == "claude":
            if os.environ.get(cfg.extraction.claude.api_key_env):
                _ok(f"Hybrid realtime Claude key set ({cfg.extraction.claude.api_key_env})")
            else:
                _fail(f"Hybrid realtime Claude key '{cfg.extraction.claude.api_key_env}' not set")
        if con == "gemini":
            if os.environ.get(cfg.extraction.gemini.api_key_env):
                _ok(f"Hybrid consolidation Gemini key set ({cfg.extraction.gemini.api_key_env})")
            else:
                _fail(f"Hybrid consolidation Gemini key '{cfg.extraction.gemini.api_key_env}' not set")
        elif con == "openai":
            if os.environ.get(cfg.extraction.openai.api_key_env):
                _ok(f"Hybrid consolidation OpenAI key set ({cfg.extraction.openai.api_key_env})")
            else:
                _fail(f"Hybrid consolidation OpenAI key '{cfg.extraction.openai.api_key_env}' not set")

    # 5. Session turn count
    try:
        sg, _ = _open_session(session_id, config_file)
        turns = len(sg.all_nodes())
        sg.close()
        if turns == 0:
            _warn("No nodes in session yet — run: fish-bridge ingest")
        else:
            _ok(f"Session has {turns} node(s)")
    except Exception as exc:
        _warn(f"Could not open session: {exc}")

    console.print()
    if all_ok:
        console.print("[green][bold]✓ All checks passed.[/bold][/green]")
    else:
        console.print("[yellow][bold]Some checks failed — see warnings above.[/bold][/yellow]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# serve-mcp — launch MCP server
# ---------------------------------------------------------------------------

@app.command(name="serve-mcp")
def serve_mcp(
    project: Path = typer.Option(Path("."), "--project", help="Project root."),
) -> None:
    """Launch the fish-bridge MCP server (for AI agent real-time capture)."""
    os.environ.setdefault("FISH_BRIDGE_PROJECT", str(project.resolve()))
    from fish_bridge.server import main as _mcp_main
    _mcp_main()


@app.command("diff")
def diff_cmd(
    file_a: Path = typer.Argument(..., help="First .chatgraph.json export (baseline)."),
    file_b: Path = typer.Argument(..., help="Second .chatgraph.json export (compared)."),
) -> None:
    """Compare two session exports and show what changed.

    Reports: nodes added, nodes removed, and status changes between A and B.
    """
    import json as _json

    def _load(p: Path) -> dict[str, dict]:
        """Load a .chatgraph.json and index nodes by id."""
        try:
            raw = _json.loads(p.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            console.print(f"[red]Error reading {p}:[/red] {exc}")
            raise typer.Exit(1)
        nodes_list = raw.get("nodes", [])
        if not isinstance(nodes_list, list):
            console.print(f"[red]Error:[/red] {p} has no 'nodes' array.")
            raise typer.Exit(1)
        return {n["id"]: n for n in nodes_list if isinstance(n, dict) and "id" in n}

    nodes_a = _load(file_a)
    nodes_b = _load(file_b)

    ids_a = set(nodes_a)
    ids_b = set(nodes_b)

    removed   = ids_a - ids_b
    added     = ids_b - ids_a
    common    = ids_a & ids_b

    changed: list[tuple[str, str, str]] = []  # (label, old_status, new_status)
    for nid in common:
        na = nodes_a[nid]
        nb = nodes_b[nid]
        old_s = na.get("status", "")
        new_s = nb.get("status", "")
        if old_s != new_s:
            label = nb.get("label", nid)
            changed.append((label, old_s, new_s))

    if not removed and not added and not changed:
        console.print("[green]No differences found — sessions are identical.[/green]")
        return

    from rich.table import Table

    if added:
        table = Table(title=f"Added ({len(added)})", box=None, show_header=True)
        table.add_column("Label", style="green")
        table.add_column("Type")
        table.add_column("Status")
        for nid in sorted(added):
            n = nodes_b[nid]
            table.add_row(n.get("label", nid), n.get("type", ""), n.get("status", ""))
        console.print(table)

    if removed:
        table = Table(title=f"Removed ({len(removed)})", box=None, show_header=True)
        table.add_column("Label", style="red")
        table.add_column("Type")
        table.add_column("Status")
        for nid in sorted(removed):
            n = nodes_a[nid]
            table.add_row(n.get("label", nid), n.get("type", ""), n.get("status", ""))
        console.print(table)

    if changed:
        table = Table(title=f"Status Changed ({len(changed)})", box=None, show_header=True)
        table.add_column("Label")
        table.add_column("Old Status", style="yellow")
        table.add_column("New Status", style="cyan")
        for label, old_s, new_s in sorted(changed):
            table.add_row(label, old_s, new_s)
        console.print(table)


# ---------------------------------------------------------------------------
# setup — first-run interactive wizard
# ---------------------------------------------------------------------------

_BACKEND_DESCRIPTIONS = {
    "local":   "local (Ollama — free, private, requires Ollama running)",
    "gemini":  "gemini (Google Gemini API — fast, requires GEMINI_API_KEY)",
    "openai":  "openai (OpenAI GPT — high quality, requires OPENAI_API_KEY)",
    "claude":  "claude (Anthropic Claude — best quality, requires ANTHROPIC_API_KEY)",
}


@app.command()
def setup(
    project: Optional[Path] = typer.Option(None, "--project", help="Project root (defaults to interactive prompt)."),
    yes:     bool           = typer.Option(False, "--yes", "-y", help="Accept all defaults without prompting."),
) -> None:
    """Interactive first-run wizard: initialise, ingest, and compile in one step."""

    console.print("\n[bold cyan]🐟 fish_bridge — First-Run Setup Wizard[/bold cyan]\n")

    # ── Step 1: project root ──────────────────────────────────────────────
    if project is not None:
        project_str = str(project)
    elif yes:
        project_str = "."
    else:
        project_str = typer.prompt(
            "Project root directory",
            default=".",
            prompt_suffix=" > ",
        )
    project_path = Path(project_str).expanduser().resolve()
    if not project_path.exists():
        console.print(f"[red]Error:[/red] Directory does not exist: {project_path}")
        raise typer.Exit(1)
    console.print(f"  [dim]Project: {project_path}[/dim]")

    # ── Step 2: AI tool ───────────────────────────────────────────────────
    if yes:
        tool = "copilot"
    else:
        tool_raw = typer.prompt(
            "AI tool you are using [copilot / claude]",
            default="copilot",
            prompt_suffix=" > ",
        ).strip().lower()
        tool = tool_raw if tool_raw in ("copilot", "claude") else "copilot"
    console.print(f"  [dim]Tool:    {tool}[/dim]")

    # ── Step 3: extraction backend ───────────────────────────────────────
    if yes:
        backend = "local"
    else:
        console.print("\n  Extraction backends:")
        for key, desc in _BACKEND_DESCRIPTIONS.items():
            console.print(f"    [bold]{key}[/bold]  —  {desc}")
        backend_raw = typer.prompt(
            "\nExtraction backend [local / gemini / openai / claude]",
            default="local",
            prompt_suffix=" > ",
        ).strip().lower()
        backend = backend_raw if backend_raw in _BACKEND_DESCRIPTIONS else "local"
    console.print(f"  [dim]Backend: {backend}[/dim]\n")

    # ── Step 4: write backend to config ──────────────────────────────────
    console.print("[bold]Step 1/3[/bold]  Writing backend config…")
    try:
        _write_config_backend(backend)
        console.print(f"  [green]✓[/green] Backend set to [bold]{backend}[/bold]")
    except Exception as exc:
        console.print(f"  [yellow]⚠  Could not write config: {exc}[/yellow]")

    # ── Step 5: init ──────────────────────────────────────────────────────
    console.print("[bold]Step 2/3[/bold]  Initialising session…")

    try:
        init(project=project_path, tool=tool, output=None, session_name=None)
    except SystemExit:
        pass  # init already printed its output

    # ── Step 6: ingest ────────────────────────────────────────────────────
    console.print("[bold]Step 3/3[/bold]  Ingesting chat history…")
    try:
        ingest(
            source=tool if tool == "copilot" else "file",
            file_path=None,
            session_name=None,
            workspace=project_path,
            project=project_path,
            output=None,
            no_compile=True,
            config_file=None,
        )
    except SystemExit:
        pass
    except Exception as exc:
        console.print(f"  [yellow]⚠  Ingest warning: {exc}[/yellow]")

    # ── Step 7: compile ───────────────────────────────────────────────────
    try:
        compile(
            project=project_path,
            output=None,
            session_name=None,
            mode="active",
            query=None,
            config_file=None,
        )
    except SystemExit:
        pass
    except Exception as exc:
        console.print(f"  [yellow]⚠  Compile warning: {exc}[/yellow]")

    console.print(
        f"\n[green bold]✓  Setup complete![/green bold]  "
        f"Run [bold]fish-bridge serve --project {project_path}[/bold] to open the graph viewer."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
