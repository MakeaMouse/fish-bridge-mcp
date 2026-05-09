"""ActiveThreadCompiler — Mode A: active nodes only, XML output.

Writes/updates the <fish_bridge> managed block in the target instruction file
(.github/copilot-instructions.md or CLAUDE.md).

Token budget enforcement order (never truncate active questions or open errors):
  1. deferred items
  2. concept/skill nodes with lowest confidence
  3. older adopted decisions
  4. file nodes not referenced by active errors or tasks
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from fish_bridge.compiler.base import _output_file_lock
from fish_bridge.graph.schema import GraphEdge, GraphNode, NodeStatus, NodeType

# Default token budget for Mode A (1 token ≈ 4 chars, roughly)
DEFAULT_TOKEN_BUDGET = 800
CHARS_PER_TOKEN = 4

# Managed block delimiters
_BLOCK_START = "<!-- FISH_BRIDGE_START -->"
_BLOCK_END   = "<!-- FISH_BRIDGE_END -->"


class ActiveThreadCompiler:
    """Compile the active session graph into a compact XML block."""

    def __init__(
        self,
        session_id: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.session_id = session_id
        self.token_budget = token_budget

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        turn_count: int = 0,
        stale_pending_after_turns: int = 30,
    ) -> str:
        """Return the compiled XML string.

        Args:
            nodes: All nodes in the session.
            edges: All edges in the session.
            turn_count: Number of turns ingested so far.
            stale_pending_after_turns: When active_count > 50, pending tasks
                older than this many turns (approximated by position in the
                nodes list) are demoted to deferred in the compiled view.
                Set to 0 to disable. Default: 30.
        """
        # Demote stale pending tasks before selecting active nodes
        nodes = self._apply_stale_demotion(nodes, turn_count, stale_pending_after_turns)

        active = self._select_active_nodes(nodes)
        active = self._apply_budget(active, edges)

        # Per-section caps scaled from token_budget. Weights: q=20% e=25% t=30% d=15% f=7% ctx=3%.
        # Formula: total_items ≈ (budget_chars - 800) / 180  (800 = XML overhead; 180 = avg chars/item)
        _budget_chars = self.token_budget * CHARS_PER_TOKEN
        _total_items  = max(8, (_budget_chars - 800) // 180)
        _cap_q   = max(3, _total_items * 20 // 100)
        _cap_e   = max(3, _total_items * 25 // 100)
        _cap_t   = max(3, _total_items * 30 // 100)
        _cap_d   = max(2, _total_items * 15 // 100)
        _cap_f   = max(2, _total_items *  7 // 100)
        _cap_ctx = max(1, _total_items *  3 // 100)  # always show ≥1 so context is never invisible

        # Cold-start placeholder: no nodes ingested yet
        if not active and turn_count == 0:
            return (
                f'<fish_bridge session="{self.session_id}" status="initializing">\n'
                "  <!-- No turns ingested yet. Run: fish-bridge ingest --source copilot -->\n"
                "  <!-- After ingesting, this block will contain your session context. -->\n"
                "</fish_bridge>"
            )

        lines: list[str] = []
        lines.append(
            f'<fish_bridge session="{self.session_id}" '
            f'turns="{turn_count}" '
            f'updated="{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}">'
        )

        # Group by status / type — apply per-section caps with priority ordering
        open_questions = [n for n in active if n.type == NodeType.QUESTION and NodeStatus(n.status) in {NodeStatus.ACTIVE, NodeStatus.DEFERRED}]
        active_errors  = [n for n in active if n.type == NodeType.ERROR    and NodeStatus(n.status) not in {NodeStatus.FIXED, NodeStatus.RESOLVED}]
        open_tasks     = [n for n in active if n.type == NodeType.TASK     and NodeStatus(n.status) in {NodeStatus.PENDING, NodeStatus.IN_PROGRESS, NodeStatus.BLOCKED}]
        decisions      = [n for n in active if n.type == NodeType.DECISION and NodeStatus(n.status) in {NodeStatus.PROPOSED, NodeStatus.ADOPTED}]
        skills_concepts= [n for n in active if n.type in {NodeType.SKILL, NodeType.CONCEPT} and NodeStatus(n.status) == NodeStatus.ACTIVE]
        files          = [n for n in active if n.type == NodeType.FILE]
        conflicts      = [n for n in active if NodeStatus(n.status) == NodeStatus.CONFLICTED]

        # Sort tasks by urgency: in_progress first, then blocked, then pending;
        # within each group most-recently created first so the AI sees what is
        # actively happening before older backlog items.
        _task_urgency = {"in_progress": 0, "blocked": 1, "pending": 2}
        open_tasks.sort(key=lambda n: (
            _task_urgency.get(n.status, 9),
            -(n.created_at.timestamp() if n.created_at else 0),
        ))

        # Questions: highest-confidence first (surface the most certain open issues)
        open_questions.sort(key=lambda n: -n.confidence)

        # Errors: highest-confidence first
        active_errors.sort(key=lambda n: -n.confidence)

        if open_questions:
            shown = open_questions[:_cap_q]
            overflow = len(open_questions) - len(shown)
            lines.append("  <open_questions>")
            for n in shown:
                lines.append(f'    <q status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</q>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more (use fish-bridge compile --mode digest for full list) -->')
            lines.append("  </open_questions>")

        if active_errors:
            shown = active_errors[:_cap_e]
            overflow = len(active_errors) - len(shown)
            lines.append("  <errors>")
            for n in shown:
                lines.append(f'    <error status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</error>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more -->')
            lines.append("  </errors>")

        if decisions:
            shown = decisions[:_cap_d]
            overflow = len(decisions) - len(shown)
            lines.append("  <decisions>")
            for n in shown:
                lines.append(f'    <d status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</d>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more -->')
            lines.append("  </decisions>")

        if open_tasks:
            shown = open_tasks[:_cap_t]
            overflow = len(open_tasks) - len(shown)
            lines.append("  <tasks>")
            for n in shown:
                lines.append(f'    <t status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</t>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more tasks (prioritised: in_progress > blocked > pending) -->')
            lines.append("  </tasks>")

        if skills_concepts and _cap_ctx > 0:
            shown = skills_concepts[:_cap_ctx]
            overflow = len(skills_concepts) - len(shown)
            lines.append("  <context>")
            for n in shown:
                lines.append(f'    <item type="{n.type}">{self._esc(n.label)}: {self._esc(n.summary)}</item>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more -->')
            lines.append("  </context>")

        if files:
            shown = files[:_cap_f]
            overflow = len(files) - len(shown)
            lines.append("  <files>")
            for n in shown:
                lines.append(f'    <f>{self._esc(n.label)}</f>')
            if overflow > 0:
                lines.append(f'    <!-- +{overflow} more -->')
            lines.append("  </files>")

        if conflicts:
            lines.append("  <conflicts>")
            for n in conflicts:
                lines.append(f'    <conflict>{self._esc(n.label)}: {self._esc(n.summary)}</conflict>')
            lines.append("  </conflicts>")

        # Resolved this session — list labels so AI knows what is done/decided
        resolved = [n for n in nodes if NodeStatus(n.status) in {NodeStatus.RESOLVED, NodeStatus.FIXED, NodeStatus.DONE, NodeStatus.ADOPTED, NodeStatus.REJECTED}]
        if resolved:
            lines.append(f'  <resolved_this_session count="{len(resolved)}">')
            for n in resolved[:10]:  # cap at 10 to avoid blowing budget
                lines.append(f'    <r type="{n.type}" status="{n.status}">{self._esc(n.label)}</r>')
            if len(resolved) > 10:
                lines.append(f'    <!-- +{len(resolved) - 10} more -->')
            lines.append("  </resolved_this_session>")

        lines.append("</fish_bridge>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Write to instruction file
    # ------------------------------------------------------------------

    def write(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_file: Path,
        turn_count: int = 0,
        section_header: str = "## Session Context (fish_bridge managed — do not edit)",
    ) -> None:
        """Write/update the managed block in output_file, with an advisory file lock."""
        xml = self.compile(nodes, edges, turn_count)
        block = f"{_BLOCK_START}\n{section_header}\n\n{xml}\n{_BLOCK_END}"

        with _output_file_lock(output_file):
            if output_file.exists():
                content = output_file.read_text(encoding="utf-8")
                pattern = re.compile(
                    re.escape(_BLOCK_START) + r".*?" + re.escape(_BLOCK_END),
                    re.DOTALL,
                )
                if pattern.search(content):
                    new_content = pattern.sub(block, content)
                else:
                    new_content = content.rstrip() + "\n\n" + block + "\n"
            else:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                new_content = block + "\n"

            output_file.write_text(new_content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Stale pending demotion
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_stale_demotion(
        nodes: list[GraphNode],
        turn_count: int,
        stale_after_turns: int,
    ) -> list[GraphNode]:
        """In-memory-only view transform: demote stale pending tasks to deferred.

        Triggered when active + pending node count exceeds 50 and stale_after_turns > 0.
        The DB is not modified — only the compiled output is affected. The oldest half
        of pending tasks (by created_at) are shown as deferred once the session is large
        enough. The stored status remains 'pending' and is visible via 'fish-bridge show'.
        """
        if stale_after_turns <= 0 or turn_count < stale_after_turns:
            return nodes  # not enough turns yet to consider anything stale

        # Count nodes that would be in the active thread
        active_count = sum(
            1 for n in nodes
            if NodeStatus(n.status) in {
                NodeStatus.ACTIVE, NodeStatus.PROPOSED, NodeStatus.ADOPTED,
                NodeStatus.PENDING, NodeStatus.IN_PROGRESS, NodeStatus.BLOCKED,
                NodeStatus.CONFLICTED, NodeStatus.UNCONFIRMED, NodeStatus.DEFERRED,
            }
        )

        if active_count <= 50:
            return nodes  # graph is small enough; no demotion needed

        # Identify pending tasks — collect them sorted oldest-first
        pending_tasks = [
            n for n in nodes
            if n.type == NodeType.TASK and NodeStatus(n.status) == NodeStatus.PENDING
        ]
        if not pending_tasks:
            return nodes

        # Sort by created_at ascending so oldest appear first
        pending_tasks.sort(key=lambda n: n.created_at)

        # Demote the oldest half to deferred (view-only copy)
        demote_count = max(1, len(pending_tasks) // 2)
        demote_ids = {n.id for n in pending_tasks[:demote_count]}

        result: list[GraphNode] = []
        for n in nodes:
            if n.id in demote_ids:
                # Shallow copy with status changed — original DB object unchanged
                import copy as _copy
                view_node = _copy.copy(n)
                view_node.status = NodeStatus.DEFERRED
                result.append(view_node)
            else:
                result.append(n)

        return result

    # ------------------------------------------------------------------
    # Node selection and budget
    # ------------------------------------------------------------------

    @staticmethod
    def _select_active_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
        """Return nodes that belong in the active thread."""
        active_statuses = {
            NodeStatus.ACTIVE,
            NodeStatus.PROPOSED,
            NodeStatus.ADOPTED,
            NodeStatus.PENDING,
            NodeStatus.IN_PROGRESS,
            NodeStatus.BLOCKED,
            NodeStatus.CONFLICTED,
            NodeStatus.UNCONFIRMED,
            NodeStatus.DEFERRED,    # included but truncated first
        }
        return [n for n in nodes if NodeStatus(n.status) in active_statuses]

    def _apply_budget(
        self, nodes: list[GraphNode], edges: list[GraphEdge]
    ) -> list[GraphNode]:
        """Trim nodes to fit within token budget, respecting priority order."""
        max_chars = self.token_budget * CHARS_PER_TOKEN
        # +65 per node: XML tags (~37) + section-header amortisation (~28)
        # +100 for outer <fish_bridge> wrapper and resolved count line
        current_chars = sum(len(n.label) + len(n.summary) + 65 for n in nodes) + 100

        if current_chars <= max_chars:
            return nodes

        # Build set of node IDs referenced by active errors or tasks
        high_priority_ids: set[str] = set()
        for n in nodes:
            if n.type in {NodeType.ERROR, NodeType.TASK} and NodeStatus(n.status) not in {NodeStatus.FIXED, NodeStatus.DONE, NodeStatus.DEFERRED}:
                high_priority_ids.add(n.id)
                for e in edges:
                    if e.from_id == n.id or e.to_id == n.id:
                        high_priority_ids.add(e.from_id)
                        high_priority_ids.add(e.to_id)

        def priority(n: GraphNode) -> int:
            """Lower = truncated first.
            0 = deferred (first to go)
            1 = low-confidence concept/skill
            2 = mid-confidence concept/skill
            3 = high-confidence concept/skill
            4 = adopted decisions not referenced by active tasks/errors
            5 = file nodes not referenced by active tasks/errors
            10 = protected: active questions, open errors, active tasks, conflicts
            """
            if NodeStatus(n.status) == NodeStatus.DEFERRED:
                return 0
            # Concepts and skills are background context — always trimmable,
            # ordered by confidence so lower-quality ones go first.
            if n.type in {NodeType.CONCEPT, NodeType.SKILL}:
                if n.confidence < 0.5:
                    return 1
                elif n.confidence < 0.7:
                    return 2
                else:
                    return 3
            if n.type == NodeType.DECISION and NodeStatus(n.status) == NodeStatus.ADOPTED:
                return 4
            if n.type == NodeType.FILE and n.id not in high_priority_ids:
                return 5
            return 10  # protected

        # Sort descending: protected (10) first, deferred (0) last.
        # We fill from most-important downward so protected nodes always fit
        # and lower-priority nodes consume whatever budget remains.
        sorted_nodes = sorted(nodes, key=priority, reverse=True)
        result: list[GraphNode] = []
        used = 100  # outer wrapper overhead
        for n in sorted_nodes:
            size = len(n.label) + len(n.summary) + 65
            if used + size > max_chars and priority(n) < 10:
                continue
            result.append(n)
            used += size

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _esc(text: str) -> str:
        """Minimal XML character escaping."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )
