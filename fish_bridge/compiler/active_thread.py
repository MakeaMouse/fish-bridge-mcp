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
    ) -> str:
        """Return the compiled XML string."""
        active = self._select_active_nodes(nodes)
        active = self._apply_budget(active, edges)

        lines: list[str] = []
        lines.append(
            f'<fish_bridge session="{self.session_id}" '
            f'turns="{turn_count}" '
            f'updated="{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")}">'
        )

        # Group by status / type
        open_questions = [n for n in active if n.type == NodeType.QUESTION and NodeStatus(n.status) in {NodeStatus.ACTIVE, NodeStatus.DEFERRED}]
        active_errors  = [n for n in active if n.type == NodeType.ERROR    and NodeStatus(n.status) not in {NodeStatus.FIXED, NodeStatus.RESOLVED}]
        open_tasks     = [n for n in active if n.type == NodeType.TASK     and NodeStatus(n.status) in {NodeStatus.PENDING, NodeStatus.IN_PROGRESS, NodeStatus.BLOCKED}]
        decisions      = [n for n in active if n.type == NodeType.DECISION and NodeStatus(n.status) in {NodeStatus.PROPOSED, NodeStatus.ADOPTED}]
        skills_concepts= [n for n in active if n.type in {NodeType.SKILL, NodeType.CONCEPT} and NodeStatus(n.status) == NodeStatus.ACTIVE]
        files          = [n for n in active if n.type == NodeType.FILE]
        conflicts      = [n for n in active if NodeStatus(n.status) == NodeStatus.CONFLICTED]

        if open_questions:
            lines.append("  <open_questions>")
            for n in open_questions:
                lines.append(f'    <q status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</q>')
            lines.append("  </open_questions>")

        if active_errors:
            lines.append("  <errors>")
            for n in active_errors:
                lines.append(f'    <error status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</error>')
            lines.append("  </errors>")

        if decisions:
            lines.append("  <decisions>")
            for n in decisions:
                lines.append(f'    <d status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</d>')
            lines.append("  </decisions>")

        if open_tasks:
            lines.append("  <tasks>")
            for n in open_tasks:
                lines.append(f'    <t status="{n.status}">{self._esc(n.label)}: {self._esc(n.summary)}</t>')
            lines.append("  </tasks>")

        if skills_concepts:
            lines.append("  <context>")
            for n in skills_concepts:
                lines.append(f'    <item type="{n.type}">{self._esc(n.label)}: {self._esc(n.summary)}</item>')
            lines.append("  </context>")

        if files:
            lines.append("  <files>")
            for n in files:
                lines.append(f'    <f>{self._esc(n.label)}</f>')
            lines.append("  </files>")

        if conflicts:
            lines.append("  <conflicts>")
            for n in conflicts:
                lines.append(f'    <conflict>{self._esc(n.label)}: {self._esc(n.summary)}</conflict>')
            lines.append("  </conflicts>")

        # Resolved this session (count only, no details)
        resolved = [n for n in nodes if NodeStatus(n.status) in {NodeStatus.RESOLVED, NodeStatus.FIXED, NodeStatus.DONE, NodeStatus.ADOPTED, NodeStatus.REJECTED}]
        if resolved:
            lines.append(f'  <resolved_this_session count="{len(resolved)}"/>')

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
        """Write/update the managed block in output_file."""
        xml = self.compile(nodes, edges, turn_count)
        block = f"{_BLOCK_START}\n{section_header}\n\n{xml}\n{_BLOCK_END}"

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
        current_chars = sum(len(n.label) + len(n.summary) + 40 for n in nodes)

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

        # Never-truncate: active questions and open errors
        protected_statuses = {NodeStatus.ACTIVE, NodeStatus.CONFLICTED}
        protected_types    = {NodeType.QUESTION, NodeType.ERROR}

        def priority(n: GraphNode) -> int:
            """Lower = truncated first."""
            if NodeStatus(n.status) == NodeStatus.DEFERRED:
                return 0
            if n.type in {NodeType.CONCEPT, NodeType.SKILL} and n.confidence < 0.7:
                return 1
            if n.type == NodeType.DECISION and NodeStatus(n.status) == NodeStatus.ADOPTED:
                return 2
            if n.type == NodeType.FILE and n.id not in high_priority_ids:
                return 3
            return 10  # protected

        sorted_nodes = sorted(nodes, key=priority)
        result: list[GraphNode] = []
        used = 0
        for n in sorted_nodes:
            size = len(n.label) + len(n.summary) + 40
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
