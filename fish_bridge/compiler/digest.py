"""DigestCompiler — Mode C: full session digest with community detection.

Produces a markdown HANDOVER.md document:
  - Community detection (Louvain) clusters related nodes into topic groups
  - Each cluster gets a header and a 2-sentence prose summary
  - Deferred and open items listed per cluster
  - Global stats section at the top

Used for:
    fish-bridge digest --mode full > HANDOVER.md
    fish-bridge digest --mode full --session current

Output is ~1000–2000 tokens for a 50-100 node session.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fish_bridge.compiler.base import AbstractCompiler
from fish_bridge.graph.algorithms import (
    community_detection,
    community_label,
    group_nodes_by_community,
)
from fish_bridge.graph.schema import GraphEdge, GraphNode, NodeStatus, NodeType

# Status sets for bucketing
_ACTIVE_STATUSES = {
    NodeStatus.ACTIVE, NodeStatus.PROPOSED, NodeStatus.PENDING,
    NodeStatus.IN_PROGRESS, NodeStatus.BLOCKED, NodeStatus.CONFLICTED,
}
_RESOLVED_STATUSES = {
    NodeStatus.RESOLVED, NodeStatus.FIXED, NodeStatus.DONE,
    NodeStatus.ADOPTED, NodeStatus.REJECTED, NodeStatus.SUPERSEDED,
}
_DEFERRED_STATUSES = {NodeStatus.DEFERRED}


class DigestCompiler(AbstractCompiler):
    """Mode C — full session digest in markdown format."""

    def __init__(
        self,
        session_id: str,
        max_nodes_per_cluster: int = 8,
        min_cluster_size: int = 2,
    ) -> None:
        super().__init__(session_id)
        self.max_nodes_per_cluster = max_nodes_per_cluster
        self.min_cluster_size = min_cluster_size

    # ------------------------------------------------------------------
    # AbstractCompiler
    # ------------------------------------------------------------------

    def compile(  # type: ignore[override]
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        **kwargs,
    ) -> str:
        return self._build_digest(nodes, edges)

    # ------------------------------------------------------------------
    # Core digest builder
    # ------------------------------------------------------------------

    def _build_digest(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lines: list[str] = []

        # Header
        lines.append(f"## Session Digest — {now}")
        lines.append(f"**Session**: `{self.session_id}`  ")
        lines.append(f"**Generated**: {now}  ")
        lines.append(f"**Total nodes**: {len(nodes)}  **Total edges**: {len(edges)}")
        lines.append("")

        # Global stats
        open_q   = [n for n in nodes if n.type == NodeType.QUESTION  and NodeStatus(n.status) in _ACTIVE_STATUSES]
        open_err = [n for n in nodes if n.type == NodeType.ERROR      and NodeStatus(n.status) not in _RESOLVED_STATUSES]
        open_t   = [n for n in nodes if n.type == NodeType.TASK       and NodeStatus(n.status) in _ACTIVE_STATUSES]
        decisions= [n for n in nodes if n.type == NodeType.DECISION   and NodeStatus(n.status) in {NodeStatus.ADOPTED, NodeStatus.PROPOSED}]
        resolved = [n for n in nodes if NodeStatus(n.status) in _RESOLVED_STATUSES]
        deferred = [n for n in nodes if NodeStatus(n.status) in _DEFERRED_STATUSES]

        lines.append("### Summary")
        if open_q:
            lines.append(f"- **{len(open_q)} open question(s)**")
        if open_err:
            lines.append(f"- **{len(open_err)} open error(s)/bug(s)**")
        if open_t:
            lines.append(f"- **{len(open_t)} pending/in-progress task(s)**")
        if decisions:
            lines.append(f"- {len(decisions)} active decision(s)")
        if resolved:
            lines.append(f"- {len(resolved)} resolved item(s) this session")
        if deferred:
            lines.append(f"- {len(deferred)} deferred item(s)")
        lines.append("")

        # Community clusters
        if len(nodes) >= self.min_cluster_size:
            community_map = community_detection(nodes, edges)
            groups = group_nodes_by_community(nodes, community_map)

            # Sort clusters by size descending
            sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

            lines.append("---")
            lines.append("")

            for cid, cluster_nodes in sorted_groups:
                if len(cluster_nodes) < self.min_cluster_size:
                    continue  # skip singletons
                self._render_cluster(lines, cluster_nodes, edges)

        # Deferred items (global list, regardless of cluster)
        all_deferred = [n for n in nodes if NodeStatus(n.status) in _DEFERRED_STATUSES]
        if all_deferred:
            lines.append("---")
            lines.append("")
            lines.append(f"### Deferred Items ({len(all_deferred)})")
            for n in all_deferred:
                icon = _node_icon(n.type)
                lines.append(f"- [ ] {icon} **{n.label}**"
                              + (f" — {n.summary}" if n.summary else ""))
            lines.append("")

        # Open questions that might not appear in clusters
        if open_q:
            lines.append("---")
            lines.append("")
            lines.append(f"### Open Questions ({len(open_q)})")
            for n in open_q:
                lines.append(f"- ❓ **{n.label}**"
                              + (f": {n.summary}" if n.summary else ""))
            lines.append("")

        return "\n".join(lines)

    def _render_cluster(
        self,
        lines: list[str],
        cluster_nodes: list[GraphNode],
        all_edges: list[GraphEdge],
    ) -> None:
        """Render one community cluster as a markdown section."""
        label   = community_label(cluster_nodes)
        n_count = len(cluster_nodes)
        lines.append(f"### {label} ({n_count} nodes)")

        # Sorted: decisions first, then errors, then questions, then tasks, then rest
        _type_priority = {
            NodeType.DECISION: 0, NodeType.ERROR: 1,
            NodeType.QUESTION: 2, NodeType.TASK: 3,
            NodeType.SKILL: 4, NodeType.CONCEPT: 5, NodeType.FILE: 6,
        }
        sorted_nodes = sorted(
            cluster_nodes,
            key=lambda n: (_type_priority.get(NodeType(n.type), 99), -n.confidence),
        )

        # Emit each node
        for n in sorted_nodes[: self.max_nodes_per_cluster]:
            icon   = _node_icon(n.type)
            status = f"[{n.status}]" if str(n.status) not in ("active", "pending") else ""
            summary_part = f": {n.summary}" if n.summary else ""
            lines.append(f"- {icon} **{n.label}** {status}{summary_part}")

        if len(sorted_nodes) > self.max_nodes_per_cluster:
            remaining = len(sorted_nodes) - self.max_nodes_per_cluster
            lines.append(f"- *(+{remaining} more nodes)*")

        lines.append("")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _node_icon(node_type) -> str:
    _icons = {
        "question": "❓",
        "decision": "✅",
        "concept":  "💡",
        "skill":    "🔧",
        "file":     "📄",
        "error":    "🐛",
        "task":     "📋",
    }
    return _icons.get(str(node_type), "•")
