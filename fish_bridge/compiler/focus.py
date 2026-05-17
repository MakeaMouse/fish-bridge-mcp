"""FocusCompiler — Mode B: query-driven subgraph context.

Given a natural language query, finds the most relevant nodes via semantic
search, expands to their graph neighborhood, and produces a compact XML
context block (similar to Mode A but scoped to the query topic).

Used for:
    fish-bridge compile --mode focus --query "CORS headers configuration"
    Or via MCP tool get_focus(query="...")

Output: ~300–800 tokens, XML format (same schema as Mode A).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fish_bridge.compiler.base import AbstractCompiler
from fish_bridge.graph.algorithms import semantic_search_nodes, subgraph_neighborhood
from fish_bridge.graph.schema import GraphEdge, GraphNode


class FocusCompiler(AbstractCompiler):
    """Mode B — query-driven subgraph context in XML format."""

    def __init__(
        self,
        session_id: str,
        max_nodes: int = 15,
        max_hops: int = 2,
        top_k_seeds: int = 5,
    ) -> None:
        super().__init__(session_id)
        self.max_nodes   = max_nodes
        self.max_hops    = max_hops
        self.top_k_seeds = top_k_seeds

    # ------------------------------------------------------------------
    # AbstractCompiler
    # ------------------------------------------------------------------

    def compile(  # type: ignore[override]
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        query: str = "",
        **kwargs,
    ) -> str:
        return self._build_focus(nodes, edges, query)

    # ------------------------------------------------------------------
    # Core focus builder
    # ------------------------------------------------------------------

    def _build_focus(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        query: str,
    ) -> str:
        if not query.strip():
            # No query — fall back to top confidence active nodes
            seed_nodes = sorted(nodes, key=lambda n: -n.confidence)[: self.top_k_seeds]
        else:
            ranked = semantic_search_nodes(query, nodes, top_k=self.top_k_seeds)
            seed_nodes = [n for n, _ in ranked]

        seed_ids = [n.id for n in seed_nodes]

        # Expand to neighborhood
        sub_nodes, sub_edges = subgraph_neighborhood(
            seed_ids, nodes, edges,
            max_hops=self.max_hops,
            max_nodes=self.max_nodes,
        )

        # Always include the seeds themselves
        sub_node_ids = {n.id for n in sub_nodes}
        for n in seed_nodes:
            if n.id not in sub_node_ids:
                sub_nodes.append(n)

        return self._render_xml(sub_nodes, sub_edges, query)

    def _render_xml(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        query: str,
    ) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        lines: list[str] = []
        q_attr = f' query="{_esc(query)}"' if query else ""
        lines.append(
            f'<fish_bridge session="{self.session_id}" mode="focus"'
            f'{q_attr} updated="{now}">'
        )

        # Group by type for readability
        type_groups: dict[str, list[GraphNode]] = {}
        for n in nodes:
            type_groups.setdefault(str(n.type), []).append(n)

        _type_order = ["question", "error", "decision", "task", "skill", "concept", "file"]
        _tag_map = {
            "question": ("open_questions", "q"),
            "error":    ("errors",         "error"),
            "decision": ("decisions",       "d"),
            "task":     ("tasks",           "t"),
            "skill":    ("context",         "item"),
            "concept":  ("context",         "item"),
            "file":     ("files",           "f"),
        }

        # Track which tags we've opened to handle context grouping
        emitted_context = False
        for type_key in _type_order:
            group = type_groups.get(type_key, [])
            if not group:
                continue
            wrapper, tag = _tag_map[type_key]

            if wrapper == "context":
                if not emitted_context:
                    lines.append("  <context>")
                    emitted_context = True
                for n in group:
                    status_attr = f' status="{n.status}"' if str(n.status) != "active" else ""
                    summary_part = f": {_esc(n.summary)}" if n.summary else ""
                    lines.append(
                        f'    <{tag} type="{n.type}"{status_attr}>'
                        f'{_esc(n.label)}{summary_part}</{tag}>'
                    )
            else:
                if emitted_context:
                    lines.append("  </context>")
                    emitted_context = False
                lines.append(f"  <{wrapper}>")
                for n in group:
                    status_attr = f' status="{n.status}"'
                    if tag == "f":
                        lines.append(f"    <{tag}>{_esc(n.label)}</{tag}>")
                    else:
                        summary_part = f": {_esc(n.summary)}" if n.summary else ""
                        lines.append(
                            f"    <{tag}{status_attr}>{_esc(n.label)}{summary_part}</{tag}>"
                        )
                lines.append(f"  </{wrapper}>")

        if emitted_context:
            lines.append("  </context>")

        # Relevant edges (within the subgraph)
        node_ids = {n.id: n.label for n in nodes}
        relevant_edges = [e for e in edges if e.from_id in node_ids and e.to_id in node_ids]
        if relevant_edges:
            lines.append("  <relationships>")
            for e in relevant_edges[:20]:  # cap for token budget
                from_label = node_ids.get(e.from_id, e.from_id)
                to_label   = node_ids.get(e.to_id,   e.to_id)
                lines.append(
                    f'    <rel type="{e.relation}">'
                    f'{_esc(from_label)} → {_esc(to_label)}'
                    f'</rel>'
                )
            lines.append("  </relationships>")

        lines.append("</fish_bridge>")
        return "\n".join(lines)


def _esc(text: str) -> str:
    """Minimal XML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
