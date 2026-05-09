"""SessionGraph — high-level CRUD operations on top of SessionStore.

Each method here corresponds to a logical operation the pipeline or CLI needs.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from fish_bridge.extraction.dedup import EmbeddingProvider, semantic_merge
from fish_bridge.graph.schema import (
    EdgeRelation,
    GraphEdge,
    GraphNode,
    NodeStatus,
    NodeType,
    SessionGraph as SessionGraphModel,
    StatusHistoryEntry,
)
from fish_bridge.graph.store import SessionStore
from fish_bridge.config import DedupConfig

# ---------------------------------------------------------------------------
# Status conflict detection
# ---------------------------------------------------------------------------

# Statuses that represent a "settled" conclusion
_TERMINAL_STATUSES: frozenset[str] = frozenset({
    "adopted", "resolved", "fixed", "done", "rejected",
})
# Statuses that represent an "open" state
_OPEN_STATUSES: frozenset[str] = frozenset({
    "active", "proposed", "pending", "in_progress", "blocked",
})


def _is_conflict(current_status: str, incoming_status: str) -> bool:
    """Return True if the incoming status is a reversal of the current status.

    Reversals:
      terminal → open       (e.g. adopted → proposed, resolved → active)
      terminal → different terminal  (e.g. adopted → rejected)
    Non-reversals (natural progressions):
      open     → terminal   (e.g. proposed → adopted)
      open     → open       (e.g. pending → in_progress)
      anything → same       (no-op)
      anything → deferred   (always allowed)
      anything → conflicted (already conflicted)
      anything → unconfirmed
    """
    if current_status == incoming_status:
        return False
    if incoming_status in {"deferred", "conflicted", "unconfirmed", "superseded"}:
        return False
    if current_status in _TERMINAL_STATUSES and incoming_status in _OPEN_STATUSES:
        return True
    if (current_status in _TERMINAL_STATUSES
            and incoming_status in _TERMINAL_STATUSES
            and current_status != incoming_status):
        return True
    return False


class SessionGraph:
    """Manages one session's graph: add nodes/edges, query, persist."""

    def __init__(
        self,
        session_id: str,
        db_path: Path,
        lock_path: Path,
        embed_provider: EmbeddingProvider | None = None,
        dedup_config: DedupConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self._store = SessionStore(db_path, lock_path)
        self._embed_provider = embed_provider or EmbeddingProvider()
        self._dedup_config = dedup_config

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, session_id: str, data_dir: Path, dedup_config: DedupConfig | None = None) -> "SessionGraph":
        """Open (or create) a session graph in data_dir."""
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path   = data_dir / f"{session_id}.db"
        lock_path = data_dir / "session.lock"
        return cls(session_id, db_path, lock_path, dedup_config=dedup_config)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> GraphNode:
        """Persist a node.  Returns the node (may have been modified)."""
        self._store.upsert_node(self.session_id, node)
        return node

    def update_node(self, node: GraphNode) -> None:
        """Persist updates to an existing node."""
        node.updated_at = datetime.now(timezone.utc)
        self._store.upsert_node(self.session_id, node)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._store.get_node(node_id)

    def all_nodes(self) -> list[GraphNode]:
        return self._store.list_nodes(self.session_id)

    def active_nodes(self) -> list[GraphNode]:
        """Nodes that belong in the active compiled thread."""
        active_statuses = {
            NodeStatus.ACTIVE,
            NodeStatus.PROPOSED,
            NodeStatus.ADOPTED,
            NodeStatus.PENDING,
            NodeStatus.IN_PROGRESS,
            NodeStatus.BLOCKED,
            NodeStatus.CONFLICTED,
            # errors that are still open
            NodeStatus.UNCONFIRMED,
        }
        return [n for n in self.all_nodes() if NodeStatus(n.status) in active_statuses]

    def find_by_label(self, label: str, node_type: NodeType | None = None) -> list[GraphNode]:
        nodes = self.all_nodes()
        label_lower = label.lower()
        hits = [n for n in nodes if n.label.lower() == label_lower]
        if node_type:
            hits = [n for n in hits if n.type == node_type]
        return hits

    def set_status(self, node_id: str, status: NodeStatus, note: str | None = None) -> bool:
        node = self.get_node(node_id)
        if node is None:
            return False
        node.push_status(status, note)
        self.update_node(node)
        return True

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        self._store.upsert_edge(self.session_id, edge)
        return edge

    def all_edges(self) -> list[GraphEdge]:
        return self._store.list_edges(self.session_id)

    def edges_for_node(self, node_id: str) -> list[GraphEdge]:
        return [
            e for e in self.all_edges()
            if e.from_id == node_id or e.to_id == node_id
        ]

    # ------------------------------------------------------------------
    # Bulk merge (from extraction engine output)
    # ------------------------------------------------------------------

    def merge_extraction(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Add extracted nodes/edges, de-duplicating by (type, label) with semantic fallback.

        Deduplication strategy (in priority order):
          1. Exact label match (same as before — free, always runs)
          2. Semantic similarity via EmbeddingProvider:
             >0.88 → merge into existing node
             0.70–0.88 → keep both, add relates-to edge
          3. If no embedding available → fall back to exact-match only

        Returns the final (possibly merged) nodes and edges that were stored.
        Edges whose endpoints don't exist after merge are skipped.
        """
        existing_nodes = self.all_nodes()

        # ----- Phase 1: exact-label dedup --------------------------------
        existing_by_key = {
            (n.type if isinstance(n.type, str) else n.type.value, n.label.lower()): n
            for n in existing_nodes
        }
        id_map:        dict[str, str]  = {}
        exact_merged:  list[GraphNode] = []  # nodes handled by exact match
        remainder:     list[GraphNode] = []  # nodes not handled by exact match

        for node in nodes:
            key = (node.type if isinstance(node.type, str) else node.type.value,
                   node.label.lower())
            if key in existing_by_key:
                existing_node = existing_by_key[key]

                # --- Conflict detection ---
                cur_status = (existing_node.status
                              if isinstance(existing_node.status, str)
                              else existing_node.status.value)
                inc_status = (node.status
                              if isinstance(node.status, str)
                              else node.status.value)

                if _is_conflict(cur_status, inc_status):
                    _resolve = (
                        self._dedup_config.auto_resolve_conflicts
                        if self._dedup_config
                        else "manual"
                    )
                    if _resolve == "auto-accept-incoming":
                        # Accept the incoming status; log the reversal for audit
                        existing_node.push_status(
                            NodeStatus(inc_status),
                            note=f"Auto-resolved (accept-incoming): {cur_status} → {inc_status}",
                        )
                    elif _resolve == "auto-keep-existing":
                        # Discard incoming status; log the attempt for audit
                        existing_node.status_history.append(
                            StatusHistoryEntry(
                                status=NodeStatus(inc_status),
                                note=f"Auto-resolved (keep-existing): incoming {inc_status} discarded, kept {cur_status}",
                            )
                        )
                        existing_node.touch()
                    else:  # "manual" (default)
                        # Mark as conflicted and create contradicts edge for user resolution
                        existing_node.push_status(
                            NodeStatus.CONFLICTED,
                            note=f"Status reversal: {cur_status} → {inc_status}",
                        )
                        # Create a contradicts edge (from existing node to itself, as marker)
                        # stored after node update so IDs are available
                        conflict_edge = GraphEdge(
                            from_id=existing_node.id,
                            to_id=existing_node.id,
                            relation=EdgeRelation.CONTRADICTS,
                            weight=0.5,
                        )
                        edges = list(edges) + [conflict_edge]
                else:
                    # Normal merge: update status if it's a natural progression
                    if inc_status not in {cur_status, ""}:
                        existing_node.push_status(NodeStatus(inc_status))

                if node.summary and node.summary != existing_node.summary:
                    existing_node.summary = node.summary
                existing_node.confidence = max(existing_node.confidence, node.confidence)
                existing_node.touch()
                self.update_node(existing_node)
                id_map[node.id] = existing_node.id
                exact_merged.append(existing_node)
            else:
                remainder.append(node)

        # ----- Phase 2: semantic dedup for remainder ----------------------
        merge_t  = self._dedup_config.merge_threshold  if self._dedup_config else None
        relate_t = self._dedup_config.relate_threshold if self._dedup_config else None
        merge_kwargs: dict = {}
        if merge_t is not None:
            merge_kwargs["merge_threshold"] = merge_t
        if relate_t is not None:
            merge_kwargs["relate_threshold"] = relate_t
        to_add, sem_edges, sem_id_map = semantic_merge(
            remainder, existing_nodes, self._embed_provider, **merge_kwargs
        )
        id_map.update(sem_id_map)

        # Persist new nodes and update merged existing nodes
        stored_nodes: list[GraphNode] = list(exact_merged)
        for node in to_add:
            self.add_node(node)
            existing_by_key[(
                node.type if isinstance(node.type, str) else node.type.value,
                node.label.lower(),
            )] = node
            stored_nodes.append(node)

        # Persist updated existing nodes that were merged by semantic pass
        for incoming_id, canonical_id in sem_id_map.items():
            if canonical_id != incoming_id:
                # This was merged; the existing node may have been mutated
                node = self._store.get_node(canonical_id)
                if node is not None:
                    self.update_node(node)

        # ----- Phase 3: resolve + persist edges ---------------------------
        all_edges = list(edges) + sem_edges
        node_ids  = {n.id for n in self.all_nodes()}
        stored_edges: list[GraphEdge] = []

        for edge in all_edges:
            resolved_from = id_map.get(edge.from_id, edge.from_id)
            resolved_to   = id_map.get(edge.to_id,   edge.to_id)
            if resolved_from not in node_ids or resolved_to not in node_ids:
                continue  # skip phantom edges
            edge.from_id = resolved_from
            edge.to_id   = resolved_to
            self.add_edge(edge)
            stored_edges.append(edge)

        return stored_nodes, stored_edges

    # ------------------------------------------------------------------
    # Fix 2: Cross-session edge inference
    # ------------------------------------------------------------------

    def infer_cross_session_edges(
        self,
        min_shared_words: int = 2,
        max_new_edges: int = 200,
    ) -> int:
        """Infer edges between nodes that have no connection but share semantic tokens.

        This is a purely heuristic, offline pass — no LLM call and no embedding
        model required. It is safe on low-resource machines (no GPU / no API key).

        Strategy (in order of preference):
          1. Label substring overlap: if two nodes share ≥ min_shared_words
             significant words, add a typed edge using type-compatibility rules.
          2. Type-lifecycle rules: tasks without a linked question/error get a
             "leads-to" edge to the first open question or unresolved error.

        Returns the number of new edges created.

        When to call:
          - After `import_json` to connect nodes from different source sessions.
          - After `fish-bridge merge --source session ...` to link merged content.
          - Explicitly via CLI: `fish-bridge compile --infer-edges`.
        """
        nodes = self.all_nodes()
        if len(nodes) < 2:
            return 0

        existing_edges = self.all_edges()
        existing_pairs: set[tuple[str, str]] = {
            (e.from_id, e.to_id) for e in existing_edges
        }
        existing_pairs.update((e.to_id, e.from_id) for e in existing_edges)

        # Build token sets for each node (significant words only)
        _stop = frozenset({
            "the", "a", "an", "is", "in", "on", "at", "to", "for", "of",
            "and", "or", "but", "with", "by", "this", "that", "it", "be",
            "are", "was", "were", "has", "have", "had", "not", "use", "using",
        })

        def _tokens(label: str) -> frozenset[str]:
            return frozenset(
                w for w in re.findall(r"\w+", label.lower()) if w not in _stop and len(w) > 2
            )

        # Type-compatible preferred relations (same map as base.py)
        _type_rels: dict[tuple[str, str], EdgeRelation] = {
            ("task",     "question"):  EdgeRelation.LEADS_TO,
            ("task",     "error"):     EdgeRelation.RESOLVES,
            ("task",     "decision"):  EdgeRelation.IMPLEMENTS,
            ("task",     "file"):      EdgeRelation.REFERENCES,
            ("task",     "skill"):     EdgeRelation.USES,
            ("error",    "file"):      EdgeRelation.CREATED_BY,
            ("error",    "task"):      EdgeRelation.BLOCKS,
            ("decision", "concept"):   EdgeRelation.DOCUMENTS,
            ("decision", "skill"):     EdgeRelation.USES,
            ("question", "concept"):   EdgeRelation.REFERENCES,
            ("question", "decision"):  EdgeRelation.LEADS_TO,
            ("skill",    "concept"):   EdgeRelation.RELATES_TO,
            ("concept",  "skill"):     EdgeRelation.RELATES_TO,
            ("file",     "decision"):  EdgeRelation.IMPLEMENTS,
            ("file",     "skill"):     EdgeRelation.USES,
        }

        token_map = {n.id: _tokens(n.label) for n in nodes}
        type_map  = {
            n.id: (n.type if isinstance(n.type, str) else n.type.value)
            for n in nodes
        }

        new_count = 0

        for i, a in enumerate(nodes):
            if new_count >= max_new_edges:
                break
            toks_a = token_map[a.id]
            if not toks_a:
                continue

            for b in nodes[i + 1:]:
                if new_count >= max_new_edges:
                    break
                if (a.id, b.id) in existing_pairs:
                    continue

                shared = toks_a & token_map[b.id]
                if len(shared) < min_shared_words:
                    continue

                ta, tb = type_map[a.id], type_map[b.id]
                relation = _type_rels.get((ta, tb)) or _type_rels.get((tb, ta)) or EdgeRelation.RELATES_TO

                edge = GraphEdge(
                    from_id=a.id,
                    to_id=b.id,
                    relation=relation,
                    weight=round(len(shared) / max(len(toks_a), len(token_map[b.id]), 1), 3),
                )
                self.add_edge(edge)
                existing_pairs.add((a.id, b.id))
                existing_pairs.add((b.id, a.id))
                new_count += 1

        return new_count

    # ------------------------------------------------------------------
    # Fix 5: Remediate existing low-confidence nodes
    # ------------------------------------------------------------------

    def remediate_low_confidence(
        self,
        threshold: float = 0.50,
    ) -> int:
        """Set status=UNCONFIRMED on stored nodes with confidence < threshold.

        This fixes nodes that were persisted before the confidence filter was
        active (e.g. nodes merged from older sessions or extracted with earlier
        prompt versions).

        Safe to call repeatedly — nodes already UNCONFIRMED or in terminal
        statuses (fixed, done, resolved, adopted, rejected, cancelled) are
        left untouched.

        Returns the number of nodes updated.
        """
        _skip_statuses = frozenset({
            "unconfirmed", "fixed", "done", "resolved",
            "adopted", "rejected", "cancelled", "superseded",
        })

        updated = 0
        for node in self.all_nodes():
            status_val = node.status if isinstance(node.status, str) else node.status.value
            if status_val in _skip_statuses:
                continue
            if node.confidence < threshold:
                node.push_status(
                    NodeStatus.UNCONFIRMED,
                    note=f"Remediated: confidence {node.confidence:.2f} < threshold {threshold:.2f}",
                )
                self.update_node(node)
                updated += 1

        return updated

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def to_model(self) -> SessionGraphModel:
        return SessionGraphModel(
            session_id=self.session_id,
            nodes=self.all_nodes(),
            edges=self.all_edges(),
        )

    def export_json(self, path: Path) -> None:
        model = self.to_model()
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def import_json(cls, path: Path, data_dir: Path) -> "SessionGraph":
        raw = json.loads(path.read_text(encoding="utf-8"))
        model = SessionGraphModel.model_validate(raw)
        sg = cls.open(model.session_id, data_dir)
        for node in model.nodes:
            sg.add_node(node)
        for edge in model.edges:
            sg.add_edge(edge)
        return sg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._store.close()
