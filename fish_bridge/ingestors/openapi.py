"""OpenAPIIngestor — extract skill/concept/decision nodes from OpenAPI specs.

Supported formats:
  OpenAPI 3.x (.yaml / .json)
  Swagger 2.0 (.yaml / .json)
  AsyncAPI 2.x (.yaml / .json)  — channels → skill nodes

No LLM extraction — pure structural parsing.
  endpoints   → skill nodes  (subtype="endpoint")
  schemas     → concept nodes
  securitySchemes → decision nodes (subtype="auth_scheme")

Optional: `pip install "fish-bridge-mcp[openapi]"` for validation via openapi-spec-validator.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeStatus, NodeType


class OpenAPIIngestor:
    """Parse OpenAPI 3.x / Swagger 2.0 / AsyncAPI specs into graph nodes."""

    def ingest(self, spec_path: Path | str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse a spec file and return (nodes, edges).

        Returns empty lists for unrecognised or unreadable files.
        """
        p = Path(spec_path)
        if not p.exists():
            return [], []

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], []

        data: dict = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml as _yaml  # type: ignore[import]
                data = _yaml.safe_load(text) or {}
            except Exception:
                pass

        if not isinstance(data, dict):
            return [], []

        # Detect format
        if "openapi" in data:
            return self._parse_openapi3(data, str(p))
        elif "swagger" in data:
            return self._parse_swagger2(data, str(p))
        elif "asyncapi" in data:
            return self._parse_asyncapi(data, str(p))
        return [], []

    # ------------------------------------------------------------------
    # OpenAPI 3.x
    # ------------------------------------------------------------------

    def _parse_openapi3(self, data: dict, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        api_title = data.get("info", {}).get("title", "API")

        # Endpoints — paths → operations
        for path, path_item in (data.get("paths") or {}).items():
            if not isinstance(path_item, dict):
                continue
            for method in ("get", "post", "put", "patch", "delete", "head", "options"):
                op = path_item.get(method)
                if not isinstance(op, dict):
                    continue
                op_id     = op.get("operationId", "")
                summary   = op.get("summary", op.get("description", ""))[:120]
                tags      = op.get("tags", [])
                label     = op_id or f"{method.upper()} {path}"

                status_codes = list((op.get("responses") or {}).keys())

                node = GraphNode(
                    type=NodeType.SKILL,
                    label=label[:60],
                    summary=summary or f"{method.upper()} {path}",
                    status=NodeStatus.ACTIVE,
                    confidence=1.0,
                    subtype="endpoint",
                    metadata={
                        "method":        method.upper(),
                        "path":          path,
                        "operation_id":  op_id,
                        "tags":          tags,
                        "status_codes":  status_codes,
                        "source":        source,
                    },
                )
                nodes.append(node)

                # Link response schema refs
                for status_code, resp_def in (op.get("responses") or {}).items():
                    if isinstance(resp_def, dict):
                        schema_ref = _extract_schema_ref(resp_def)
                        if schema_ref:
                            node.metadata.setdefault("response_schemas", []).append(schema_ref)

        # Security schemes → decision nodes
        for scheme_name, scheme_def in (
            (data.get("components") or {}).get("securitySchemes") or {}
        ).items():
            if not isinstance(scheme_def, dict):
                continue
            auth_type = scheme_def.get("type", "")
            scheme_node = GraphNode(
                type=NodeType.DECISION,
                label=f"{scheme_name} auth",
                summary=f"Security scheme: {scheme_name} ({auth_type})",
                status=NodeStatus.ACTIVE,
                confidence=1.0,
                subtype="auth_scheme",
                metadata={
                    "scheme_name": scheme_name,
                    "type":        auth_type,
                    "flows":       list((scheme_def.get("flows") or {}).keys()),
                    "source":      source,
                },
            )
            nodes.append(scheme_node)

        # Component schemas → concept nodes
        for schema_name, schema_def in (
            (data.get("components") or {}).get("schemas") or {}
        ).items():
            if not isinstance(schema_def, dict):
                continue
            properties = list((schema_def.get("properties") or {}).keys())[:10]
            concept_node = GraphNode(
                type=NodeType.CONCEPT,
                label=schema_name[:60],
                summary=f"Schema: {schema_name}" + (f" — {', '.join(properties[:5])}" if properties else ""),
                status=NodeStatus.ACTIVE,
                confidence=1.0,
                subtype="schema",
                metadata={
                    "schema_name": schema_name,
                    "schema_keys": properties,
                    "source":      source,
                },
            )
            nodes.append(concept_node)

        return nodes, edges

    # ------------------------------------------------------------------
    # Swagger 2.0
    # ------------------------------------------------------------------

    def _parse_swagger2(self, data: dict, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        base_path = data.get("basePath", "")

        for path, path_item in (data.get("paths") or {}).items():
            if not isinstance(path_item, dict):
                continue
            for method in ("get", "post", "put", "patch", "delete"):
                op = path_item.get(method)
                if not isinstance(op, dict):
                    continue
                op_id   = op.get("operationId", "")
                summary = op.get("summary", op.get("description", ""))[:120]
                label   = op_id or f"{method.upper()} {base_path}{path}"

                node = GraphNode(
                    type=NodeType.SKILL,
                    label=label[:60],
                    summary=summary or f"{method.upper()} {path}",
                    status=NodeStatus.ACTIVE,
                    confidence=1.0,
                    subtype="endpoint",
                    metadata={
                        "method":       method.upper(),
                        "path":         f"{base_path}{path}",
                        "operation_id": op_id,
                        "consumes":     op.get("consumes", []),
                        "produces":     op.get("produces", []),
                        "source":       source,
                    },
                )
                nodes.append(node)

        # securityDefinitions
        for scheme_name, scheme_def in (data.get("securityDefinitions") or {}).items():
            if not isinstance(scheme_def, dict):
                continue
            nodes.append(GraphNode(
                type=NodeType.DECISION,
                label=f"{scheme_name} auth",
                summary=f"Security definition: {scheme_name} ({scheme_def.get('type', '')})",
                status=NodeStatus.ACTIVE,
                confidence=1.0,
                subtype="auth_scheme",
                metadata={"scheme_name": scheme_name, "type": scheme_def.get("type", ""), "source": source},
            ))

        return nodes, edges

    # ------------------------------------------------------------------
    # AsyncAPI 2.x
    # ------------------------------------------------------------------

    def _parse_asyncapi(self, data: dict, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []

        for channel, channel_def in (data.get("channels") or {}).items():
            if not isinstance(channel_def, dict):
                continue
            ops = []
            if "publish" in channel_def:
                ops.append(("publish", channel_def["publish"]))
            if "subscribe" in channel_def:
                ops.append(("subscribe", channel_def["subscribe"]))

            for op_type, op_def in ops:
                op_id   = op_def.get("operationId", "") if isinstance(op_def, dict) else ""
                summary = op_def.get("summary", "") if isinstance(op_def, dict) else ""
                label   = op_id or f"{op_type} {channel}"
                nodes.append(GraphNode(
                    type=NodeType.SKILL,
                    label=label[:60],
                    summary=summary or f"AsyncAPI {op_type} on {channel}",
                    status=NodeStatus.ACTIVE,
                    confidence=1.0,
                    subtype="endpoint",
                    metadata={"channel": channel, "operation": op_type, "source": source},
                ))

        return nodes, []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_schema_ref(resp_def: dict) -> str | None:
    """Extract a $ref schema name from a response definition."""
    content = resp_def.get("content", {})
    for media_type, media_def in content.items():
        if isinstance(media_def, dict):
            schema = media_def.get("schema", {})
            if isinstance(schema, dict):
                ref = schema.get("$ref", "")
                if ref:
                    return ref.split("/")[-1]
    return None
