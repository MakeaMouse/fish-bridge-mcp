"""IaCIngestor — extract architectural decision nodes from Infrastructure-as-Code files.

Supported formats:
  Terraform (.tf)           — regex HCL parsing (no python-hcl2 dependency required)
  CDK synth output          — CloudFormation JSON/YAML produced by `cdk synth`
  CloudFormation (.yaml/.json) — Resources section
  docker-compose.yml        — services → concept nodes

No LLM extraction — pure structural parsing.
Each IaC resource → decision node (subtype="iac_resource").
Resource dependencies → depends-on edges.

Optional: `pip install "fish-bridge-mcp[iac]"` for python-hcl2 (richer Terraform parsing).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeStatus, NodeType

# ---------------------------------------------------------------------------
# Resource type → node type mapping
# ---------------------------------------------------------------------------

_RESOURCE_TYPE_MAP: dict[str, NodeType] = {
    # Compute
    "aws_lambda_function":        NodeType.DECISION,
    "aws_ecs_service":            NodeType.DECISION,
    "aws_ecs_task_definition":    NodeType.DECISION,
    "aws_instance":               NodeType.DECISION,
    # Storage
    "aws_dynamodb_table":         NodeType.DECISION,
    "aws_s3_bucket":              NodeType.DECISION,
    "aws_rds_cluster":            NodeType.DECISION,
    "aws_elasticache_cluster":    NodeType.DECISION,
    "aws_opensearchservice_domain": NodeType.DECISION,
    # API
    "aws_api_gateway_rest_api":   NodeType.DECISION,
    "aws_api_gateway_v2_api":     NodeType.DECISION,
    # Networking
    "aws_vpc":                    NodeType.DECISION,
    "aws_subnet":                 NodeType.CONCEPT,
    "aws_security_group":         NodeType.CONCEPT,
    "aws_lb":                     NodeType.DECISION,
    # IAM
    "aws_iam_role":               NodeType.CONCEPT,
    "aws_iam_policy":             NodeType.CONCEPT,
    # Config/meta
    "variable":                   NodeType.CONCEPT,
    "locals":                     NodeType.CONCEPT,
    "output":                     NodeType.CONCEPT,
    "data":                       NodeType.CONCEPT,
}

# CloudFormation / CDK resource type prefix → node type
_CFN_TYPE_MAP: dict[str, NodeType] = {
    "AWS::Lambda::Function":          NodeType.DECISION,
    "AWS::DynamoDB::Table":           NodeType.DECISION,
    "AWS::S3::Bucket":                NodeType.DECISION,
    "AWS::ApiGateway::RestApi":       NodeType.DECISION,
    "AWS::ApiGatewayV2::Api":         NodeType.DECISION,
    "AWS::ECS::Service":              NodeType.DECISION,
    "AWS::RDS::DBCluster":            NodeType.DECISION,
    "AWS::ElastiCache::CacheCluster": NodeType.DECISION,
    "AWS::OpenSearchService::Domain": NodeType.DECISION,
    "AWS::IAM::Role":                 NodeType.CONCEPT,
    "AWS::IAM::Policy":               NodeType.CONCEPT,
    "AWS::EC2::VPC":                  NodeType.DECISION,
    "AWS::EC2::SecurityGroup":        NodeType.CONCEPT,
    "AWS::ElasticLoadBalancingV2::LoadBalancer": NodeType.DECISION,
}


class IaCIngestor:
    """Parse IaC files and produce decision/concept nodes per resource."""

    def ingest(self, path: Path | str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse one IaC file and return (nodes, edges).

        Auto-detects format from extension and content.
        Returns empty lists for unrecognised or unreadable files.
        """
        p = Path(path)
        if not p.exists():
            return [], []

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], []

        name = p.name.lower()

        if name.endswith(".tf"):
            return self._parse_terraform(text, str(p))
        elif name in {"docker-compose.yml", "docker-compose.yaml"}:
            return self._parse_docker_compose(text, str(p))
        elif name.endswith((".yaml", ".yml", ".json")):
            return self._parse_cfn_or_cdk(text, str(p))
        return [], []

    def ingest_project(self, project_path: Path | str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Scan project root (non-recursively) for all IaC files."""
        root = Path(project_path)
        all_nodes: list[GraphNode] = []
        all_edges: list[GraphEdge] = []

        patterns = [
            "*.tf",
            "docker-compose.yml",
            "docker-compose.yaml",
            "template.yaml",
            "template.json",
            "cdk.out/*.template.json",
            "cloudformation/*.yaml",
            "cloudformation/*.json",
            "infra/*.yaml",
            "infra/*.tf",
        ]
        seen: set[Path] = set()
        for pat in patterns:
            for f in root.glob(pat):
                if f not in seen:
                    seen.add(f)
                    n, e = self.ingest(f)
                    all_nodes.extend(n)
                    all_edges.extend(e)

        return all_nodes, all_edges

    # ------------------------------------------------------------------
    # Terraform (.tf) — regex-based HCL parsing
    # ------------------------------------------------------------------

    def _parse_terraform(self, text: str, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        node_by_ref: dict[str, GraphNode] = {}  # "aws_lambda_function.my_func" → node

        # Match: resource "aws_lambda_function" "my_func" { ... }
        for m in re.finditer(
            r'resource\s+"([^"]+)"\s+"([^"]+)"\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            text, re.DOTALL
        ):
            rtype = m.group(1)
            rname = m.group(2)
            body  = m.group(3)

            node = _make_iac_node(rtype, rname, "terraform", source)
            nodes.append(node)
            node_by_ref[f"{rtype}.{rname}"] = node

            # Extract depends_on = [aws_lambda_function.other, ...]
            dep_m = re.search(r'depends_on\s*=\s*\[([^\]]+)\]', body)
            if dep_m:
                for dep in re.findall(r'[\w.]+', dep_m.group(1)):
                    # Will resolve to edge after all nodes parsed
                    node.metadata.setdefault("_depends_on", []).append(dep)

        # Resolve depends_on edges
        for node in nodes:
            for dep_ref in node.metadata.pop("_depends_on", []):
                target = node_by_ref.get(dep_ref)
                if target:
                    edges.append(_dep_edge(node.id, target.id))

        return nodes, edges

    # ------------------------------------------------------------------
    # CloudFormation / CDK synth output
    # ------------------------------------------------------------------

    def _parse_cfn_or_cdk(self, text: str, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        data: dict = {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try YAML
            try:
                import yaml as _yaml  # type: ignore[import]
                data = _yaml.safe_load(text) or {}
            except Exception:
                pass

        if not isinstance(data, dict):
            return [], []

        # CloudFormation / CDK synth structure: {"Resources": {"LogicalId": {"Type": "...", ...}}}
        resources = data.get("Resources", {})
        if not resources or not isinstance(resources, dict):
            return [], []

        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        node_by_logical: dict[str, GraphNode] = {}

        for logical_id, res_def in resources.items():
            if not isinstance(res_def, dict):
                continue
            cfn_type = res_def.get("Type", "")
            if not cfn_type:
                continue

            ntype = _CFN_TYPE_MAP.get(cfn_type, NodeType.CONCEPT)
            # Use last segment of type as short name: AWS::Lambda::Function → Lambda Function
            short_type = cfn_type.split("::")[-1] if "::" in cfn_type else cfn_type
            label = f"{short_type} {logical_id}"[:60]

            props = res_def.get("Properties", {})
            region = props.get("Region", "")
            node = GraphNode(
                type=ntype,
                label=label,
                summary=f"{cfn_type} resource: {logical_id}",
                status=NodeStatus.ACTIVE,
                confidence=1.0,
                subtype="iac_resource",
                metadata={
                    "resource_type": cfn_type,
                    "logical_id": logical_id,
                    "source": source,
                    **({"region": region} if region else {}),
                },
            )
            nodes.append(node)
            node_by_logical[logical_id] = node

        # DependsOn edges
        for logical_id, res_def in resources.items():
            if not isinstance(res_def, dict):
                continue
            depends_on = res_def.get("DependsOn", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            source_node = node_by_logical.get(logical_id)
            if not source_node:
                continue
            for dep in depends_on:
                target = node_by_logical.get(dep)
                if target:
                    edges.append(_dep_edge(source_node.id, target.id))

        return nodes, edges

    # ------------------------------------------------------------------
    # docker-compose
    # ------------------------------------------------------------------

    def _parse_docker_compose(self, text: str, source: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        try:
            import yaml as _yaml  # type: ignore[import]
            data = _yaml.safe_load(text) or {}
        except Exception:
            # Fallback: regex service name extraction
            nodes = []
            for m in re.finditer(r'^(\w[\w-]*):\s*$', text, re.MULTILINE):
                name = m.group(1)
                if name not in {"version", "services", "volumes", "networks"}:
                    nodes.append(_make_service_node(name, source))
            return nodes, []

        services = data.get("services", {}) if isinstance(data, dict) else {}
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        node_by_name: dict[str, GraphNode] = {}

        for svc_name, svc_def in services.items():
            node = _make_service_node(svc_name, source)
            if isinstance(svc_def, dict):
                image = svc_def.get("image", "")
                if image:
                    node.metadata["image"] = image
                    node.summary = f"Docker service {svc_name} using image {image}"
            nodes.append(node)
            node_by_name[svc_name] = node

        # depends_on edges
        for svc_name, svc_def in services.items():
            if not isinstance(svc_def, dict):
                continue
            dep_val = svc_def.get("depends_on", [])
            if isinstance(dep_val, dict):
                dep_names = list(dep_val.keys())
            elif isinstance(dep_val, list):
                dep_names = dep_val
            else:
                dep_names = []
            src = node_by_name.get(svc_name)
            for dep in dep_names:
                tgt = node_by_name.get(dep)
                if src and tgt:
                    edges.append(_dep_edge(src.id, tgt.id))

        return nodes, []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_iac_node(resource_type: str, resource_name: str, provider: str, source: str) -> GraphNode:
    ntype = _RESOURCE_TYPE_MAP.get(resource_type, NodeType.CONCEPT)
    short = resource_type.split("_", 1)[-1].replace("_", " ")
    label = f"{short} {resource_name}"[:60]
    return GraphNode(
        type=ntype,
        label=label,
        summary=f"Terraform {resource_type}: {resource_name}",
        status=NodeStatus.ACTIVE,
        confidence=1.0,
        subtype="iac_resource",
        metadata={
            "resource_type": resource_type,
            "resource_name": resource_name,
            "provider":      provider,
            "source":        source,
        },
    )


def _make_service_node(name: str, source: str) -> GraphNode:
    return GraphNode(
        type=NodeType.CONCEPT,
        label=f"{name} service",
        summary=f"Docker Compose service: {name}",
        status=NodeStatus.ACTIVE,
        confidence=1.0,
        subtype="iac_resource",
        metadata={"resource_type": "docker_service", "resource_name": name, "source": source},
    )


def _dep_edge(from_id: str, to_id: str) -> GraphEdge:
    return GraphEdge(from_id=from_id, to_id=to_id, relation=EdgeRelation.DEPENDS_ON, weight=1.0)
