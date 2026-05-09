"""DependencyFileIngestor — extract skill nodes from package manifests.

Supported manifests:
  package.json         — npm / Node.js
  pyproject.toml       — Python (PEP 621 + setuptools)
  requirements.txt     — Python (pip)
  Cargo.toml           — Rust
  go.mod               — Go
  pom.xml              — Java (Maven) — basic support
  Gemfile              — Ruby

No LLM extraction required — pure structural parsing.
Each direct (non-dev) dependency → skill node (subtype="library").
Dev dependencies → skill node with metadata.is_dev=True.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType


# Registry mapping by manifest type
_REGISTRY = {
    "package.json":   "npm",
    "pyproject.toml": "pypi",
    "requirements.txt": "pypi",
    "Cargo.toml":     "crates.io",
    "go.mod":         "pkg.go.dev",
    "Gemfile":        "rubygems",
    "pom.xml":        "maven",
}


class DependencyFileIngestor:
    """Read package manifest files and produce skill nodes for each dependency."""

    def ingest(self, manifest_path: Path | str) -> list[GraphNode]:
        """Parse a manifest file and return a list of skill nodes.

        Returns an empty list if the file is unrecognised or unreadable.
        """
        path = Path(manifest_path)
        if not path.exists():
            return []

        name = path.name
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if name == "package.json":
            return list(self._parse_package_json(text))
        elif name == "pyproject.toml":
            return list(self._parse_pyproject_toml(text))
        elif name == "requirements.txt" or name.endswith("requirements.txt"):
            return list(self._parse_requirements_txt(text))
        elif name == "Cargo.toml":
            return list(self._parse_cargo_toml(text))
        elif name == "go.mod":
            return list(self._parse_go_mod(text))
        elif name == "Gemfile":
            return list(self._parse_gemfile(text))
        elif name == "pom.xml":
            return list(self._parse_pom_xml(text))
        return []

    def ingest_project(self, project_path: Path | str) -> list[GraphNode]:
        """Scan a project root and parse all recognised manifest files."""
        root = Path(project_path)
        nodes: list[GraphNode] = []
        manifest_names = list(_REGISTRY.keys())
        for name in manifest_names:
            candidate = root / name
            if candidate.exists():
                nodes.extend(self.ingest(candidate))
        return nodes

    # ------------------------------------------------------------------
    # Per-format parsers
    # ------------------------------------------------------------------

    def _parse_package_json(self, text: str) -> Iterator[GraphNode]:
        import json as _json
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            return

        deps     = data.get("dependencies", {})
        dev_deps = data.get("devDependencies", {})

        for pkg, version in deps.items():
            yield self._make_node(pkg, version, "npm", is_dev=False)

        for pkg, version in dev_deps.items():
            yield self._make_node(pkg, version, "npm", is_dev=True)

    def _parse_pyproject_toml(self, text: str) -> Iterator[GraphNode]:
        # Use stdlib tomllib (Python 3.11+) or fallback to regex
        data: dict = {}
        try:
            import tomllib
            data = tomllib.loads(text)
        except ImportError:
            pass  # fallback below

        if data:
            # PEP 621 [project.dependencies]
            for dep_str in data.get("project", {}).get("dependencies", []):
                pkg, version = _split_pep508(dep_str)
                yield self._make_node(pkg, version, "pypi", is_dev=False)
            # [project.optional-dependencies.*]
            for group, deps in data.get("project", {}).get("optional-dependencies", {}).items():
                for dep_str in deps:
                    pkg, version = _split_pep508(dep_str)
                    yield self._make_node(pkg, version, "pypi", is_dev=(group == "dev"))
            # Poetry [tool.poetry.dependencies]
            for pkg, ver_spec in data.get("tool", {}).get("poetry", {}).get("dependencies", {}).items():
                if pkg == "python":
                    continue
                version = ver_spec if isinstance(ver_spec, str) else ""
                yield self._make_node(pkg, version, "pypi", is_dev=False)
        else:
            # Regex fallback
            for m in re.finditer(r'"([A-Za-z0-9_.-]+)\s*([^"]*)"', text):
                pkg = m.group(1)
                version = m.group(2).strip()
                if re.match(r'^[A-Za-z]', pkg):
                    yield self._make_node(pkg, version, "pypi", is_dev=False)

    def _parse_requirements_txt(self, text: str) -> Iterator[GraphNode]:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # pkg==version or pkg>=version etc.
            m = re.match(r'^([A-Za-z0-9_.-]+)\s*([><=!~^].+)?', line)
            if m:
                yield self._make_node(m.group(1), m.group(2) or "", "pypi", is_dev=False)

    def _parse_cargo_toml(self, text: str) -> Iterator[GraphNode]:
        in_deps = False
        in_dev  = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped in ("[dependencies]", "[workspace.dependencies]"):
                in_deps, in_dev = True, False
            elif stripped == "[dev-dependencies]":
                in_deps, in_dev = True, True
            elif stripped.startswith("["):
                in_deps = False
            elif in_deps:
                m = re.match(r'^(\w[\w-]*)\s*=\s*"([^"]+)"', stripped)
                if m:
                    yield self._make_node(m.group(1), m.group(2), "crates.io", is_dev=in_dev)
                # Also handle {version = "..."} form
                m2 = re.match(r'^(\w[\w-]*)\s*=\s*\{[^}]*version\s*=\s*"([^"]+)"', stripped)
                if m2:
                    yield self._make_node(m2.group(1), m2.group(2), "crates.io", is_dev=in_dev)

    def _parse_go_mod(self, text: str) -> Iterator[GraphNode]:
        for line in text.splitlines():
            # require github.com/foo/bar v1.2.3
            m = re.match(r'^\s*([a-zA-Z0-9./~_-]+)\s+(v[^\s]+)', line)
            if m:
                pkg     = m.group(1)
                version = m.group(2)
                if pkg not in {"go"} and not pkg.startswith("//"):
                    yield self._make_node(pkg.split("/")[-1], version, "pkg.go.dev", is_dev=False,
                                          full_path=pkg)

    def _parse_gemfile(self, text: str) -> Iterator[GraphNode]:
        for line in text.splitlines():
            m = re.match(r"^\s*gem\s+'([^']+)'(?:,\s*'([^']+)')?", line)
            if not m:
                m = re.match(r'^\s*gem\s+"([^"]+)"(?:,\s*"([^"]+)")?', line)
            if m:
                yield self._make_node(m.group(1), m.group(2) or "", "rubygems", is_dev=False)

    def _parse_pom_xml(self, text: str) -> Iterator[GraphNode]:
        # Minimal: extract <artifactId> within <dependency> blocks
        for block in re.finditer(r'<dependency>(.*?)</dependency>', text, re.DOTALL):
            content = block.group(1)
            art_m = re.search(r'<artifactId>([^<]+)</artifactId>', content)
            ver_m = re.search(r'<version>([^<]+)</version>', content)
            scope_m = re.search(r'<scope>([^<]+)</scope>', content)
            if art_m:
                is_dev = (scope_m.group(1) in {"test", "provided"}) if scope_m else False
                yield self._make_node(
                    art_m.group(1),
                    ver_m.group(1) if ver_m else "",
                    "maven",
                    is_dev=is_dev,
                )

    # ------------------------------------------------------------------
    # Node factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_node(
        pkg: str,
        version: str,
        registry: str,
        is_dev: bool,
        full_path: str | None = None,
    ) -> GraphNode:
        label = pkg[:50]
        ver_str = version.strip() if version else ""
        summary = f"{registry} library {label}"
        if ver_str:
            summary += f" {ver_str}"
        if is_dev:
            summary += " (dev dependency)"
        return GraphNode(
            type=NodeType.SKILL,
            label=label,
            summary=summary,
            status=NodeStatus.ACTIVE,
            confidence=1.0,
            subtype="library",
            metadata={
                "version":  ver_str,
                "registry": registry,
                "is_dev":   is_dev,
                **({"full_path": full_path} if full_path else {}),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_pep508(dep_str: str) -> tuple[str, str]:
    """Split a PEP 508 dependency string into (package, version_spec)."""
    m = re.match(r'^([A-Za-z0-9_.-]+)\s*([><=!~^,\s].+)?', dep_str.strip())
    if m:
        return m.group(1), (m.group(2) or "").strip()
    return dep_str.strip(), ""
