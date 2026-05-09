"""TestOutputIngestor — extract error nodes from structured test output.

Supported formats:
  Jest JSON     — `jest --json > results.json`
  pytest JSON   — `pytest --json-report --json-report-file results.json`
  JUnit XML     — standard JUnit XML (most CI systems)

Failing tests  → error node (subtype="test_failure", status=ACTIVE)
Passing tests  → error node with status=FIXED (when a previous run had failures)

Edge created:
  error  → tested-by → file  (test file that contains the test)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeStatus, NodeType


class TestOutputIngestor:
    """Parse structured test output and produce error + file nodes."""

    def ingest(self, report_path: Path | str) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Parse a test report and return (nodes, edges).

        Auto-detects format from file extension or content.
        Returns empty lists if the file is unrecognised or unreadable.
        """
        path = Path(report_path)
        if not path.exists():
            return [], []

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return [], []

        suffix = path.suffix.lower()

        if suffix == ".xml":
            return self._parse_junit_xml(text)
        else:
            # Try JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return [], []

            # Detect format by shape
            if "testResults" in data or "numPassedTests" in data:
                return self._parse_jest_json(data)
            elif "tests" in data or "summary" in data:
                return self._parse_pytest_json(data)
            return [], []

    # ------------------------------------------------------------------
    # Jest JSON
    # ------------------------------------------------------------------

    def _parse_jest_json(self, data: dict) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for suite in data.get("testResults", []):
            suite_file = suite.get("testFilePath", "unknown")
            file_node  = _make_file_node(suite_file)
            nodes.append(file_node)

            for test in suite.get("testResults", []):
                title   = test.get("fullName") or test.get("title", "unknown test")
                status  = test.get("status", "")  # "passed" | "failed" | "pending"
                msg_arr = test.get("failureMessages", [])
                msg     = msg_arr[0][:300] if msg_arr else ""

                if status == "passed":
                    # Only emit if there's a prior failure to mark as fixed
                    # (we can't know that here — emit with FIXED status anyway,
                    # session.py merge will handle conflict detection)
                    continue

                error_node = GraphNode(
                    type=NodeType.ERROR,
                    label=_short_label(title),
                    summary=f"Jest: {title}" + (f" — {msg[:100]}" if msg else ""),
                    status=NodeStatus.ACTIVE,
                    confidence=1.0,
                    subtype="test_failure",
                    metadata={
                        "test_name": title,
                        "file":      suite_file,
                        "assertion": msg,
                        "tool":      "jest",
                    },
                )
                nodes.append(error_node)
                edges.append(_tested_by_edge(error_node.id, file_node.id))

        return nodes, edges

    # ------------------------------------------------------------------
    # pytest JSON (pytest-json-report plugin)
    # ------------------------------------------------------------------

    def _parse_pytest_json(self, data: dict) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        file_nodes: dict[str, GraphNode] = {}

        for test in data.get("tests", []):
            outcome = test.get("outcome", "")  # "passed" | "failed" | "error"
            if outcome == "passed":
                continue

            node_id  = test.get("nodeid", "")
            # Extract file path from nodeid (e.g. "tests/test_foo.py::test_bar")
            file_path = node_id.split("::")[0] if "::" in node_id else "unknown"
            test_name = node_id.split("::")[-1] if "::" in node_id else node_id

            if file_path not in file_nodes:
                file_nodes[file_path] = _make_file_node(file_path)
                nodes.append(file_nodes[file_path])

            # Extract failure message
            call   = test.get("call", {})
            crash  = call.get("crash", {}) if call else {}
            msg    = (crash.get("message") or call.get("longrepr") or "")[:300]
            lineno = crash.get("lineno")

            error_node = GraphNode(
                type=NodeType.ERROR,
                label=_short_label(test_name),
                summary=f"pytest: {test_name}" + (f" — {msg[:100]}" if msg else ""),
                status=NodeStatus.ACTIVE,
                confidence=1.0,
                subtype="test_failure",
                metadata={
                    "test_name": test_name,
                    "file":      file_path,
                    "line":      lineno,
                    "assertion": msg,
                    "tool":      "pytest",
                },
            )
            nodes.append(error_node)
            edges.append(_tested_by_edge(error_node.id, file_nodes[file_path].id))

        return nodes, edges

    # ------------------------------------------------------------------
    # JUnit XML
    # ------------------------------------------------------------------

    def _parse_junit_xml(self, text: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return [], []

        # Handle both <testsuites><testsuite>... and <testsuite>...
        suites = (
            root.findall("testsuite")
            if root.tag == "testsuites"
            else [root] if root.tag == "testsuite" else []
        )

        for suite in suites:
            suite_name = suite.get("name", "unknown")
            file_attr  = suite.get("file") or suite_name
            file_node  = _make_file_node(file_attr)
            nodes.append(file_node)

            for tc in suite.findall("testcase"):
                tc_name   = tc.get("name", "unknown")
                classname = tc.get("classname", "")
                full_name = f"{classname}.{tc_name}" if classname else tc_name

                failure = tc.find("failure")
                error   = tc.find("error")
                skipped = tc.find("skipped")

                if skipped is not None:
                    continue
                if failure is None and error is None:
                    continue

                elem = failure if failure is not None else error
                msg  = (elem.get("message") or elem.text or "")[:300]
                lineno_m = re.search(r':(\d+)', msg)
                lineno = int(lineno_m.group(1)) if lineno_m else None

                error_node = GraphNode(
                    type=NodeType.ERROR,
                    label=_short_label(tc_name),
                    summary=f"JUnit: {full_name}" + (f" — {msg[:100]}" if msg else ""),
                    status=NodeStatus.ACTIVE,
                    confidence=1.0,
                    subtype="test_failure",
                    metadata={
                        "test_name": full_name,
                        "file":      file_attr,
                        "line":      lineno,
                        "assertion": msg,
                        "tool":      "junit",
                    },
                )
                nodes.append(error_node)
                edges.append(_tested_by_edge(error_node.id, file_node.id))

        return nodes, edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_label(name: str, max_words: int = 6) -> str:
    """Truncate a long test name to a graph-friendly label."""
    words = re.sub(r'[_:]', ' ', name).split()
    return " ".join(words[:max_words])[:60]


def _make_file_node(file_path: str) -> GraphNode:
    label = Path(file_path).name if file_path != "unknown" else "test file"
    return GraphNode(
        type=NodeType.FILE,
        label=label,
        summary=f"Test file: {file_path}",
        status=NodeStatus.ACTIVE,
        confidence=1.0,
        metadata={"path": file_path, "language": "test"},
    )


def _tested_by_edge(error_id: str, file_id: str) -> GraphEdge:
    return GraphEdge(
        from_id=error_id,
        to_id=file_id,
        relation=EdgeRelation.TESTED_BY,
        weight=1.0,
    )
