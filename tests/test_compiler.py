"""Tests for ActiveThreadCompiler output and token budget enforcement."""
from __future__ import annotations

from pathlib import Path

import pytest

from fish_bridge.compiler.active_thread import (
    ActiveThreadCompiler,
    _BLOCK_START,
    _BLOCK_END,
)
from fish_bridge.graph.schema import GraphEdge, GraphNode, NodeStatus, NodeType


def _make_node(label: str, ntype: NodeType, status: NodeStatus, confidence: float = 0.9) -> GraphNode:
    return GraphNode(type=ntype, label=label, summary=f"Summary of {label}", status=status, confidence=confidence)


class TestActiveThreadCompiler:

    def test_compile_basic_output(self):
        nodes = [
            _make_node("DNC caching strategy", NodeType.QUESTION, NodeStatus.ACTIVE),
            _make_node("Use Redis", NodeType.DECISION, NodeStatus.PROPOSED),
            _make_node("redis", NodeType.SKILL, NodeStatus.ACTIVE),
        ]
        compiler = ActiveThreadCompiler("test-session")
        xml = compiler.compile(nodes, [], turn_count=3)
        assert "<fish_bridge" in xml
        assert "DNC caching strategy" in xml
        assert "Use Redis" in xml
        assert "redis" in xml
        assert "</fish_bridge>" in xml

    def test_resolved_nodes_listed_in_summary(self):
        nodes = [
            _make_node("OldBug", NodeType.ERROR, NodeStatus.FIXED),
            _make_node("ActiveQ", NodeType.QUESTION, NodeStatus.ACTIVE),
        ]
        compiler = ActiveThreadCompiler("s")
        xml = compiler.compile(nodes, [])
        assert "resolved_this_session" in xml
        # Resolved items must appear in the summary block (so AI knows what is done)
        assert "OldBug" in xml
        # But resolved items must NOT appear in the live error/task/decision sections
        assert "<error" not in xml or "OldBug" not in xml.split("<resolved_this_session")[0]

    def test_xml_special_chars_escaped(self):
        node = _make_node("Redis & friends <fast>", NodeType.CONCEPT, NodeStatus.ACTIVE)
        compiler = ActiveThreadCompiler("s")
        xml = compiler.compile([node], [])
        assert "&amp;" in xml
        assert "&lt;" in xml

    def test_write_creates_file_if_not_exists(self, tmp_path: Path):
        out = tmp_path / ".github" / "copilot-instructions.md"
        nodes = [_make_node("Test question", NodeType.QUESTION, NodeStatus.ACTIVE)]
        compiler = ActiveThreadCompiler("s")
        compiler.write(nodes, [], out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert _BLOCK_START in content
        assert _BLOCK_END in content

    def test_write_updates_existing_block(self, tmp_path: Path):
        out = tmp_path / "instructions.md"
        out.write_text("# Project\n\n<!-- some other content -->\n")
        compiler = ActiveThreadCompiler("s")
        compiler.write([_make_node("Q1", NodeType.QUESTION, NodeStatus.ACTIVE)], [], out)
        content = out.read_text(encoding="utf-8")
        assert "# Project" in content
        assert _BLOCK_START in content

        # Second write should replace block, not duplicate it
        compiler.write([_make_node("Q2", NodeType.QUESTION, NodeStatus.ACTIVE)], [], out)
        content2 = out.read_text(encoding="utf-8")
        assert content2.count(_BLOCK_START) == 1
        assert "Q2" in content2

    def test_token_budget_drops_deferred_first(self, tmp_path: Path):
        """With a tiny budget, deferred nodes should be dropped before protected ones."""
        nodes = [
            _make_node("Critical open question", NodeType.QUESTION, NodeStatus.ACTIVE),
            _make_node("Deferred item number one which is very long and takes space", NodeType.TASK, NodeStatus.DEFERRED),
            _make_node("Another deferred thing with lots of extra words here", NodeType.CONCEPT, NodeStatus.DEFERRED),
        ]
        compiler = ActiveThreadCompiler("s", token_budget=50)  # very tight
        result = compiler._apply_budget(nodes, [])
        labels = [n.label for n in result]
        assert "Critical open question" in labels
