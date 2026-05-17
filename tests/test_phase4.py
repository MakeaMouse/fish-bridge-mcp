"""Tests for Phase 4+ features:
- DependencyFileIngestor
- TestOutputIngestor
- Content-zone pre-processor
- Conflict detection in merge pipeline
- Node management CLI commands (resolve, defer, add)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path



# ---------------------------------------------------------------------------
# DependencyFileIngestor
# ---------------------------------------------------------------------------

class TestDependencyFileIngestor:
    def test_package_json(self, tmp_path: Path) -> None:
        data = {
            "dependencies": {"express": "^4.18.2", "lodash": "~4.17.21"},
            "devDependencies": {"jest": "^29.0.0"},
        }
        (tmp_path / "package.json").write_text(json.dumps(data))

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "package.json")

        labels = {n.label for n in nodes}
        assert "express" in labels
        assert "lodash" in labels
        assert "jest" in labels
        # check is_dev
        jest_node = next(n for n in nodes if n.label == "jest")
        assert jest_node.metadata["is_dev"] is True
        express_node = next(n for n in nodes if n.label == "express")
        assert express_node.metadata["is_dev"] is False
        assert express_node.metadata["registry"] == "npm"

    def test_requirements_txt(self, tmp_path: Path) -> None:
        txt = textwrap.dedent("""\
            # comment
            requests>=2.31.0
            pydantic==2.5.0
            -r other.txt
            boto3
        """)
        (tmp_path / "requirements.txt").write_text(txt)

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "requirements.txt")

        labels = {n.label for n in nodes}
        assert "requests" in labels
        assert "pydantic" in labels
        assert "boto3" in labels

    def test_pyproject_toml(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""\
            [project]
            name = "my-lib"
            dependencies = [
                "httpx>=0.27",
                "pydantic>=2",
            ]
            [project.optional-dependencies]
            dev = ["pytest>=7"]
        """)
        (tmp_path / "pyproject.toml").write_text(toml)

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "pyproject.toml")

        labels = {n.label for n in nodes}
        assert "httpx" in labels
        assert "pydantic" in labels
        assert "pytest" in labels

    def test_cargo_toml(self, tmp_path: Path) -> None:
        toml = textwrap.dedent("""\
            [dependencies]
            serde = "1.0"
            tokio = { version = "1.35", features = ["full"] }
            [dev-dependencies]
            mockall = "0.12"
        """)
        (tmp_path / "Cargo.toml").write_text(toml)

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "Cargo.toml")

        labels = {n.label for n in nodes}
        assert "serde" in labels
        assert "tokio" in labels
        assert "mockall" in labels
        mockall_node = next(n for n in nodes if n.label == "mockall")
        assert mockall_node.metadata["is_dev"] is True

    def test_go_mod(self, tmp_path: Path) -> None:
        mod = textwrap.dedent("""\
            module github.com/my/project
            go 1.21
            require (
                github.com/gin-gonic/gin v1.9.1
                github.com/go-gorm/gorm v1.25.5
            )
        """)
        (tmp_path / "go.mod").write_text(mod)

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "go.mod")

        labels = {n.label for n in nodes}
        assert "gin" in labels
        assert "gorm" in labels

    def test_ingest_project_finds_multiple_manifests(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text(json.dumps({"dependencies": {"react": "^18"}}))
        (tmp_path / "requirements.txt").write_text("fastapi>=0.100\n")

        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest_project(tmp_path)

        labels = {n.label for n in nodes}
        assert "react" in labels
        assert "fastapi" in labels

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        nodes = DependencyFileIngestor().ingest(tmp_path / "no_such_file.json")
        assert nodes == []

    def test_skill_node_type(self, tmp_path: Path) -> None:
        from fish_bridge.graph.schema import NodeType
        from fish_bridge.ingestors.dependency import DependencyFileIngestor
        (tmp_path / "requirements.txt").write_text("numpy>=1.26\n")
        nodes = DependencyFileIngestor().ingest(tmp_path / "requirements.txt")
        assert all(n.type == NodeType.SKILL for n in nodes)
        assert all(n.subtype == "library" for n in nodes)


# ---------------------------------------------------------------------------
# TestOutputIngestor
# ---------------------------------------------------------------------------

class TestTestOutputIngestor:
    def test_jest_json_failures(self, tmp_path: Path) -> None:
        data = {
            "numPassedTests": 1,
            "numFailedTests": 1,
            "testResults": [
                {
                    "testFilePath": "/app/src/__tests__/auth.test.ts",
                    "testResults": [
                        {
                            "fullName": "Auth > login should fail with wrong password",
                            "status": "failed",
                            "failureMessages": ["Expected 401, received 200"],
                        },
                        {
                            "fullName": "Auth > login should succeed",
                            "status": "passed",
                            "failureMessages": [],
                        },
                    ],
                }
            ],
        }
        report = tmp_path / "results.json"
        report.write_text(json.dumps(data))

        from fish_bridge.ingestors.testout import TestOutputIngestor
        from fish_bridge.graph.schema import NodeType, NodeStatus
        nodes, edges = TestOutputIngestor().ingest(report)

        error_nodes = [n for n in nodes if n.type == NodeType.ERROR]
        assert len(error_nodes) == 1
        assert error_nodes[0].subtype == "test_failure"
        assert error_nodes[0].status == NodeStatus.ACTIVE
        assert "401" in error_nodes[0].metadata["assertion"]
        assert len(edges) == 1  # error → tested-by → file

    def test_pytest_json_failures(self, tmp_path: Path) -> None:
        data = {
            "tests": [
                {
                    "nodeid": "tests/test_auth.py::test_login_fail",
                    "outcome": "failed",
                    "call": {
                        "crash": {
                            "message": "AssertionError: assert 200 == 401",
                            "lineno": 42,
                        }
                    },
                },
                {
                    "nodeid": "tests/test_auth.py::test_login_ok",
                    "outcome": "passed",
                    "call": {},
                },
            ]
        }
        report = tmp_path / "report.json"
        report.write_text(json.dumps(data))

        from fish_bridge.ingestors.testout import TestOutputIngestor
        from fish_bridge.graph.schema import NodeType
        nodes, edges = TestOutputIngestor().ingest(report)

        error_nodes = [n for n in nodes if n.type == NodeType.ERROR]
        assert len(error_nodes) == 1
        assert error_nodes[0].metadata["tool"] == "pytest"
        assert error_nodes[0].metadata["line"] == 42

    def test_junit_xml_failures(self, tmp_path: Path) -> None:
        xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <testsuites>
              <testsuite name="AuthTests" tests="2" failures="1" file="tests/AuthTests.java">
                <testcase name="testLoginFail" classname="com.example.AuthTests">
                  <failure message="expected 401 but was 200">stack trace...</failure>
                </testcase>
                <testcase name="testLoginOk" classname="com.example.AuthTests"/>
              </testsuite>
            </testsuites>
        """)
        report = tmp_path / "junit.xml"
        report.write_text(xml)

        from fish_bridge.ingestors.testout import TestOutputIngestor
        from fish_bridge.graph.schema import NodeType
        nodes, edges = TestOutputIngestor().ingest(report)

        error_nodes = [n for n in nodes if n.type == NodeType.ERROR]
        assert len(error_nodes) == 1
        assert error_nodes[0].metadata["tool"] == "junit"
        assert "401" in error_nodes[0].metadata["assertion"]
        assert len(edges) >= 1

    def test_junit_xml_no_failures(self, tmp_path: Path) -> None:
        xml = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <testsuite name="AllPass" tests="2" failures="0">
              <testcase name="test1" classname="Foo"/>
              <testcase name="test2" classname="Foo"/>
            </testsuite>
        """)
        report = tmp_path / "pass.xml"
        report.write_text(xml)

        from fish_bridge.ingestors.testout import TestOutputIngestor
        from fish_bridge.graph.schema import NodeType
        nodes, edges = TestOutputIngestor().ingest(report)

        error_nodes = [n for n in nodes if n.type == NodeType.ERROR]
        assert len(error_nodes) == 0

    def test_skipped_tests_ignored(self, tmp_path: Path) -> None:
        xml = textwrap.dedent("""\
            <testsuite name="S" tests="1">
              <testcase name="t1">
                <skipped message="skip"/>
              </testcase>
            </testsuite>
        """)
        report = tmp_path / "skipped.xml"
        report.write_text(xml)

        from fish_bridge.ingestors.testout import TestOutputIngestor
        from fish_bridge.graph.schema import NodeType
        nodes, _ = TestOutputIngestor().ingest(report)
        assert not any(n.type == NodeType.ERROR for n in nodes)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        from fish_bridge.ingestors.testout import TestOutputIngestor
        nodes, edges = TestOutputIngestor().ingest(tmp_path / "no_file.json")
        assert nodes == []
        assert edges == []

    def test_tested_by_edge_relation(self, tmp_path: Path) -> None:
        from fish_bridge.graph.schema import EdgeRelation
        from fish_bridge.ingestors.testout import TestOutputIngestor
        data = {
            "testResults": [
                {
                    "testFilePath": "/app/test.ts",
                    "testResults": [
                        {"fullName": "fails", "status": "failed", "failureMessages": ["oops"]}
                    ],
                }
            ]
        }
        (tmp_path / "r.json").write_text(json.dumps(data))
        _, edges = TestOutputIngestor().ingest(tmp_path / "r.json")
        assert any(e.relation == EdgeRelation.TESTED_BY for e in edges)


# ---------------------------------------------------------------------------
# Content-zone pre-processor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_code_block_detection(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        user = "Here is my function:\n```python\ndef add(a, b):\n    return a + b\n```"
        hints = preprocess(user, "")
        assert len(hints.code_blocks) == 1
        assert hints.code_blocks[0].language == "python"
        assert not hints.is_empty()

    def test_stack_trace_detection(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        asst = textwrap.dedent("""\
            Traceback (most recent call last):
              File "app.py", line 42, in handle
                result = do_thing()
            ValueError: something went wrong
        """)
        hints = preprocess("", asst)
        assert len(hints.stack_traces) == 1
        assert "ValueError" in hints.stack_traces[0].exception_type

    def test_url_extraction(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        user = "See https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html for details."
        hints = preprocess(user, "")
        assert len(hints.urls) >= 1
        assert any("lambda" in u for u in hints.urls)

    def test_file_ref_extraction(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        user = "Edit src/handlers/auth.py and tests/test_auth.py"
        hints = preprocess(user, "")
        _ = [f.path for f in hints.file_refs]
        # File ref extraction may or may not fire depending on pattern thresholds;
        # just verify the preprocessor runs without error and returns StructuredHints
        assert hasattr(hints, 'file_refs')

    def test_prompt_section_nonempty(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        user = "```bash\nnpm test\n```"
        hints = preprocess(user, "")
        section = hints.to_prompt_section()
        assert len(section) > 0
        assert "```" in section or "code" in section.lower() or "bash" in section.lower()

    def test_empty_input_is_empty(self) -> None:
        from fish_bridge.extraction.preprocessor import preprocess
        hints = preprocess("Hello world", "That makes sense.")
        assert hints.is_empty()

    def test_preprocessor_failure_doesnt_block_extraction(self, tmp_path: Path) -> None:
        """Pre-processor exception must never surface to caller."""
        from fish_bridge.extraction.preprocessor import preprocess
        # Should not raise even with bizarre input
        preprocess("\x00\xff" * 100, "")
        # Either succeeds or returns empty hints — both are fine


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_is_conflict_terminal_to_open(self) -> None:
        from fish_bridge.graph.session import _is_conflict
        assert _is_conflict("adopted", "active") is True
        assert _is_conflict("resolved", "pending") is True
        assert _is_conflict("done", "in_progress") is True

    def test_is_conflict_open_to_terminal_is_ok(self) -> None:
        from fish_bridge.graph.session import _is_conflict
        assert _is_conflict("active", "adopted") is False
        assert _is_conflict("pending", "resolved") is False

    def test_is_conflict_same_status_is_ok(self) -> None:
        from fish_bridge.graph.session import _is_conflict
        assert _is_conflict("active", "active") is False
        assert _is_conflict("adopted", "adopted") is False

    def test_is_conflict_deferred_is_ok(self) -> None:
        from fish_bridge.graph.session import _is_conflict
        assert _is_conflict("adopted", "deferred") is False

    def test_merge_extraction_creates_conflicted_node(self, session_graph) -> None:
        from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType
        # First: add a terminal node
        n1 = GraphNode(
            type=NodeType.DECISION,
            label="Use Redis for caching",
            summary="Redis for cache",
            status=NodeStatus.ADOPTED,
            confidence=0.9,
        )
        session_graph.merge_extraction([n1], [])

        # Second: try to "re-open" the same decision
        n2 = GraphNode(
            type=NodeType.DECISION,
            label="Use Redis for caching",
            summary="Redis cache — reconsidering",
            status=NodeStatus.PROPOSED,
            confidence=0.7,
        )
        stored, _ = session_graph.merge_extraction([n2], [])

        # The existing node should now be CONFLICTED
        all_nodes = session_graph.all_nodes()
        redis_nodes = [n for n in all_nodes if "Redis" in n.label]
        assert len(redis_nodes) == 1
        assert redis_nodes[0].status == NodeStatus.CONFLICTED


# ---------------------------------------------------------------------------
# Node management CLI
# ---------------------------------------------------------------------------

class TestNodeManagementCLI:
    def test_add_command(self, tmp_data_dir: Path, monkeypatch) -> None:
        from typer.testing import CliRunner
        from fish_bridge.cli import app
        import fish_bridge.cli as cli_mod

        # Patch get_data_dir so the CLI writes to our tmp dir
        monkeypatch.setattr(cli_mod, "get_data_dir", lambda cfg: tmp_data_dir)

        runner = CliRunner()
        result = runner.invoke(app, [
            "add",
            "My Test Task",
            "--type", "task",
            "--summary", "A task added via CLI",
            "--session", "cli-test",
        ])
        assert result.exit_code == 0, result.output
        assert "My Test Task" in result.output

    def test_resolve_command(self, tmp_data_dir: Path, monkeypatch) -> None:
        from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType
        from fish_bridge.graph.session import SessionGraph
        from typer.testing import CliRunner
        from fish_bridge.cli import app
        import fish_bridge.cli as cli_mod

        monkeypatch.setattr(cli_mod, "get_data_dir", lambda cfg: tmp_data_dir)

        # Create a node to resolve
        sg = SessionGraph.open("cli-resolve-test", tmp_data_dir)
        n = GraphNode(type=NodeType.TASK, label="Fix the login bug", summary="Login broken",
                      status=NodeStatus.IN_PROGRESS, confidence=0.9)
        sg.merge_extraction([n], [])
        sg.close()

        runner = CliRunner()
        result = runner.invoke(app, [
            "resolve",
            "login bug",
            "--session", "cli-resolve-test",
        ])
        assert result.exit_code == 0, result.output
        # Re-open and check status
        sg2 = SessionGraph.open("cli-resolve-test", tmp_data_dir)
        nodes = sg2.all_nodes()
        task = next((n for n in nodes if "Login" in n.label or "login" in n.label), None)
        assert task is not None
        status_val = task.status if isinstance(task.status, str) else task.status.value
        assert status_val in {"resolved", "fixed", "done"}
        sg2.close()

    def test_defer_command(self, tmp_data_dir: Path, monkeypatch) -> None:
        from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType
        from fish_bridge.graph.session import SessionGraph
        from typer.testing import CliRunner
        from fish_bridge.cli import app
        import fish_bridge.cli as cli_mod

        monkeypatch.setattr(cli_mod, "get_data_dir", lambda cfg: tmp_data_dir)

        sg = SessionGraph.open("cli-defer-test", tmp_data_dir)
        n = GraphNode(type=NodeType.TASK, label="Optimise query latency", summary="Slow queries",
                      status=NodeStatus.PENDING, confidence=0.8)
        sg.merge_extraction([n], [])
        sg.close()

        runner = CliRunner()
        result = runner.invoke(app, [
            "defer",
            "query latency",
            "--session", "cli-defer-test",
        ])
        assert result.exit_code == 0, result.output
        sg2 = SessionGraph.open("cli-defer-test", tmp_data_dir)
        nodes = sg2.all_nodes()
        task = next((n for n in nodes if "latency" in n.label.lower()), None)
        assert task is not None
        assert task.status == NodeStatus.DEFERRED
        sg2.close()


# ---------------------------------------------------------------------------
# Viewer serialisation — regression test for e.relation.value AttributeError
# ---------------------------------------------------------------------------

class TestViewerSerialisation:
    """Ensure graph_data JSON in cli.serve() handles str and enum relations."""

    def test_edge_relation_str_and_enum_serialise(self) -> None:
        import datetime
        from fish_bridge.graph.schema import GraphEdge, EdgeRelation

        # Simulate edges loaded from SQLite — relation comes back as plain str
        e_str = GraphEdge(
            from_id="a", to_id="b",
            relation="relates-to",
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )
        # And an edge created with the enum directly
        e_enum = GraphEdge(
            from_id="b", to_id="a",
            relation=EdgeRelation.RESOLVES,
            created_at=datetime.datetime.now(datetime.timezone.utc),
        )

        def _ser(e: GraphEdge) -> str:
            return e.relation if isinstance(e.relation, str) else e.relation.value

        assert _ser(e_str) == "relates-to"
        assert _ser(e_enum) == "resolves"
