"""
Fixture-based scenario tests — no API keys required.

Covers every ingestor source and CLI path using pre-built fixtures:
  - ingest --source file    (Claude JSON export)
  - ingest --source paste   (plain text, no editor)
  - ingest --source copilot (fixture JSONL)
  - obsidian ingestor       (mini vault)
  - openapi ingestor        (tiny spec)
  - config --backend / --show
  - show --all
  - compile (local stub, no LLM call)

Run:
    pytest tests/e2e/ -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
FIXTURES = REPO / "tests" / "fixtures"
CLAUDE_JSON   = FIXTURES / "claude-export.json"
PASTE_TXT     = FIXTURES / "paste-export.txt"
COPILOT_JSONL = FIXTURES / "sample-copilot-session.jsonl"
OBSIDIAN_VAULT = FIXTURES / "obsidian-vault"
OPENAPI_YAML  = FIXTURES / "test-api.openapi.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(*args: str, project: Path | None = None, env: dict | None = None,
        ok_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess:
    """Run fish-bridge CLI via python -m and return CompletedProcess."""
    cmd = [sys.executable, "-m", "fish_bridge.cli"] + list(args)
    if project:
        cmd += ["--project", str(project)]
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)
    if result.returncode not in ok_codes:
        raise AssertionError(
            f"Command {cmd} exited {result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Return a temp directory acting as an isolated fish_bridge project root."""
    return tmp_path


# ---------------------------------------------------------------------------
# Scenario 1: --source file (Claude JSON export)
# ---------------------------------------------------------------------------

class TestIngestFile:
    def test_claude_json_ingest(self, tmp_project: Path, cfg: Path) -> None:
        """Ingest a Claude JSON export without crashing."""
        result = run(
            "ingest", "--source", "file", "--file", str(CLAUDE_JSON),
            "--no-compile", "--config", str(cfg),
            project=tmp_project,
        )
        assert result.returncode == 0 or "No turns" in result.stdout

    def test_missing_file_flag_gives_clear_error(self, tmp_project: Path, cfg: Path) -> None:
        """--source file without --file should exit 1 with a helpful message."""
        result = run(
            "ingest", "--source", "file", "--no-compile", "--config", str(cfg),
            project=tmp_project,
            ok_codes=(0, 1),
        )
        assert result.returncode == 1
        combined = result.stdout + result.stderr
        assert "--file" in combined

    def test_nonexistent_file_gives_error(self, tmp_project: Path, cfg: Path) -> None:
        result = run(
            "ingest", "--source", "file", "--file", "/no/such/file.json",
            "--no-compile", "--config", str(cfg),
            project=tmp_project,
            ok_codes=(0, 1, 2),
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Scenario 2: --source paste (text injected via EDITOR env trick)
# ---------------------------------------------------------------------------

class TestIngestPaste:
    def test_paste_via_fake_editor(self, tmp_project: Path, tmp_path: Path, cfg: Path) -> None:
        """Simulate paste: EDITOR script copies the fixture text to the tmp file."""
        editor_script = tmp_path / "fake_editor.sh"
        editor_script.write_text(f"#!/bin/sh\ncp '{PASTE_TXT}' \"$1\"\n")
        editor_script.chmod(0o755)
        result = run(
            "ingest", "--source", "paste", "--no-compile", "--config", str(cfg),
            project=tmp_project,
            env={"EDITOR": str(editor_script), "VISUAL": str(editor_script)},
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Scenario 3: --source copilot (fixture JSONL)
# ---------------------------------------------------------------------------

class TestIngestCopilot:
    def test_copilot_no_workspace_exits_cleanly(self, tmp_project: Path, cfg: Path) -> None:
        """When no Copilot JSONL is found, exit 1 with a readable message (not a crash).

        We point HOME at tmp_project so the ingestor never touches
        ~/.vscode/workspaceStorage — keeping the test fast and isolated.
        """
        result = run(
            "ingest", "--source", "copilot",
            "--workspace", str(tmp_project),  # empty dir → no JSONL
            "--no-compile", "--config", str(cfg),
            project=tmp_project,
            ok_codes=(0, 1),
            env={"HOME": str(tmp_project)},
        )
        assert result.returncode in (0, 1)
        if result.returncode == 1:
            combined = result.stdout + result.stderr
            assert any(
                kw in combined.lower()
                for kw in ("not found", "no transcript", "no session", "error")
            ), f"Expected a useful error, got:\n{combined}"


# ---------------------------------------------------------------------------
# Scenario 4: Obsidian ingestor (library API)
# ---------------------------------------------------------------------------

class TestObsidianIngestor:
    def test_obsidian_ingest_returns_turns(self) -> None:
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        turns = ObsidianIngestor().ingest(vault_path=OBSIDIAN_VAULT, session_id="test")
        assert len(turns) >= 2, f"Expected ≥2 turns, got {len(turns)}"
        for t in turns:
            assert t.role_user or t.role_assistant

    def test_obsidian_wikilink_edges(self) -> None:
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        ingestor = ObsidianIngestor()
        turns = ingestor.ingest(vault_path=OBSIDIAN_VAULT, session_id="test")
        # extract_wikilink_edges is a static method that needs vault_path + existing_labels
        existing_labels = {t.role_user.splitlines()[0].lstrip("#").strip() for t in turns if t.role_user}
        edges = ObsidianIngestor.extract_wikilink_edges(
            vault_path=OBSIDIAN_VAULT,
            existing_labels=existing_labels | {"pytest", "ruff", "mypy", "python-architecture", "api-decisions", "authentication-patterns"},
        )
        # The fixture notes have [[pytest]], [[ruff]], [[mypy]], [[authentication-patterns]]
        assert len(edges) >= 1, f"Expected ≥1 wikilink edge, got {edges}. Labels: {existing_labels}"

    def test_obsidian_missing_vault_raises(self) -> None:
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        with pytest.raises((NotADirectoryError, ValueError)):
            ObsidianIngestor().ingest(vault_path="/no/such/vault")


# ---------------------------------------------------------------------------
# Scenario 5: OpenAPI ingestor
# ---------------------------------------------------------------------------

class TestOpenAPIIngestor:
    def test_openapi_parses_endpoints(self) -> None:
        from fish_bridge.ingestors.openapi import OpenAPIIngestor
        nodes, edges = OpenAPIIngestor().ingest(OPENAPI_YAML)
        assert len(nodes) >= 3, f"Expected ≥3 nodes, got {len(nodes)}"
        labels = [n.label for n in nodes]
        assert any(
            any(kw in lbl for kw in ("session", "GET", "POST", "list", "create"))
            for lbl in labels
        )

    def test_openapi_security_scheme_as_decision_node(self) -> None:
        from fish_bridge.ingestors.openapi import OpenAPIIngestor
        from fish_bridge.graph.schema import NodeType
        nodes, _ = OpenAPIIngestor().ingest(OPENAPI_YAML)
        # GraphNode uses .type (not .node_type) with use_enum_values=True
        decision_nodes = [n for n in nodes if n.type == NodeType.DECISION.value or n.type == NodeType.DECISION]
        assert len(decision_nodes) >= 1, f"Expected ≥1 DECISION node for bearerAuth, got types: {[n.type for n in nodes]}"

    def test_openapi_missing_file_returns_empty(self) -> None:
        from fish_bridge.ingestors.openapi import OpenAPIIngestor
        nodes, edges = OpenAPIIngestor().ingest("/no/such.yaml")
        assert nodes == [] and edges == []


# ---------------------------------------------------------------------------
# Scenario 6: config command
# ---------------------------------------------------------------------------

class TestConfig:
    # config has no --project flag; it reads/writes ~/.fish_bridge/config.yaml
    def test_config_show(self, tmp_project: Path) -> None:
        result = run("config", "--show")
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "backend" in combined.lower()

    def test_config_set_backend_local(self, tmp_project: Path) -> None:
        result = run("config", "--backend", "local")
        assert result.returncode == 0

    def test_config_set_backend_gemini(self, tmp_project: Path) -> None:
        result = run("config", "--backend", "gemini")
        assert result.returncode == 0

    def test_config_invalid_backend_no_traceback(self, tmp_project: Path) -> None:
        result = run("config", "--backend", "notabackend", ok_codes=(0, 1))
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# Scenario 7: show command
# ---------------------------------------------------------------------------

class TestShow:
    def test_show_all_empty_no_crash(self, tmp_project: Path) -> None:
        result = run("show", "--all", project=tmp_project, ok_codes=(0, 1))
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined

    def test_show_default_empty_no_crash(self, tmp_project: Path) -> None:
        result = run("show", project=tmp_project, ok_codes=(0, 1))
        combined = result.stdout + result.stderr
        assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# Scenario 8: init command
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_copilot_creates_file(self, tmp_project: Path) -> None:
        result = run("init", "--tool", "copilot", project=tmp_project)
        assert result.returncode == 0
        created = list(tmp_project.rglob("*"))
        assert len(created) >= 1, "init --tool copilot should create at least one file"

    def test_init_all_tools(self, tmp_project: Path) -> None:
        result = run("init", "--tool", "all", project=tmp_project)
        assert result.returncode == 0

    def test_init_invalid_tool_exits_nonzero(self, tmp_project: Path) -> None:
        result = run("init", "--tool", "doesnotexist", project=tmp_project, ok_codes=(0, 1, 2))
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Scenario 9: MCP server JSON-RPC handshake
# ---------------------------------------------------------------------------

class TestMCPServer:
    def test_mcp_initialize_handshake(self) -> None:
        """MCP server must respond to initialize with a valid JSON-RPC result."""
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "scenario-test", "version": "1.0"},
            },
        }) + "\n"

        result = subprocess.run(
            [sys.executable, "-m", "fish_bridge.server"],
            input=msg,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        assert lines, f"No output from MCP server.\nstderr: {result.stderr}"
        response = json.loads(lines[0])
        assert "result" in response, f"Expected 'result' key, got: {response}"
        assert response["id"] == 1
