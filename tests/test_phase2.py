"""Phase 2 tests: document, codebase, obsidian ingestors, session import, hybrid backend."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# DocumentIngestor
# ---------------------------------------------------------------------------

class TestDocumentIngestor:

    def test_markdown_splits_on_headings(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor

        md = textwrap.dedent("""\
            # Overview
            This is the overview section.

            ## Background
            Background information here.

            ## Implementation
            Implementation details here.
        """)
        f = tmp_path / "test.md"
        f.write_text(md, encoding="utf-8")

        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        # 3 headings → 3 turns
        assert len(turns) == 3
        assert turns[0].role_user.endswith("Overview")
        assert "Overview" in turns[0].role_user

    def test_markdown_preamble_included(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor

        md = "Preamble text before any heading.\n\n## Section\nContent."
        f = tmp_path / "doc.md"
        f.write_text(md, encoding="utf-8")

        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        assert len(turns) == 2  # preamble + section

    def test_json_splits_by_top_level_key(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor

        data = {"config": {"key": "value"}, "endpoints": ["/api/v1"]}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data), encoding="utf-8")

        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        assert len(turns) == 2
        labels = {t.role_user.split("section: ")[-1] for t in turns}
        assert "config" in labels
        assert "endpoints" in labels

    def test_plain_text_chunked(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor

        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        f = tmp_path / "test.txt"
        f.write_text(text, encoding="utf-8")

        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        assert len(turns) >= 1

    def test_file_not_found_raises(self):
        from fish_bridge.ingestors.document import DocumentIngestor
        with pytest.raises(FileNotFoundError):
            DocumentIngestor().ingest(file_path=Path("/nonexistent/file.md"), session_id="test")

    def test_raw_text_input(self):
        from fish_bridge.ingestors.document import DocumentIngestor
        turns = DocumentIngestor().ingest(
            text="## Decision\nUse Redis.\n\n## Question\nWhat TTL?",
            session_id="test",
            format="markdown",
        )
        assert len(turns) == 2

    def test_source_field_is_document(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor
        f = tmp_path / "readme.md"
        f.write_text("## Section\nContent.", encoding="utf-8")
        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        assert all(t.source == "document" for t in turns)

    def test_yaml_splits_by_key(self, tmp_path):
        from fish_bridge.ingestors.document import DocumentIngestor
        yaml_content = "project: fish_bridge\nversion: 0.1.0\nauthor: user\n"
        f = tmp_path / "meta.yaml"
        f.write_text(yaml_content, encoding="utf-8")
        turns = DocumentIngestor().ingest(file_path=f, session_id="test")
        assert len(turns) == 3


# ---------------------------------------------------------------------------
# ObsidianIngestor
# ---------------------------------------------------------------------------

class TestObsidianIngestor:

    def _create_vault(self, tmp_path: Path) -> Path:
        vault = tmp_path / "vault"
        vault.mkdir()
        # Note 1: with frontmatter
        (vault / "Redis.md").write_text(
            "---\ntags: [project, backend]\nstatus: active\n---\n"
            "Redis is an in-memory data store.\n[[Caching]] strategy matters.",
            encoding="utf-8",
        )
        # Note 2: no frontmatter
        (vault / "Caching.md").write_text(
            "Caching reduces latency.\n\nUse TTL to expire entries.",
            encoding="utf-8",
        )
        # Note 3: subfolder
        sub = vault / "Projects"
        sub.mkdir()
        (sub / "fish_bridge.md").write_text(
            "---\ntags: [project]\n---\nThe fish_bridge project.",
            encoding="utf-8",
        )
        # System directory — should be skipped
        obs = vault / ".obsidian"
        obs.mkdir()
        (obs / "config.json").write_text("{}", encoding="utf-8")
        return vault

    def test_ingests_all_notes(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test")
        assert len(turns) == 3  # Redis, Caching, Projects/fish_bridge (not .obsidian)

    def test_source_field_is_obsidian(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test")
        assert all(t.source == "obsidian" for t in turns)

    def test_tag_filter_limits_notes(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        # Only "project" tagged notes
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test", tag_filter="project")
        # Redis and fish_bridge have "project" tag, Caching does not
        assert len(turns) == 2

    def test_folder_filter(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test", folder_filter="Projects")
        assert len(turns) == 1

    def test_frontmatter_in_user_message(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test")
        redis_turn = next(t for t in turns if "Redis" in t.role_user)
        assert "status: active" in redis_turn.role_user

    def test_max_notes_cap(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        turns = ObsidianIngestor().ingest(vault_path=vault, session_id="test", max_notes=2)
        assert len(turns) <= 2

    def test_vault_not_found_raises(self):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        with pytest.raises(NotADirectoryError):
            ObsidianIngestor().ingest(vault_path="/no/such/vault", session_id="test")

    def test_wikilink_edge_extraction(self, tmp_path):
        from fish_bridge.ingestors.obsidian import ObsidianIngestor
        vault = self._create_vault(tmp_path)
        # Redis.md contains [[Caching]] link
        pairs = ObsidianIngestor.extract_wikilink_edges(vault, {"Redis", "Caching"})
        assert ("Redis", "Caching") in pairs


# ---------------------------------------------------------------------------
# CodebaseIngestor
# ---------------------------------------------------------------------------

class TestCodebaseIngestor:

    def test_returns_empty_list_without_git(self, tmp_path):
        """Non-git directory returns empty list without crashing."""
        from fish_bridge.ingestors.codebase import CodebaseIngestor
        turns = CodebaseIngestor().ingest(path=tmp_path, session_id="test", include_docs=False)
        assert turns == []

    def test_includes_readme_when_present(self, tmp_path):
        from fish_bridge.ingestors.codebase import CodebaseIngestor
        (tmp_path / "README.md").write_text("## Overview\nThis project does X.", encoding="utf-8")
        turns = CodebaseIngestor().ingest(
            path=tmp_path, session_id="test",
            include_commits=False, include_docs=True,
        )
        assert len(turns) >= 1
        assert any("README" in t.role_user for t in turns)

    def test_git_commits_yield_turns(self, tmp_path):
        """Use the actual fish_bridge repo (has git history)."""
        from fish_bridge.ingestors.codebase import CodebaseIngestor
        import subprocess
        repo = Path(__file__).parent.parent  # fish_bridge repo root
        # Check it's actually a git repo
        r = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=repo,
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip("Not a git repository")
        turns = CodebaseIngestor().ingest(
            path=repo, session_id="test",
            n_commits=5, include_docs=False,
        )
        assert len(turns) > 0
        assert all(t.source == "git_commit" for t in turns)

    def test_no_docs_flag_skips_docs(self, tmp_path):
        from fish_bridge.ingestors.codebase import CodebaseIngestor
        (tmp_path / "README.md").write_text("## Docs\nContent.", encoding="utf-8")
        turns = CodebaseIngestor().ingest(
            path=tmp_path, session_id="test",
            include_commits=False, include_docs=False,
        )
        assert turns == []


# ---------------------------------------------------------------------------
# SessionFileIngestor
# ---------------------------------------------------------------------------

class TestSessionFileIngestor:

    def _make_export(self, tmp_path: Path, n_nodes: int = 3) -> Path:
        from fish_bridge.graph.schema import GraphNode, GraphEdge, SessionGraph, NodeType, EdgeRelation
        nodes = [GraphNode(type=NodeType.CONCEPT, label=f"concept-{i}", summary=f"Summary {i}") for i in range(n_nodes)]
        edges = [GraphEdge(from_id=nodes[0].id, to_id=nodes[1].id, relation=EdgeRelation.RELATES_TO)]
        model = SessionGraph(session_id="source-session", nodes=nodes, edges=edges)
        path = tmp_path / "prior.chatgraph.json"
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def test_load_returns_nodes_and_edges(self, tmp_path):
        from fish_bridge.ingestors.session_file import SessionFileIngestor
        path = self._make_export(tmp_path, n_nodes=3)
        nodes, edges = SessionFileIngestor.load(path)
        assert len(nodes) == 3
        assert len(edges) == 1

    def test_load_stamps_original_session_id(self, tmp_path):
        from fish_bridge.ingestors.session_file import SessionFileIngestor
        path = self._make_export(tmp_path)
        nodes, _ = SessionFileIngestor.load(path)
        for node in nodes:
            assert node.metadata.get("original_session_id") == "source-session"

    def test_load_strips_embeddings(self, tmp_path):
        from fish_bridge.graph.schema import GraphNode, SessionGraph, NodeType
        node = GraphNode(type=NodeType.SKILL, label="test-skill")
        node.embedding = [0.1, 0.2, 0.3]
        model = SessionGraph(session_id="src", nodes=[node], edges=[])
        path = tmp_path / "with_embed.chatgraph.json"
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

        from fish_bridge.ingestors.session_file import SessionFileIngestor
        nodes, _ = SessionFileIngestor.load(path)
        assert nodes[0].embedding is None

    def test_summary_returns_metadata(self, tmp_path):
        from fish_bridge.ingestors.session_file import SessionFileIngestor
        path = self._make_export(tmp_path, n_nodes=5)
        summary = SessionFileIngestor.summary(path)
        assert summary["session_id"] == "source-session"
        assert summary["node_count"] == 5

    def test_import_merges_into_session_graph(self, tmp_path):
        from fish_bridge.ingestors.session_file import SessionFileIngestor
        from fish_bridge.graph.session import SessionGraph

        path = self._make_export(tmp_path, n_nodes=3)
        nodes, edges = SessionFileIngestor.load(path)

        sg = SessionGraph.open("test-import", tmp_path / "data")
        stored_nodes, stored_edges = sg.merge_extraction(nodes, edges)
        assert len(stored_nodes) == 3
        sg.close()


# ---------------------------------------------------------------------------
# HybridBackend
# ---------------------------------------------------------------------------

class TestHybridBackend:

    def _stub_backend(self, label: str):
        from fish_bridge.extraction.base import AbstractExtractionBackend
        from fish_bridge.graph.schema import GraphNode, NodeType

        class StubBackend(AbstractExtractionBackend):
            def __init__(self, name: str):
                self.name = name
                self.call_count = 0
            def _call_llm(self, u, a):
                self.call_count += 1
                return {"nodes": [{"type": "concept", "label": f"{self.name}-node",
                                   "summary": "", "status": "active", "confidence": 0.9}],
                        "edges": []}
        return StubBackend(label)

    def test_uses_realtime_by_default(self):
        from fish_bridge.extraction.hybrid import HybridBackend
        from fish_bridge.graph.schema import RawTurn

        rt = self._stub_backend("realtime")
        cloud = self._stub_backend("cloud")
        backend = HybridBackend(rt, cloud, consolidation_every_n=5)

        turn = RawTurn(session_id="t", turn_number=1, role_user="u", role_assistant="a")
        backend.extract(turn)
        assert rt.call_count == 1
        assert cloud.call_count == 0

    def test_uses_cloud_on_nth_turn(self):
        from fish_bridge.extraction.hybrid import HybridBackend
        from fish_bridge.graph.schema import RawTurn

        rt = self._stub_backend("realtime")
        cloud = self._stub_backend("cloud")
        backend = HybridBackend(rt, cloud, consolidation_every_n=3)

        turn = RawTurn(session_id="t", turn_number=1, role_user="u", role_assistant="a")
        for _ in range(3):
            backend.extract(turn)

        # Turn 3 → cloud (3 % 3 == 0)
        assert cloud.call_count == 1
        assert rt.call_count == 2

    def test_build_backend_hybrid(self):
        from fish_bridge.config import FishBridgeConfig, ExtractionConfig, HybridConfig, build_backend
        from fish_bridge.extraction.hybrid import HybridBackend
        from fish_bridge.extraction.local import OllamaBackend

        cfg = FishBridgeConfig(
            extraction=ExtractionConfig(
                backend="hybrid",
                hybrid=HybridConfig(
                    realtime_backend="local",
                    consolidation_backend="local",  # use local for both so no API key needed
                    consolidation_every_n=5,
                ),
            )
        )
        backend = build_backend(cfg)
        assert isinstance(backend, HybridBackend)
        assert isinstance(backend._realtime, OllamaBackend)
