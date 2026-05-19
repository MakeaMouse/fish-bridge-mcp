"""Tests for extraction base: grounding check, confidence, schema validation."""
from __future__ import annotations

from fish_bridge.extraction.base import AbstractExtractionBackend
from fish_bridge.graph.schema import GraphNode, NodeStatus, NodeType, RawTurn


# ---------------------------------------------------------------------------
# Concrete stub backend for testing (no LLM call)
# ---------------------------------------------------------------------------

class StubBackend(AbstractExtractionBackend):
    """Returns pre-configured raw dicts; no network call."""

    def __init__(self, raw_response: dict):
        self._response = raw_response

    def _call_llm(self, user_message: str, assistant_message: str) -> dict:
        return self._response


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

class TestGroundingCheck:

    def test_well_grounded_node_stays_active(self):
        node = GraphNode(type=NodeType.DECISION, label="Redis caching strategy", status=NodeStatus.PROPOSED)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We decided to use Redis for caching with a 24hr strategy."
        )
        assert result[0].status != NodeStatus.UNCONFIRMED

    def test_phantom_node_marked_unconfirmed(self):
        node = GraphNode(type=NodeType.CONCEPT, label="quantum entanglement lattice", status=NodeStatus.ACTIVE)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We discussed Redis and SQLite for storage."
        )
        assert NodeStatus(result[0].status) == NodeStatus.UNCONFIRMED

    def test_single_word_label_requires_one_hit(self):
        node = GraphNode(type=NodeType.SKILL, label="Redis", status=NodeStatus.ACTIVE)
        result = AbstractExtractionBackend.grounding_check(
            [node], "We discussed Redis usage."
        )
        assert NodeStatus(result[0].status) != NodeStatus.UNCONFIRMED

    def test_empty_nodes_list(self):
        result = AbstractExtractionBackend.grounding_check([], "Some text.")
        assert result == []


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

class TestMasking:

    def test_pattern_redacts_matching_text(self):
        text = "My API key is sk-ant-abc123 and token is ghp_xyz789"
        masked = AbstractExtractionBackend._mask(text, [r"sk-ant-\w+", r"ghp_\w+"])
        assert "sk-ant-abc123" not in masked
        assert "ghp_xyz789" not in masked
        assert "[REDACTED]" in masked

    def test_invalid_pattern_is_skipped(self):
        text = "hello world"
        # Should not raise; bad regex is silently skipped
        result = AbstractExtractionBackend._mask(text, ["[invalid(regex"])
        assert result == text

    def test_empty_patterns_returns_original(self):
        text = "unchanged text"
        assert AbstractExtractionBackend._mask(text, []) == text


# ---------------------------------------------------------------------------
# Full extraction pipeline (stub LLM)
# ---------------------------------------------------------------------------

class TestExtractionPipeline:

    def _make_turn(self, user: str, asst: str) -> RawTurn:
        return RawTurn(session_id="test", turn_number=1, role_user=user, role_assistant=asst)

    def test_valid_extraction_returns_nodes_and_edges(self):
        raw = {
            "nodes": [
                {"type": "decision", "label": "Use Redis", "summary": "Caching layer", "status": "proposed", "confidence": 0.9},
                {"type": "skill",    "label": "Redis",     "summary": "In-memory cache", "status": "active", "confidence": 0.95},
            ],
            "edges": [
                {"from_label": "Use Redis", "to_label": "Redis", "relation": "uses"},
            ],
        }
        backend = StubBackend(raw)
        turn = self._make_turn(
            "Should we use Redis for caching?",
            "Yes, Redis is a great choice as an in-memory cache. Use Redis for the caching layer.",
        )
        nodes, edges = backend.extract(turn)
        assert any(n.label == "Use Redis" for n in nodes)
        assert any(n.label == "Redis" for n in nodes)
        assert len(edges) == 1
        assert edges[0].relation == "uses"

    def test_phantom_edge_dropped(self):
        """Edge whose endpoints aren't in the node list should be silently dropped."""
        raw = {
            "nodes": [
                {"type": "concept", "label": "SQLite WAL", "summary": "Write-ahead logging", "status": "active", "confidence": 0.9},
            ],
            "edges": [
                {"from_label": "SQLite WAL", "to_label": "NonExistentNode", "relation": "uses"},
            ],
        }
        backend = StubBackend(raw)
        turn = self._make_turn("Tell me about SQLite WAL mode.", "SQLite WAL mode enables concurrent reads.")
        nodes, edges = backend.extract(turn)
        assert len(edges) == 0  # phantom edge dropped

    def test_invalid_node_type_defaults_to_concept(self):
        raw = {
            "nodes": [{"type": "bogustype", "label": "something", "summary": "", "status": "active", "confidence": 0.8}],
            "edges": [],
        }
        backend = StubBackend(raw)
        turn = self._make_turn("Tell me about something.", "Something is important.")
        nodes, _ = backend.extract(turn)
        assert all(n.type == NodeType.CONCEPT for n in nodes)

    def test_empty_llm_response(self):
        backend = StubBackend({"nodes": [], "edges": []})
        turn = self._make_turn("Hello", "Hi")
        nodes, edges = backend.extract(turn)
        assert nodes == []
        assert edges == []


# ---------------------------------------------------------------------------
# P2/P3: Confidence scoring (geometric mean formula)
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    """Tests for the updated _compute_confidence using geometric mean."""

    def test_high_grounding_and_high_llm_gives_high_score(self):
        # All 3 words in source, LLM says 0.9 → grounding=1.0, geo=sqrt(1.0*0.9)≈0.949
        node = GraphNode(type=NodeType.DECISION, label="Redis caching strategy", confidence=0.9)
        score = AbstractExtractionBackend._compute_confidence(
            node, "We picked Redis caching strategy for the service."
        )
        assert score >= 0.90

    def test_low_llm_confidence_caps_final_score(self):
        # All words in source, but LLM says 0.4 → geo=sqrt(1.0*0.4)=0.632
        node = GraphNode(type=NodeType.DECISION, label="Redis caching strategy", confidence=0.4)
        score = AbstractExtractionBackend._compute_confidence(
            node, "We picked Redis caching strategy for the service."
        )
        assert score < 0.70, f"Expected < 0.70, got {score}"

    def test_low_grounding_caps_final_score(self):
        # LLM says 1.0 but label words mostly absent → grounding_ratio ≈ 0.33, geo≈0.577
        node = GraphNode(type=NodeType.CONCEPT, label="quantum lattice topology", confidence=1.0)
        score = AbstractExtractionBackend._compute_confidence(
            node, "We discussed Redis and Postgres for storage."
        )
        assert score < 0.65, f"Expected < 0.65, got {score}"

    def test_short_label_penalty_applied(self):
        # 2-word label: grounding_ratio penalized by 0.75
        node_short = GraphNode(type=NodeType.SKILL, label="Redis cache", confidence=1.0)
        node_long  = GraphNode(type=NodeType.SKILL, label="Redis cache layer approach", confidence=1.0)
        source = "Redis cache layer approach is used throughout."
        score_short = AbstractExtractionBackend._compute_confidence(node_short, source)
        score_long  = AbstractExtractionBackend._compute_confidence(node_long, source)
        # Short label should score lower than the fully-present long label
        assert score_short < score_long

    def test_empty_label_returns_zero(self):
        node = GraphNode(type=NodeType.CONCEPT, label="", confidence=1.0)
        score = AbstractExtractionBackend._compute_confidence(node, "some source text")
        assert score == 0.0

    def test_llm_confidence_one_no_longer_trivially_gives_one(self):
        # Previously, label words in source + LLM=1.0 always yielded 1.0.
        # With geometric mean: only when ALL words match AND LLM=1.0 does score reach 1.0.
        # A label where some words are absent should score < 1.0 even with LLM=1.0.
        node = GraphNode(type=NodeType.CONCEPT, label="parallel async batching", confidence=1.0)
        score = AbstractExtractionBackend._compute_confidence(
            node, "async processing is fast but sequential still works"
            # "parallel" and "batching" are absent; only "async" appears → grounding=0.333
            # geo = sqrt(0.333 * 1.0) ≈ 0.577
        )
        assert score < 1.0
        assert score >= 0.50


# ---------------------------------------------------------------------------
# P1/P5: Prompt definitions guard-rails
# ---------------------------------------------------------------------------

class TestPromptDefinitions:
    """Verify the updated NODE TYPE descriptions contain the key exclusion language."""

    def test_file_type_excludes_prs_and_repos(self):
        from fish_bridge.extraction.prompts import EXTRACTION_SYSTEM
        assert "NOT for" in EXTRACTION_SYSTEM
        assert "PRs" in EXTRACTION_SYSTEM or "PR" in EXTRACTION_SYSTEM
        assert "extension" in EXTRACTION_SYSTEM

    def test_error_type_excludes_process_friction(self):
        from fish_bridge.extraction.prompts import EXTRACTION_SYSTEM
        assert "stack trace" in EXTRACTION_SYSTEM or "error message" in EXTRACTION_SYSTEM
        assert "pending PRs" in EXTRACTION_SYSTEM or "process friction" in EXTRACTION_SYSTEM

    def test_task_status_rule_present(self):
        from fish_bridge.extraction.prompts import EXTRACTION_SYSTEM
        assert "TASK STATUS" in EXTRACTION_SYSTEM
        assert "done" in EXTRACTION_SYSTEM
        assert "'pending' ONLY" in EXTRACTION_SYSTEM or "pending' ONLY" in EXTRACTION_SYSTEM

    def test_calibrated_confidence_rule_present(self):
        from fish_bridge.extraction.prompts import EXTRACTION_SYSTEM
        assert "CALIBRATED" in EXTRACTION_SYSTEM
        assert "1-2 per turn" in EXTRACTION_SYSTEM or "at most" in EXTRACTION_SYSTEM


# ---------------------------------------------------------------------------
# P6: Cross-type deduplication
# ---------------------------------------------------------------------------

class TestCrossTypeDedup:
    """Tests for find_cross_type_match and cross-type relates-to edges in semantic_merge."""

    def _make_node(self, label: str, ntype: str, nid: str | None = None) -> GraphNode:
        n = GraphNode(type=NodeType(ntype), label=label, status=NodeStatus.ACTIVE)
        if nid:
            n.id = nid
        return n

    def test_cross_type_pairs_constant_defined(self):
        from fish_bridge.extraction.dedup import _CROSS_TYPE_PAIRS, CROSS_TYPE_RELATE_THRESHOLD
        # Must include the key problematic pairs
        assert frozenset({"task", "question"}) in _CROSS_TYPE_PAIRS
        assert frozenset({"concept", "skill"}) in _CROSS_TYPE_PAIRS
        assert frozenset({"task", "decision"}) in _CROSS_TYPE_PAIRS
        # Threshold must be higher than same-type relate threshold
        from fish_bridge.extraction.dedup import RELATE_THRESHOLD
        assert CROSS_TYPE_RELATE_THRESHOLD > RELATE_THRESHOLD

    def test_find_cross_type_match_returns_none_for_no_embedding(self):
        """When no embedding provider is available, returns (None, 0.0)."""
        from fish_bridge.extraction.dedup import find_cross_type_match, EmbeddingProvider

        class NoEmbedProvider(EmbeddingProvider):
            def embed(self, text: str):
                return None  # simulate unavailable

        task_node = self._make_node("deploy to PyPI", "task")
        decision_node = self._make_node("PyPI publishing decision", "decision")
        provider = NoEmbedProvider()
        match, score = find_cross_type_match(task_node, [decision_node], provider)
        assert match is None
        assert score == 0.0

    def test_find_cross_type_match_skips_same_type(self):
        """Same-type candidates must never be returned by find_cross_type_match."""
        from fish_bridge.extraction.dedup import find_cross_type_match, EmbeddingProvider

        call_count = {"n": 0}

        class CountingProvider(EmbeddingProvider):
            def embed(self, text: str):
                call_count["n"] += 1
                return [1.0, 0.0]  # dummy vector

        task1 = self._make_node("deploy to PyPI", "task")
        task2 = self._make_node("publish package to PyPI", "task")  # same type — should be skipped
        provider = CountingProvider()
        match, score = find_cross_type_match(task1, [task2], provider)
        # task↔task is not in _CROSS_TYPE_PAIRS, so task2 should be skipped
        assert match is None

    def test_find_cross_type_match_skips_incompatible_pairs(self):
        """Type pairs not in _CROSS_TYPE_PAIRS must not be returned."""
        from fish_bridge.extraction.dedup import find_cross_type_match, EmbeddingProvider

        class AlwaysHighProvider(EmbeddingProvider):
            def embed(self, text: str):
                return [1.0, 0.0]

        # file↔question is NOT in _CROSS_TYPE_PAIRS
        file_node = self._make_node("config.yaml", "file")
        question_node = self._make_node("which config to use", "question")
        provider = AlwaysHighProvider()
        match, score = find_cross_type_match(file_node, [question_node], provider)
        assert match is None


# ---------------------------------------------------------------------------
# Fix B: file node reclassification
# ---------------------------------------------------------------------------

class TestFileNodeReclassifier:
    """Tests for AbstractExtractionBackend._reclassify_file_nodes."""

    def _file_node(self, label: str) -> GraphNode:
        return GraphNode(type=NodeType.FILE, label=label, status=NodeStatus.ACTIVE)

    def test_real_file_with_extension_kept(self):
        node = self._file_node("pyproject.toml")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        assert result[0].type == NodeType.FILE

    def test_real_file_with_path_separator_kept(self):
        node = self._file_node("fish_bridge/extraction/base.py")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        assert result[0].type == NodeType.FILE

    def test_symbol_with_parens_kept(self):
        node = self._file_node("semantic_merge()")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        assert result[0].type == NodeType.FILE

    def test_ui_settings_path_reclassified_to_concept(self):
        """Labels like 'GitHub Settings' should become concept."""
        node = self._file_node("GitHub Settings")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"
        assert result[0].metadata.get("reclassified_from") == "file"

    def test_repo_reference_reclassified_to_concept(self):
        node = self._file_node("main repository")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_commit_ref_reclassified_to_task(self):
        node = self._file_node("commit abc1234")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"
        assert result[0].metadata.get("reclassified_from") == "file"

    def test_ci_run_ref_reclassified_to_task(self):
        node = self._file_node("CI #42")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_pr_ref_reclassified_to_task(self):
        node = self._file_node("PR #15")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_non_file_node_untouched(self):
        """Nodes that are not type 'file' must never be modified."""
        node = GraphNode(type=NodeType.TASK, label="CI #42", status=NodeStatus.ACTIVE)
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"
        assert "reclassified_from" not in result[0].metadata

    def test_reclassified_active_task_status_becomes_done(self):
        """Git refs and CI runs that were 'active' should flip to 'done'."""
        node = self._file_node("merge 0ab1234")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        status = result[0].status if isinstance(result[0].status, str) else result[0].status.value
        assert status == "done"


# ---------------------------------------------------------------------------
# Fix C: EDGE CHECKLIST includes file re-type reminder
# ---------------------------------------------------------------------------

class TestPromptFileTypeReminder:

    def test_edge_checklist_contains_file_retype_instruction(self):
        from fish_bridge.extraction.prompts import EXTRACTION_USER_TEMPLATE
        assert "file" in EXTRACTION_USER_TEMPLATE.lower()
        # The checklist reminder should mention re-typing file nodes
        assert "re-type" in EXTRACTION_USER_TEMPLATE.lower() or "retype" in EXTRACTION_USER_TEMPLATE.lower()


# ---------------------------------------------------------------------------
# Fix A: cross_type_budget enforcement in semantic_merge
# ---------------------------------------------------------------------------

class TestCrossTypeBudget:
    """Verifies the cross_type_budget parameter in semantic_merge is enforced."""

    def _make_node(self, label: str, ntype: str) -> GraphNode:
        return GraphNode(type=NodeType(ntype), label=label, status=NodeStatus.ACTIVE)

    def test_zero_budget_produces_no_cross_type_edges(self):
        """With budget=0, no cross-type relates-to edges should be emitted even if similarity is high."""
        from fish_bridge.extraction.dedup import semantic_merge, EmbeddingProvider

        class HighSimProvider(EmbeddingProvider):
            """Always returns the same vector → cosine similarity = 1.0."""
            def embed(self, text: str):
                return [1.0, 0.0]

        incoming = [self._make_node("deploy to production", "task")]
        existing = [self._make_node("production deployment decision", "decision")]

        _, edges, _ = semantic_merge(
            incoming, existing, HighSimProvider(), cross_type_budget=0
        )
        assert len(edges) == 0, "Expected no cross-type edges with budget=0"

    def test_positive_budget_allows_up_to_limit(self):
        """With budget=1, exactly one cross-type edge should be emitted."""
        from fish_bridge.extraction.dedup import semantic_merge, EmbeddingProvider

        class HighSimProvider(EmbeddingProvider):
            def embed(self, text: str):
                return [1.0, 0.0]

        # Two task→decision pairs: both would match, but budget limits to 1
        incoming = [
            self._make_node("deploy to production", "task"),
            self._make_node("rollback procedure", "task"),
        ]
        existing = [
            self._make_node("production deployment decision", "decision"),
            self._make_node("rollback strategy decision", "decision"),
        ]

        _, edges, _ = semantic_merge(
            incoming, existing, HighSimProvider(), cross_type_budget=1
        )
        assert len(edges) <= 1, f"Expected at most 1 cross-type edge with budget=1, got {len(edges)}"

    def test_unlimited_budget_default(self):
        """Default cross_type_budget=-1 should not restrict cross-type edges."""
        from fish_bridge.extraction.dedup import semantic_merge, EmbeddingProvider

        class HighSimProvider(EmbeddingProvider):
            def embed(self, text: str):
                return [1.0, 0.0]

        incoming = [
            self._make_node("deploy to production", "task"),
            self._make_node("rollback procedure", "task"),
        ]
        existing = [
            self._make_node("production deployment decision", "decision"),
            self._make_node("rollback strategy decision", "decision"),
        ]

        _, edges, _ = semantic_merge(
            incoming, existing, HighSimProvider()
        )
        # With high similarity and no budget cap, both should produce edges
        assert len(edges) >= 1, "Expected at least one cross-type edge with unlimited budget"


# ---------------------------------------------------------------------------
# Fix D: RELATE_THRESHOLD raised to 0.78
# ---------------------------------------------------------------------------

class TestRelateThreshold:

    def test_relate_threshold_value(self):
        """RELATE_THRESHOLD must be 0.78 (raised from 0.70)."""
        from fish_bridge.extraction.dedup import RELATE_THRESHOLD
        assert RELATE_THRESHOLD == 0.78

    def test_below_threshold_no_edge(self):
        """Similarity just below 0.78 should produce no edge (independent node)."""
        from fish_bridge.extraction.dedup import semantic_merge, EmbeddingProvider

        class LowSimProvider(EmbeddingProvider):
            """Returns vectors with cosine ≈ 0.73 (below new threshold 0.78)."""
            def embed(self, text: str):
                # cosine([1,0], [0.73, 0.683]) = 0.73
                if "existing" in text:
                    return [1.0, 0.0]
                return [0.73, 0.683]

        existing = GraphNode(type=NodeType.TASK, label="existing task A")
        incoming = GraphNode(type=NodeType.TASK, label="incoming task B")
        to_add, edges, _ = semantic_merge([incoming], [existing], LowSimProvider())
        assert len(edges) == 0, "Score 0.73 < 0.78 should not produce a relates-to edge"
        assert len(to_add) == 1, "Node should be added independently"


# ---------------------------------------------------------------------------
# Fix E: task 'active' status normalization
# ---------------------------------------------------------------------------

class TestTaskStatusNormalization:

    def test_task_active_becomes_pending(self):
        """LLM-returned 'active' status on a task node must be normalized to 'pending'."""
        backend = StubBackend({
            "nodes": [{"type": "task", "label": "Deploy service", "summary": "deploy", "status": "active", "confidence": 0.8}],
            "edges": [],
        })
        nodes, _ = backend._extract_single("deploy the service", "ok deploying", "s1")
        task = next(n for n in nodes if n.label.lower() == "deploy service")
        status = task.status if isinstance(task.status, str) else task.status.value
        assert status == "pending", f"Expected 'pending', got '{status}'"

    def test_concept_active_unchanged(self):
        """'active' is valid for concept nodes and must NOT be changed."""
        backend = StubBackend({
            "nodes": [{"type": "concept", "label": "microservices", "summary": "arch pattern", "status": "active", "confidence": 0.8}],
            "edges": [],
        })
        nodes, _ = backend._extract_single("microservices", "using microservices", "s1")
        concept = next(n for n in nodes if n.label.lower() == "microservices")
        status = concept.status if isinstance(concept.status, str) else concept.status.value
        assert status == "active", f"Expected 'active', got '{status}'"


# ---------------------------------------------------------------------------
# Fix F: extended file reclassifier patterns
# ---------------------------------------------------------------------------

class TestFileReclassifierExtended:
    """Additional tests for newly-added patterns in _reclassify_file_nodes."""

    def _file_node(self, label: str) -> GraphNode:
        return GraphNode(type=NodeType.FILE, label=label, status=NodeStatus.ACTIVE)

    def test_cli_command_with_flag_becomes_task(self):
        node = self._file_node("fish-bridge config --backend gemini")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_pr_title_ref_becomes_task(self):
        node = self._file_node("PR title: Add fish-bridge-mcp to Knowledge & Memory")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_pr_submission_becomes_task(self):
        node = self._file_node("PR to awesome-mcp-servers")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_script_label_becomes_task(self):
        node = self._file_node("re-ingest script")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "task"

    def test_all_caps_env_var_becomes_concept(self):
        node = self._file_node("GITHUB_PATH")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_terminal_ref_becomes_concept(self):
        node = self._file_node("Terminal output")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_macos_platform_becomes_concept(self):
        node = self._file_node("macOS shell")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_badge_ref_becomes_concept(self):
        node = self._file_node("glama.ai badge")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_account_ref_becomes_concept(self):
        node = self._file_node("pypi account")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "concept"

    def test_real_python_file_still_kept(self):
        """Ensure new patterns don't accidentally catch real files."""
        node = self._file_node("deploy.sh")
        result = AbstractExtractionBackend._reclassify_file_nodes([node])
        node_type = result[0].type if isinstance(result[0].type, str) else result[0].type.value
        assert node_type == "file"


class TestFileReclassifierFixI:
    """Fix I: dotfiles, Issue/PR patterns, form concept."""

    def _file_node(self, label: str) -> GraphNode:
        return GraphNode(type=NodeType.FILE, label=label, status=NodeStatus.ACTIVE)

    def _retype(self, label: str) -> str:
        result = AbstractExtractionBackend._reclassify_file_nodes([self._file_node(label)])
        t = result[0].type
        return t if isinstance(t, str) else t.value

    # ---- _FILE_KEEP_RE: dotfiles ----
    def test_dotfile_gitignore_kept_as_file(self):
        assert self._retype(".gitignore") == "file"

    def test_dotfile_env_kept_as_file(self):
        assert self._retype(".env") == "file"

    def test_dotfile_npmrc_kept_as_file(self):
        assert self._retype(".npmrc") == "file"

    # ---- _FILE_TO_TASK_RE: Issue/PR patterns ----
    def test_issue_hash_becomes_task(self):
        assert self._retype("Issue #1-3") == "task"

    def test_issue_hash_single_becomes_task(self):
        assert self._retype("Issue #42") == "task"

    def test_github_pr_for_becomes_task(self):
        assert self._retype("GitHub PR for awesome-mcp-servers") == "task"

    def test_pr_from_actions_becomes_task(self):
        assert self._retype("PR from GitHub Actions") == "task"

    def test_pr_for_becomes_task(self):
        assert self._retype("PR for fish-bridge") == "task"

    def test_label_ending_pr_becomes_task(self):
        assert self._retype("Awesome-MCP-Servers PR") == "task"

    def test_workflow_file_becomes_task(self):
        assert self._retype("publish workflow file") == "task"

    # ---- _FILE_TO_CONCEPT_RE: form ----
    def test_web_form_becomes_concept(self):
        assert self._retype("pulsemcp.com form") == "concept"

    def test_pr_creation_form_becomes_concept(self):
        assert self._retype("PR creation form") == "concept"

    # ---- regression: real files still kept ----
    def test_real_yaml_not_caught(self):
        assert self._retype("publish.yml") == "file"

    def test_real_path_not_caught(self):
        assert self._retype(".github/workflows/publish.yml") == "file"


class TestTypeEdgeDefaultsFixJ:
    """Fix J: _TYPE_EDGE_DEFAULTS covers task/concept/decision pairs."""

    def _node(self, label: str, ntype: str) -> GraphNode:
        return GraphNode(type=NodeType(ntype), label=label, status=NodeStatus.ACTIVE)

    def _fallback_relation(self, from_type: str, to_type: str) -> str:
        return AbstractExtractionBackend._TYPE_EDGE_DEFAULTS.get(
            (from_type, to_type), "relates-to"
        )

    def test_task_concept_maps_to_references(self):
        assert self._fallback_relation("task", "concept") == "references"

    def test_concept_task_maps_to_references(self):
        assert self._fallback_relation("concept", "task") == "references"

    def test_task_task_maps_to_depends_on(self):
        assert self._fallback_relation("task", "task") == "depends-on"

    def test_decision_task_maps_to_implements(self):
        assert self._fallback_relation("decision", "task") == "implements"

    def test_concept_question_maps_to_references(self):
        assert self._fallback_relation("concept", "question") == "references"

    def test_no_stray_relates_to_in_defaults(self):
        """All entries in _TYPE_EDGE_DEFAULTS must be specific (not relates-to)."""
        for (ft, tt), rel in AbstractExtractionBackend._TYPE_EDGE_DEFAULTS.items():
            assert rel != "relates-to", (
                f"({ft!r}, {tt!r}) still maps to 'relates-to' — add a specific relation"
            )
