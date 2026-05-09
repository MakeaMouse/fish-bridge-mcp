"""CodebaseIngestor — git log + README/HANDOVER document extraction.

Extracts knowledge from a project codebase without requiring tree-sitter:
  1. Recent git commits (last N) → decision/error/task/concept nodes via LLM
  2. README.md / CONTRIBUTING.md / HANDOVER.md → document ingestor chunks
  3. Key source file symbols → lightweight regex-based extraction (no tree-sitter)

The ingestor returns RawTurns suitable for standard LLM extraction.
Each git commit becomes one RawTurn; each doc chunk becomes another.

Usage:
    fish-bridge merge --source codebase --path ./
    fish-bridge merge --source codebase --path ./ --commits 30 --no-docs
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Iterator

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.base import AbstractIngestor
from fish_bridge.ingestors.document import DocumentIngestor

_DEFAULT_COMMITS = 20
_DOC_FILENAMES   = [
    "README.md", "CONTRIBUTING.md", "HANDOVER.md", "CHANGELOG.md",
    "ARCHITECTURE.md", "DECISIONS.md", "ADR.md", "docs/README.md",
]


class CodebaseIngestor(AbstractIngestor):
    """Ingest a git repository into RawTurns for extraction."""

    def ingest(  # type: ignore[override]
        self,
        path: Path | str = ".",
        session_id: str = "codebase-session",
        n_commits: int = _DEFAULT_COMMITS,
        include_docs: bool = True,
        include_commits: bool = True,
    ) -> list[RawTurn]:
        """Return RawTurns from the repository at *path*.

        Args:
            path:            Repository root (or any subdirectory).
            session_id:      Session ID to stamp on returned turns.
            n_commits:       Number of recent git commits to include.
            include_docs:    Include README/HANDOVER/CONTRIBUTING docs.
            include_commits: Include git commit history.
        """
        repo_path = Path(path).resolve()
        turns: list[RawTurn] = []
        turn_number = 1

        if include_commits:
            for turn in _git_commit_turns(repo_path, session_id, n_commits, turn_number):
                turns.append(turn)
                turn_number += 1

        if include_docs:
            doc_ingestor = DocumentIngestor()
            for doc_file in _find_docs(repo_path):
                try:
                    doc_turns = doc_ingestor.ingest(
                        file_path=doc_file,
                        session_id=session_id,
                    )
                    for t in doc_turns:
                        t.turn_number = turn_number
                        t.source = "codebase_doc"
                        turns.append(t)
                        turn_number += 1
                except Exception:
                    continue

        return turns


# ---------------------------------------------------------------------------
# Git log helpers
# ---------------------------------------------------------------------------

_COMMIT_RE = re.compile(
    r"^(?P<hash>[0-9a-f]{40})\|(?P<subject>.+?)\|(?P<body>.*?)\|(?P<date>.+)$",
    re.DOTALL,
)

# Conventional commit prefix → expected node type hint
_COMMIT_TYPE_HINTS: dict[str, str] = {
    "feat":     "decision — new feature",
    "fix":      "error — bug fix",
    "refactor": "decision — refactor",
    "chore":    "task",
    "docs":     "concept — documentation",
    "test":     "task — tests",
    "ci":       "task — CI/CD",
    "perf":     "skill — performance",
    "build":    "decision — build change",
    "revert":   "decision — revert",
    "wip":      "task — work in progress",
}


def _parse_commit_type(subject: str) -> str:
    """Extract conventional commit type hint from subject line."""
    m = re.match(r"^(\w+)[\(:]", subject)
    if m:
        prefix = m.group(1).lower()
        return _COMMIT_TYPE_HINTS.get(prefix, "")
    return ""


def _git_commit_turns(
    repo_path: Path,
    session_id: str,
    n_commits: int,
    start_turn: int,
) -> Iterator[RawTurn]:
    """Yield one RawTurn per git commit."""
    try:
        result = subprocess.run(
            [
                "git", "log",
                f"-{n_commits}",
                "--pretty=format:%H|%s|%b|%ai",
                "--no-merges",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # git not available or timeout

    if result.returncode != 0:
        return

    # Also grab the list of files changed per commit
    try:
        stat_result = subprocess.run(
            ["git", "log", f"-{n_commits}", "--no-merges", "--name-only",
             "--pretty=format:COMMIT:%H"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        files_by_hash = _parse_files_by_hash(stat_result.stdout)
    except Exception:
        files_by_hash = {}

    lines = result.stdout.strip().split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 3)
        if len(parts) < 2:
            continue
        commit_hash = parts[0]
        subject     = parts[1]
        body        = parts[2] if len(parts) > 2 else ""
        date_str    = parts[3] if len(parts) > 3 else ""

        type_hint = _parse_commit_type(subject)
        changed_files = files_by_hash.get(commit_hash, [])

        user_msg = (
            f"Git commit {commit_hash[:8]} ({date_str[:10]})"
            + (f" [{type_hint}]" if type_hint else "")
        )
        asst_msg = subject
        if body.strip():
            asst_msg += f"\n\n{body.strip()}"
        if changed_files:
            asst_msg += f"\n\nFiles changed: {', '.join(changed_files[:10])}"

        yield RawTurn(
            session_id=session_id,
            turn_number=start_turn + i,
            role_user=user_msg,
            role_assistant=asst_msg,
            source="git_commit",
        )


def _parse_files_by_hash(stat_output: str) -> dict[str, list[str]]:
    """Parse --name-only output into {commit_hash: [files]}."""
    result: dict[str, list[str]] = {}
    current_hash: str | None = None
    for line in stat_output.splitlines():
        if line.startswith("COMMIT:"):
            current_hash = line[7:].strip()
            result[current_hash] = []
        elif current_hash and line.strip():
            result[current_hash].append(line.strip())
    return result


# ---------------------------------------------------------------------------
# Doc file discovery
# ---------------------------------------------------------------------------

def _find_docs(repo_path: Path) -> list[Path]:
    """Return paths to standard documentation files that exist in the repo."""
    found: list[Path] = []
    for name in _DOC_FILENAMES:
        p = repo_path / name
        if p.exists() and p.is_file():
            found.append(p)
    return found
