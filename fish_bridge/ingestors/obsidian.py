"""ObsidianIngestor — Obsidian vault reader.

Converts an Obsidian markdown vault into RawTurns for LLM extraction.
Each note becomes one RawTurn, containing:
  - role_user: note metadata (path, tags, status from frontmatter)
  - role_assistant: note content (wikilinks expanded inline)

Wikilinks [[target]] are preserved in the text so the extractor can create
concept→documents→concept edges.  They are also returned separately via
extract_wikilink_edges() for direct graph insertion without LLM cost.

Usage:
    fish-bridge merge --source obsidian --vault ~/Documents/MyVault
    fish-bridge merge --source obsidian --vault ~/Documents/MyVault --tag project
    fish-bridge merge --source obsidian --vault ~/Documents/MyVault --folder Projects/
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fish_bridge.graph.schema import EdgeRelation, GraphEdge, GraphNode, NodeType, RawTurn
from fish_bridge.ingestors.base import AbstractIngestor

_MAX_NOTE_CHARS = 4000
_WIKILINK_RE    = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_TAG_RE         = re.compile(r"^#([A-Za-z0-9_/-]+)")


class ObsidianIngestor(AbstractIngestor):
    """Reads an Obsidian markdown vault and returns RawTurns for extraction."""

    def ingest(  # type: ignore[override]
        self,
        vault_path: Path | str | None = None,
        session_id: str = "obsidian-session",
        tag_filter: str | None = None,
        folder_filter: str | None = None,
        max_notes: int = 500,
    ) -> list[RawTurn]:
        """Return RawTurns from the Obsidian vault.

        Args:
            vault_path:    Path to the Obsidian vault root directory.
            session_id:    Session ID to stamp on returned turns.
            tag_filter:    Only include notes with this tag (e.g. "project").
            folder_filter: Only include notes under this subfolder.
            max_notes:     Cap to avoid overwhelming the graph (default: 500).
        """
        if vault_path is None:
            raise ValueError("vault_path is required.")
        vault = Path(vault_path).resolve()
        if not vault.is_dir():
            raise NotADirectoryError(f"Vault not found: {vault}")

        notes = list(_find_notes(vault, folder_filter, max_notes))
        turns: list[RawTurn] = []

        for i, note_path in enumerate(notes, start=1):
            try:
                raw_content = note_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            frontmatter, body = _parse_frontmatter(raw_content)

            # Apply tag filter
            if tag_filter:
                note_tags = _extract_tags(frontmatter, body)
                if tag_filter.lower() not in {t.lower() for t in note_tags}:
                    continue

            # Build synthetic user/assistant pair
            rel_path = note_path.relative_to(vault)
            user_msg = _build_user_context(rel_path, frontmatter)
            asst_msg = body[:_MAX_NOTE_CHARS]

            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_number=i,
                    role_user=user_msg,
                    role_assistant=asst_msg,
                    source="obsidian",
                )
            )

        return turns

    @staticmethod
    def extract_wikilink_edges(
        vault_path: Path | str,
        existing_labels: set[str],
    ) -> list[tuple[str, str]]:
        """Return list of (source_label, target_label) wikilink pairs.

        Useful for creating concept→documents→concept edges directly
        without LLM extraction.  Only returns edges where both sides
        are in existing_labels.
        """
        vault = Path(vault_path).resolve()
        pairs: list[tuple[str, str]] = []
        label_lower = {l.lower(): l for l in existing_labels}

        for note_path in vault.rglob("*.md"):
            try:
                content = note_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source_label = note_path.stem
            if source_label.lower() not in label_lower:
                continue
            canonical_source = label_lower[source_label.lower()]
            for m in _WIKILINK_RE.finditer(content):
                target = m.group(1).strip()
                if target.lower() in label_lower:
                    pairs.append((canonical_source, label_lower[target.lower()]))

        return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_notes(
    vault: Path,
    folder_filter: str | None,
    max_notes: int,
) -> list[Path]:
    """Return markdown files in vault, optionally filtered by subfolder."""
    if folder_filter:
        search_root = vault / folder_filter
        if not search_root.is_dir():
            search_root = vault  # fallback to full vault
    else:
        search_root = vault

    notes: list[Path] = []
    # Skip .obsidian system directory
    for p in sorted(search_root.rglob("*.md")):
        if ".obsidian" in p.parts:
            continue
        notes.append(p)
        if len(notes) >= max_notes:
            break
    return notes


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter dict and remaining body text."""
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}, content
    try:
        import yaml
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            fm = {}
    except Exception:
        fm = {}
    body = content[m.end():].strip()
    return fm, body


def _extract_tags(frontmatter: dict[str, Any], body: str) -> list[str]:
    """Collect tags from frontmatter 'tags' field and inline #hashtags."""
    tags: list[str] = []

    # Frontmatter tags: list or space-separated string
    fm_tags = frontmatter.get("tags", [])
    if isinstance(fm_tags, list):
        tags.extend(str(t) for t in fm_tags)
    elif isinstance(fm_tags, str):
        tags.extend(fm_tags.split())

    # Inline hashtags in body
    for line in body.splitlines():
        for m in _TAG_RE.finditer(line):
            tags.append(m.group(1))

    return tags


def _build_user_context(rel_path: Path, frontmatter: dict[str, Any]) -> str:
    """Build a short context string for the user role of the RawTurn."""
    parts = [f"Obsidian note: {rel_path}"]
    if frontmatter.get("status"):
        parts.append(f"status: {frontmatter['status']}")
    if frontmatter.get("tags"):
        tags = frontmatter["tags"]
        if isinstance(tags, list):
            parts.append(f"tags: {', '.join(str(t) for t in tags[:5])}")
        elif isinstance(tags, str):
            parts.append(f"tags: {tags}")
    if frontmatter.get("area") or frontmatter.get("type"):
        area = frontmatter.get("area") or frontmatter.get("type")
        parts.append(f"area: {area}")
    return " | ".join(parts)
