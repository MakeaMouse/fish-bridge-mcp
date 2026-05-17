"""DocumentIngestor — markdown / JSON / YAML document ingestor.

Converts structured documents into RawTurn objects for LLM extraction.
Each logical section of the document becomes one RawTurn, keeping individual
turns under the token budget of the extraction backend.

Supported formats:
  - Markdown (.md, .mdx): split on H2/H3 headings
  - JSON (.json): top-level keys become individual turns
  - YAML (.yaml, .yml): top-level keys become individual turns
  - Plain text (.txt): split on blank-line paragraphs (~1500 chars/chunk)

Usage:
    fish-bridge merge --source document --file spec.md
    fish-bridge merge --source document --file openapi.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.base import AbstractIngestor

# Threshold above which a section is chunked further
_MAX_SECTION_CHARS = 4000


class DocumentIngestor(AbstractIngestor):
    """Convert a document file into RawTurns for extraction."""

    def ingest(  # type: ignore[override]
        self,
        file_path: Path | str | None = None,
        text: str | None = None,
        session_id: str = "doc-session",
        format: str | None = None,
    ) -> list[RawTurn]:
        """Return RawTurns from a document file or raw text string.

        Args:
            file_path:  Path to the document file.
            text:       Raw document text (used when file_path is None).
            session_id: Session ID to stamp on returned turns.
            format:     Override format detection: "markdown" | "json" | "yaml" | "text".
        """
        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")
            content = path.read_text(encoding="utf-8", errors="replace")
            fmt = format or _detect_format(path)
            source_name = path.name
        elif text is not None:
            content = text
            fmt = format or "text"
            source_name = "pasted-document"
        else:
            raise ValueError("Provide file_path or text.")

        sections = _split_document(content, fmt)
        turns: list[RawTurn] = []

        for i, (heading, body) in enumerate(sections, start=1):
            if not body.strip():
                continue
            # Build a synthetic user/assistant pair.
            # "user" = document source context, "assistant" = section content.
            user_msg = (
                f"Document: {source_name}"
                + (f" — section: {heading}" if heading else "")
            )
            turns.append(
                RawTurn(
                    session_id=session_id,
                    turn_number=i,
                    role_user=user_msg,
                    role_assistant=body[:_MAX_SECTION_CHARS],
                    source="document",
                )
            )

        return turns


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".mdx", ".markdown"):
        return "markdown"
    if suffix == ".json":
        return "json"
    if suffix in (".yaml", ".yml"):
        return "yaml"
    return "text"


# ---------------------------------------------------------------------------
# Document splitters
# ---------------------------------------------------------------------------

def _split_document(content: str, fmt: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) pairs."""
    if fmt == "markdown":
        return _split_markdown(content)
    if fmt == "json":
        return _split_json(content)
    if fmt == "yaml":
        return _split_yaml(content)
    return _split_text(content)


def _split_markdown(content: str) -> list[tuple[str, str]]:
    """Split markdown on H1/H2/H3 headings."""
    # Pattern: line starting with 1–3 # characters
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(content))

    if not matches:
        # No headings — treat the whole document as one section
        return [("", content)]

    sections: list[tuple[str, str]] = []

    # Content before the first heading
    preamble = content[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for idx, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()
        if len(body) > _MAX_SECTION_CHARS:
            # Sub-chunk long sections by blank lines
            sub_chunks = _chunk_by_paragraphs(body, _MAX_SECTION_CHARS)
            for j, chunk in enumerate(sub_chunks):
                sections.append((f"{heading} ({j + 1}/{len(sub_chunks)})", chunk))
        else:
            sections.append((heading, body))

    return sections


def _split_json(content: str) -> list[tuple[str, str]]:
    """Split a JSON document: each top-level key becomes one section."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [("", content)]  # Treat as plain text if not valid JSON

    if not isinstance(data, dict):
        # Array or scalar — treat whole thing as one section
        return [("", json.dumps(data, indent=2))]

    sections: list[tuple[str, str]] = []
    for key, value in data.items():
        body = json.dumps({key: value}, indent=2, ensure_ascii=False)
        sections.append((str(key), body[:_MAX_SECTION_CHARS]))
    return sections


def _split_yaml(content: str) -> list[tuple[str, str]]:
    """Split a YAML document: each top-level key becomes one section."""
    try:
        import yaml  # core dependency
        data = yaml.safe_load(content)
    except Exception:
        return [("", content)]

    if not isinstance(data, dict):
        return [("", content)]

    sections: list[tuple[str, str]] = []
    for key, value in data.items():
        try:
            import yaml as _yaml
            body = _yaml.dump({key: value}, default_flow_style=False, allow_unicode=True)
        except Exception:
            body = str(value)
        sections.append((str(key), body[:_MAX_SECTION_CHARS]))
    return sections


def _split_text(content: str) -> list[tuple[str, str]]:
    """Split plain text on double newlines into ~_MAX_SECTION_CHARS chunks."""
    chunks = _chunk_by_paragraphs(content, _MAX_SECTION_CHARS)
    return [("", chunk) for chunk in chunks]


def _chunk_by_paragraphs(text: str, max_len: int) -> list[str]:
    """Split text on blank lines, merging paragraphs up to max_len chars."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > max_len and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).lstrip("\n")
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_len]]
