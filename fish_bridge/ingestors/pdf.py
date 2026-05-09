"""PDFIngestor — extract text from PDF files for knowledge-graph ingestion.

Requires the optional ``pdf`` extra::

    pip install "fish_bridge[pdf]"      # installs pypdf>=4.0

Each page of the PDF becomes one RawTurn fed into the extraction pipeline.
Empty pages are skipped.  Very long pages are split at ``_MAX_PAGE_CHARS``
to stay within LLM token budgets.

Usage (CLI)::

    fish-bridge merge --source pdf --file design-spec.pdf
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from fish_bridge.graph.schema import RawTurn
from fish_bridge.ingestors.base import AbstractIngestor

_MAX_PAGE_CHARS = 4_000  # Characters per synthetic turn


class PDFIngestor(AbstractIngestor):
    """Convert a PDF file into RawTurns using *pypdf*."""

    def ingest(  # type: ignore[override]
        self,
        file_path: Path | str,
        session_id: str = "pdf-session",
    ) -> list[RawTurn]:
        """Extract text from *file_path* and return one RawTurn per page chunk.

        Args:
            file_path:  Path to the PDF file.
            session_id: Session ID stamped on the returned turns.

        Raises:
            ImportError:     If ``pypdf`` is not installed.
            FileNotFoundError: If *file_path* does not exist.
        """
        try:
            from pypdf import PdfReader  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF ingestion.  "
                "Install with: pip install 'fish_bridge[pdf]'"
            ) from exc

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        reader = PdfReader(str(path))
        turns: list[RawTurn] = []
        turn_number = 0
        source_name = path.name

        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = (page.extract_text() or "").strip()
            if not raw_text:
                continue

            # Split long pages into chunks
            chunks = _chunk_text(raw_text, _MAX_PAGE_CHARS)
            for chunk_idx, chunk in enumerate(chunks, start=1):
                turn_number += 1
                label = (
                    f"Document: {source_name} — page {page_num}"
                    + (f" part {chunk_idx}" if len(chunks) > 1 else "")
                )
                turns.append(
                    RawTurn(
                        session_id=session_id,
                        turn_number=turn_number,
                        role_user=label,
                        role_assistant=chunk,
                        source="pdf",
                    )
                )

        return turns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars*, breaking on paragraphs."""
    if len(text) <= max_chars:
        return [text]

    # Try paragraph-level splits first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        # If a single paragraph exceeds max_chars, hard-wrap it
        if len(para) > max_chars:
            for line in textwrap.wrap(para, max_chars):
                chunks.append(line)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text[:max_chars]]
