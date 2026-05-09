"""AbstractCompiler — shared base for all fish_bridge compiler modes."""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from fish_bridge.graph.schema import GraphEdge, GraphNode


@contextmanager
def _output_file_lock(output_path: Path) -> Generator[None, None, None]:
    """Advisory file lock to serialise concurrent writes to the same output file.

    Uses fcntl.flock (POSIX) on macOS/Linux and msvcrt.locking (Windows).
    Falls back to a no-op on platforms where neither is available so the
    compiler never crashes — concurrent writes are merely unprotected there.
    """
    lock_path = output_path.parent / f".{output_path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except (ImportError, OSError):
        # Graceful degradation: no lock, allow the write through
        yield
    finally:
        fh.close()


class AbstractCompiler(ABC):
    """Base class for Mode A (active thread), Mode B (focus), Mode C (digest)."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    @abstractmethod
    def compile(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        **kwargs,
    ) -> str:
        """Produce the compiled output string."""
        ...

    def write(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_file: Path,
        **kwargs,
    ) -> None:
        """Compile and write output to a file, with an advisory file lock."""
        content = self.compile(nodes, edges, **kwargs)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with _output_file_lock(output_file):
            output_file.write_text(content, encoding="utf-8")
