"""Source text for search results, read from disk when a hit needs it.

Search hits are padded with surrounding lines before they reach the model. Those lines
used to come from a snapshot of every indexed file taken at startup, so a file edited
during the session (by the agent or in an editor) was shown as it was before the edit.
Reading on demand keeps the context current and avoids loading the whole repository.
"""

from __future__ import annotations

from pathlib import Path

MAX_SOURCE_BYTES = 2_000_000


class LiveSources:
    """`path -> current file text`, cached until the file changes on disk."""

    def __init__(self, root: Path, *, indexed_at: float | None = None) -> None:
        self._root = root.resolve()
        self._indexed_at = indexed_at
        self._cache: dict[str, tuple[int, int, str]] = {}

    def _resolve(self, path: str) -> Path | None:
        full = (self._root / path).resolve()
        if not full.is_relative_to(self._root) or not full.is_file():
            return None
        return full

    def get(self, path: str, default: str | None = None) -> str | None:
        full = self._resolve(path)
        if full is None:
            return default
        try:
            stat = full.stat()
            if stat.st_size > MAX_SOURCE_BYTES:
                return default
            cached = self._cache.get(path)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                return cached[2]
            text = full.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return default
        self._cache[path] = (stat.st_mtime_ns, stat.st_size, text)
        return text

    def changed_since_index(self, path: str) -> bool:
        """True when the file was modified after the index was built, so its chunks are old."""
        if self._indexed_at is None:
            return False
        full = self._resolve(path)
        if full is None:
            return False
        try:
            return full.stat().st_mtime > self._indexed_at
        except OSError:
            return False
