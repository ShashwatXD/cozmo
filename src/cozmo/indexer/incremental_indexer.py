"""Incremental indexer - SHA-256 change detection for files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

class IncrementalIndexer:
    """Detects changed files by comparing SHA-256 content hashes."""

    def __init__(self) -> None:
        self._hashes: dict[str, str] = {}

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def changed_files(self, root: Path, files: list[Path]) -> list[Path]:
        """Return files whose content hash differs from the stored hash."""
        changed: list[Path] = []
        for fpath in files:
            try:
                content = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = str(fpath.relative_to(root))
            new_hash = self._hash(content)
            if self._hashes.get(rel) != new_hash:
                changed.append(fpath)
        return changed

    def update_hash(self, path: Path, content: str) -> None:
        """Store hash for a file path (should be relative)."""
        self._hashes[str(path)] = self._hash(content)

    def save(self, path: Path) -> None:
        """Persist hashes to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._hashes), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Load hashes from JSON."""
        if path.exists():
            self._hashes = json.loads(path.read_text(encoding="utf-8"))
