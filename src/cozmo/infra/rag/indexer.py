"""
Repository indexer - walk files → chunk → embed → store.

What: Indexer service used by `cozmo index`.
Why: one place builds the RAG index the agent searches.
Layer: app/infra boundary (app use-case style).
"""

from __future__ import annotations

from pathlib import Path

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.chunking import chunk_text
from cozmo.infra.rag.store import VectorStore

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cozmo", "dist", "build"}
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".dart",
    ".ts",
    ".js",
    ".rs",
    ".go",
}


class RepoIndexer:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def store(self) -> VectorStore:
        return self._store

    def index_dir(self, root: Path) -> int:
        """Index text files under root. Returns chunk count."""
        self._store.clear()
        root = root.resolve()
        chunks_total = 0
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            rel = str(path.relative_to(root))
            chunks = chunk_text(rel, text)
            if not chunks:
                continue
            embeddings = self._embedder.embed_many([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings, strict=True):
                self._store.add(chunk, emb)
            chunks_total += len(chunks)
        return chunks_total
