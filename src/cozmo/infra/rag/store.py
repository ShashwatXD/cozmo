"""In-memory / JSON vector store (default VectorStore backend)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cozmo.domain.rag import Chunk, SearchHit


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class JsonVectorStore:
    """
    In-memory chunk embeddings with exact cosine top-k; persists to JSON.

    Search runs as one matrix-vector product over a cached, row-normalized
    float32 matrix that is rebuilt lazily after any mutation.
    """

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []
        self._matrix: np.ndarray | None = None
        self.version = 0  # bumped on every mutation; lets callers cache derived indexes

    def __len__(self) -> int:
        return len(self._items)

    def _touch(self) -> None:
        self._matrix = None
        self.version += 1

    def clear(self) -> None:
        self._items.clear()
        self._touch()

    def drop_paths(self, paths: set[str]) -> int:
        """Remove all chunks whose path is in *paths*. Returns removed count."""
        if not paths:
            return 0
        before = len(self._items)
        self._items = [(c, e) for c, e in self._items if c.path not in paths]
        removed = before - len(self._items)
        if removed:
            self._touch()
        return removed

    def paths(self) -> set[str]:
        return {c.path for c, _ in self._items}

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        self._items.append((chunk, embedding))
        self._touch()

    def items(self) -> list[tuple[Chunk, list[float]]]:
        return list(self._items)

    def _normalized_matrix(self) -> np.ndarray | None:
        if self._matrix is None and self._items:
            dims = {len(e) for _, e in self._items}
            if len(dims) != 1:
                return None  # mixed dimensions (model switch mid-index): use slow path
            mat = np.asarray([e for _, e in self._items], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._matrix = mat / norms
        return self._matrix

    def search(self, query_embedding: list[float], *, top_k: int = 5) -> list[SearchHit]:
        if not self._items or top_k <= 0:
            return []
        mat = self._normalized_matrix()
        if mat is None or mat.shape[1] != len(query_embedding):
            scored = [
                SearchHit(chunk=chunk, score=cosine(query_embedding, emb))
                for chunk, emb in self._items
            ]
            scored.sort(key=lambda h: h.score, reverse=True)
            return scored[:top_k]
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q)) or 1.0
        sims = mat @ (q / q_norm)
        k = min(top_k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [SearchHit(chunk=self._items[i][0], score=float(sims[i])) for i in top]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"chunk": asdict(chunk), "embedding": emb} for chunk, emb in self._items]
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> JsonVectorStore:
        store = cls()
        if not path.exists():
            return store
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw:
            c = row["chunk"]
            store.add(
                Chunk(
                    id=c["id"],
                    path=c["path"],
                    start_line=c["start_line"],
                    text=c["text"],
                ),
                row["embedding"],
            )
        return store


# Prefer JsonVectorStore or the domain VectorStore Protocol.
VectorStore = JsonVectorStore
