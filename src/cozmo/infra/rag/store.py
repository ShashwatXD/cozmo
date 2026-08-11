"""In-memory / JSON vector store (default VectorStore backend)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from cozmo.domain.rag import Chunk, SearchHit


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class JsonVectorStore:
    """In-memory chunk embeddings with cosine top-k search; persists to JSON."""

    def __init__(self) -> None:
        self._items: list[tuple[Chunk, list[float]]] = []

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        self._items.append((chunk, embedding))

    def items(self) -> list[tuple[Chunk, list[float]]]:
        return list(self._items)

    def search(self, query_embedding: list[float], *, top_k: int = 5) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=chunk, score=cosine(query_embedding, emb))
            for chunk, emb in self._items
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"chunk": asdict(chunk), "embedding": emb} for chunk, emb in self._items
        ]
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


# Backward-compatible alias — prefer JsonVectorStore or domain VectorStore Protocol.
VectorStore = JsonVectorStore
