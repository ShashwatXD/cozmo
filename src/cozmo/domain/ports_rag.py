"""Embedding + reranking + vector store ports for RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from cozmo.domain.rag import Chunk, RetrievalCandidate, SearchHit


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Score query–document pairs and return the best subset."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_k: int = 10,
    ) -> list[RetrievalCandidate]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Chunk embeddings with top-k search (JSON or ANN backend in infra)."""

    def __len__(self) -> int: ...

    def clear(self) -> None: ...

    def add(self, chunk: Chunk, embedding: list[float]) -> None: ...

    def items(self) -> list[tuple[Chunk, list[float]]]: ...

    def search(self, query_embedding: list[float], *, top_k: int = 5) -> list[SearchHit]: ...

    def save(self, path: Path) -> None: ...
