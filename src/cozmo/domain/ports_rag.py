"""Embedding + reranking ports for RAG."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from cozmo.domain.rag import RetrievalCandidate


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
