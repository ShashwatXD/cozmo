"""Semantic search – thin wrapper around VectorStore for convenience."""

from __future__ import annotations

from cozmo.domain.ports_rag import Embedder
from cozmo.domain.rag import SearchHit
from cozmo.infra.rag.store import VectorStore


class SemanticSearch:
    """Embed a query and search the vector store.

    What: convenience API over VectorStore + Embedder.
    Why: single call instead of embed-then-search.
    Layer: search.
    """

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        q_emb = self._embedder.embed(query)
        return self._store.search(q_emb, top_k=top_k)
