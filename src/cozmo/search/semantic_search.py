"""Semantic search over a VectorStore."""

from __future__ import annotations

from cozmo.domain.ports_rag import Embedder
from cozmo.domain.rag import SearchHit
from cozmo.infra.rag.store import VectorStore

class SemanticSearch:

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        q_emb = self._embedder.embed(query)
        return self._store.search(q_emb, top_k=top_k)
