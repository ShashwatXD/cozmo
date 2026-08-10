"""Retrieval pipeline: hybrid recall → rerank → context expansion."""

from __future__ import annotations

from cozmo.domain.index import CodeIndex
from cozmo.domain.ports_rag import Embedder, Reranker
from cozmo.domain.rag import ExpandedHit, RetrievalCandidate
from cozmo.infra.rag.store import VectorStore
from cozmo.search.context_expand import expand_hits
from cozmo.search.hybrid_search import HybridSearch
from cozmo.search.rerank import LexicalReranker


class RetrievalPipeline:
    """Hybrid → top *candidate_k* → rerank top *top_k* → expand context."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        code_index: CodeIndex | None = None,
        sources: dict[str, str] | None = None,
        reranker: Reranker | None = None,
        candidate_k: int = 50,
        top_k: int = 10,
        expand_before: int = 12,
        expand_after: int = 12,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._code_index = code_index
        self._sources = sources or {}
        self._reranker = reranker or LexicalReranker()
        self._candidate_k = candidate_k
        self._top_k = top_k
        self._expand_before = expand_before
        self._expand_after = expand_after
        self._hybrid: HybridSearch | None = None
        if code_index is not None:
            self._hybrid = HybridSearch(store, embedder, code_index)

    def retrieve(
        self,
        query: str,
        *,
        candidate_k: int | None = None,
        top_k: int | None = None,
    ) -> list[ExpandedHit]:
        ck = candidate_k if candidate_k is not None else self._candidate_k
        tk = top_k if top_k is not None else self._top_k
        candidates = self._recall(query, candidate_k=ck)
        ranked = self._reranker.rerank(query, candidates, top_k=tk)
        return expand_hits(
            ranked,
            self._sources,
            before=self._expand_before,
            after=self._expand_after,
        )

    def _recall(self, query: str, *, candidate_k: int) -> list[RetrievalCandidate]:
        if self._hybrid is not None:
            hits = self._hybrid.search(query, top_k=candidate_k)
            return [
                RetrievalCandidate(
                    path=h.path,
                    start_line=h.start_line,
                    text=h.text,
                    score=h.score,
                )
                for h in hits
            ]
        q_emb = self._embedder.embed(query)
        vec = self._store.search(q_emb, top_k=candidate_k)
        return [
            RetrievalCandidate(
                path=h.chunk.path,
                start_line=h.chunk.start_line,
                text=h.chunk.text,
                score=h.score,
                chunk_id=h.chunk.id,
            )
            for h in vec
        ]
