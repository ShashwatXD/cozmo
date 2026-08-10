"""Hybrid search – BM25 + vector similarity via Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from cozmo.domain.index import CodeIndex
from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.store import VectorStore

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+")

@dataclass(frozen=True)
class HybridHit:
    """A result from hybrid search."""

    path: str
    start_line: int
    text: str
    score: float
    bm25_rank: int | None
    vector_rank: int | None

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]

class _BM25:
    """Minimal BM25 over a corpus of (key, text) pairs."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs: list[tuple[str, Counter[str]]] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def add(self, key: str, text: str) -> None:
        tfs = Counter(_tokenize(text))
        self._docs.append((key, tfs))
        for term in tfs:
            self._df[term] += 1

    def build(self) -> None:
        total = sum(sum(tfs.values()) for _, tfs in self._docs)
        self._avgdl = total / len(self._docs) if self._docs else 1.0

    def score(self, query: str) -> list[tuple[str, float]]:
        q_terms = _tokenize(query)
        n = len(self._docs)
        results: list[tuple[str, float]] = []
        for key, tfs in self._docs:
            dl = sum(tfs.values()) or 1
            s = 0.0
            for t in q_terms:
                if t not in tfs:
                    continue
                tf = tfs[t]
                df = self._df.get(t, 0)
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                num = tf * (self._k1 + 1)
                den = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                s += idf * num / den
            if s > 0:
                results.append((key, s))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

class HybridSearch:

    RRF_K = 60

    def __init__(
        self, store: VectorStore, embedder: Embedder, index: CodeIndex
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._index = index
        self._bm25 = _BM25()
        self._texts: dict[str, tuple[str, int, str]] = {}
        self._built = False

    def build(self) -> None:
        """Index symbols + vector-store chunks into BM25."""
        for path, fs in self._index.files.items():
            for sym in fs.symbols:
                key = sym.qualified_name
                text = f"{sym.name} {sym.docstring}"
                self._bm25.add(key, text)
                self._texts[key] = (
                    sym.location.path,
                    sym.location.start_line,
                    text,
                )
        for chunk, _emb in self._store.items():
            if chunk.id in self._texts:
                continue
            self._bm25.add(chunk.id, chunk.text)
            self._texts[chunk.id] = (chunk.path, chunk.start_line, chunk.text)
        self._bm25.build()
        self._built = True

    def search(self, query: str, *, top_k: int = 10) -> list[HybridHit]:
        if not self._built:
            self.build()

        recall_n = max(top_k * 5, 50)

        bm25_results = self._bm25.score(query)
        bm25_ranks: dict[str, int] = {
            key: rank + 1 for rank, (key, _) in enumerate(bm25_results[:recall_n])
        }

        # Vector retrieval — wide pool for later rerank
        q_emb = self._embedder.embed(query)
        vec_hits = self._store.search(q_emb, top_k=recall_n)
        vec_ranks: dict[str, int] = {
            h.chunk.id: rank + 1 for rank, h in enumerate(vec_hits)
        }

        all_keys = set(bm25_ranks) | set(vec_ranks)
        fused: list[tuple[str, float, int | None, int | None]] = []
        k = self.RRF_K
        for key in all_keys:
            score = 0.0
            br = bm25_ranks.get(key)
            vr = vec_ranks.get(key)
            if br is not None:
                score += 1.0 / (k + br)
            if vr is not None:
                score += 1.0 / (k + vr)
            fused.append((key, score, br, vr))

        fused.sort(key=lambda x: x[1], reverse=True)

        hits: list[HybridHit] = []
        for key, score, br, vr in fused[:top_k]:
            if key in self._texts:
                path, start_line, text = self._texts[key]
            else:
                # From vector store only – find chunk info
                for h in vec_hits:
                    if h.chunk.id == key:
                        path, start_line, text = h.chunk.path, h.chunk.start_line, h.chunk.text
                        break
                else:
                    continue
            hits.append(HybridHit(path, start_line, text, score, br, vr))
        return hits
