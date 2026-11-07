"""Hybrid search: BM25 + vector similarity via Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
import threading
import weakref
from collections import Counter
from dataclasses import dataclass

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.store import VectorStore

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


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
    """
    Lowercased identifier tokens plus their sub-words.

    ``detect_stuck`` also yields ``detect`` and ``stuck``; ``WorkspaceGuard``
    also yields ``workspace`` and ``guard``, so natural-language queries can
    match code identifiers.
    """
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        out.append(low)
        parts = [p.lower() for piece in raw.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) > 1)
    return out


class _BM25:
    """
    Okapi BM25 over an inverted index.

    Query cost is proportional to the postings of the query terms, not to the
    corpus size, so it stays flat as the repository grows.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._keys: list[str] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 1.0

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, text: str) -> None:
        doc = len(self._keys)
        tfs = Counter(_tokenize(text))
        self._keys.append(key)
        self._lengths.append(sum(tfs.values()) or 1)
        for term, tf in tfs.items():
            self._postings.setdefault(term, []).append((doc, tf))

    def build(self) -> None:
        n = len(self._keys)
        self._avgdl = (sum(self._lengths) / n) if n else 1.0
        self._idf = {
            term: math.log((n - len(post) + 0.5) / (len(post) + 0.5) + 1.0)
            for term, post in self._postings.items()
        }

    def score(self, query: str) -> list[tuple[str, float]]:
        k1, b, avgdl = self._k1, self._b, self._avgdl
        acc: dict[int, float] = {}
        for term in set(_tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for doc, tf in postings:
                den = tf + k1 * (1 - b + b * self._lengths[doc] / avgdl)
                acc[doc] = acc.get(doc, 0.0) + idf * tf * (k1 + 1) / den
        ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
        return [(self._keys[doc], s) for doc, s in ranked]


@dataclass
class _LexicalIndex:
    version: object
    size: int
    bm25: _BM25
    texts: dict[str, tuple[str, int, str]]


# One BM25 index per live vector store, rebuilt only when the store mutates.
# The agent constructs a fresh RetrievalPipeline per semantic_search call, so
# without this cache every call re-tokenized the whole repository.
_INDEX_CACHE: weakref.WeakKeyDictionary[object, _LexicalIndex] = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.Lock()


def _store_signature(store: VectorStore) -> tuple[object, int]:
    return getattr(store, "version", None), len(store)


def _lexical_index(store: VectorStore) -> _LexicalIndex:
    version, size = _store_signature(store)
    with _CACHE_LOCK:
        try:
            cached = _INDEX_CACHE.get(store)
        except TypeError:  # store not weak-referenceable
            cached = None
        if cached and version is not None and cached.version == version and cached.size == size:
            return cached
    bm25 = _BM25()
    texts: dict[str, tuple[str, int, str]] = {}
    for chunk, _emb in store.items():
        bm25.add(chunk.id, chunk.text)
        texts[chunk.id] = (chunk.path, chunk.start_line, chunk.text)
    bm25.build()
    index = _LexicalIndex(version=version, size=size, bm25=bm25, texts=texts)
    with _CACHE_LOCK:
        try:
            _INDEX_CACHE[store] = index
        except TypeError:
            pass
    return index


class HybridSearch:
    """BM25 + vector RRF over embedded chunks (no AST / symbol index)."""

    RRF_K = 60

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder
        self._index: _LexicalIndex | None = None

    def build(self) -> None:
        """Index vector-store chunks into BM25 (cached per store version)."""
        self._index = _lexical_index(self._store)

    def _ensure_index(self) -> _LexicalIndex:
        version, size = _store_signature(self._store)
        idx = self._index
        if idx is None or idx.version != version or idx.size != size or version is None:
            self.build()
        assert self._index is not None
        return self._index

    def lexical_search(self, query: str, *, top_k: int = 10) -> list[HybridHit]:
        """BM25-only ranking."""
        idx = self._ensure_index()
        out: list[HybridHit] = []
        for rank, (key, score) in enumerate(idx.bm25.score(query)[:top_k], start=1):
            path, start_line, text = idx.texts[key]
            out.append(HybridHit(path, start_line, text, score, rank, None))
        return out

    def search(self, query: str, *, top_k: int = 10) -> list[HybridHit]:
        idx = self._ensure_index()
        recall_n = max(top_k * 5, 50)

        bm25_ranks: dict[str, int] = {
            key: rank + 1 for rank, (key, _) in enumerate(idx.bm25.score(query)[:recall_n])
        }

        q_emb = self._embedder.embed(query)
        vec_hits = self._store.search(q_emb, top_k=recall_n)
        vec_ranks: dict[str, int] = {h.chunk.id: rank + 1 for rank, h in enumerate(vec_hits)}
        vec_text = {h.chunk.id: (h.chunk.path, h.chunk.start_line, h.chunk.text) for h in vec_hits}

        k = self.RRF_K
        fused: list[tuple[str, float, int | None, int | None]] = []
        for key in bm25_ranks.keys() | vec_ranks.keys():
            br = bm25_ranks.get(key)
            vr = vec_ranks.get(key)
            score = (1.0 / (k + br) if br is not None else 0.0) + (
                1.0 / (k + vr) if vr is not None else 0.0
            )
            fused.append((key, score, br, vr))
        # Tie-break on key so ordering is deterministic across runs.
        fused.sort(key=lambda x: (-x[1], x[0]))

        hits: list[HybridHit] = []
        for key, score, br, vr in fused[:top_k]:
            meta = idx.texts.get(key) or vec_text.get(key)
            if meta is None:
                continue
            path, start_line, text = meta
            hits.append(HybridHit(path, start_line, text, score, br, vr))
        return hits
