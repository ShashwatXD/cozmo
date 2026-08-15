"""Tests for hybrid + semantic search over embeddings."""

from __future__ import annotations

from cozmo.domain.rag import Chunk
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.store import VectorStore
from cozmo.search.hybrid_search import HybridSearch
from cozmo.search.semantic_search import SemanticSearch


def test_semantic_search_ranks_similar_chunk() -> None:
    embedder = StubEmbedder(dim=64)
    store = VectorStore()
    store.add(
        Chunk(id="a", path="a.py", start_line=1, text="def hello(): return 'hi'"),
        embedder.embed("def hello(): return 'hi'"),
    )
    store.add(
        Chunk(id="b", path="math.py", start_line=1, text="return a - b off-by-one"),
        embedder.embed("return a - b off-by-one"),
    )
    hits = SemanticSearch(store, embedder).search("off-by-one subtract", top_k=2)
    assert hits
    assert hits[0].chunk.path == "math.py"


def test_hybrid_rrf_fuses_bm25_and_vector() -> None:
    embedder = StubEmbedder(dim=64)
    store = VectorStore()
    store.add(
        Chunk(id="c1", path="noise.py", start_line=1, text="unrelated billing total"),
        embedder.embed("unrelated billing total"),
    )
    store.add(
        Chunk(id="c2", path="math.py", start_line=3, text="def subtract: return a minus b"),
        embedder.embed("def subtract: return a minus b"),
    )
    hybrid = HybridSearch(store, embedder)
    hits = hybrid.search("subtract minus", top_k=2)
    assert hits
    assert hits[0].path == "math.py"
    assert hits[0].bm25_rank is not None or hits[0].vector_rank is not None
