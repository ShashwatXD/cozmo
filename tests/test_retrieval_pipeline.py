"""Tests for retrieval pipeline: recall → rerank → context expand."""

from __future__ import annotations

from cozmo.domain.rag import Chunk, RetrievalCandidate
from cozmo.infra.rag.embedder import HashingEmbedder
from cozmo.infra.rag.store import VectorStore
from cozmo.search.context_expand import expand_hits
from cozmo.search.pipeline import RetrievalPipeline
from cozmo.search.rerank import LexicalReranker


def test_lexical_reranker_prefers_whitespace_insensitive_match() -> None:
    reranker = LexicalReranker()
    candidates = [
        RetrievalCandidate(
            path="noise.py",
            start_line=1,
            text="completely unrelated billing total",
            score=0.9,
        ),
        RetrievalCandidate(
            path="math_utils.py",
            start_line=3,
            text="    return a - b",
            score=0.1,
        ),
    ]
    ranked = reranker.rerank("a-b", candidates, top_k=2)
    assert ranked[0].path == "math_utils.py"
    assert ranked[0].score > ranked[1].score


def test_context_expand_surrounding_lines() -> None:
    sources = {
        "math_utils.py": (
            "# bug: off-by-one on purpose for agent demos\n"
            "def add(a, b):\n"
            "    return a - b\n"
        )
    }
    hits = [
        RetrievalCandidate(
            path="math_utils.py",
            start_line=3,
            text="    return a - b",
            score=1.0,
        )
    ]
    expanded = expand_hits(hits, sources, before=2, after=0)
    assert len(expanded) == 1
    assert expanded[0].start_line == 1
    assert "def add" in expanded[0].text
    assert "a - b" in expanded[0].text


def test_pipeline_vector_only_reranks_and_expands() -> None:
    embedder = HashingEmbedder(dim=64)
    store = VectorStore()
    sources = {
        "math_utils.py": (
            "# header\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "# footer\n"
        ),
        "other.py": "def hello():\n    return 'hi'\n",
    }
    store.add(
        Chunk(id="c1", path="other.py", start_line=1, text="def hello(): return hi"),
        embedder.embed("def hello(): return hi"),
    )
    store.add(
        Chunk(id="c2", path="math_utils.py", start_line=3, text="return a - b"),
        embedder.embed("return a - b"),
    )

    pipe = RetrievalPipeline(
        store,
        embedder,
        sources=sources,
        candidate_k=10,
        top_k=2,
        expand_before=2,
        expand_after=1,
    )
    hits = pipe.retrieve("a-b")
    assert hits
    assert hits[0].path == "math_utils.py"
    assert "def add" in hits[0].text or "a - b" in hits[0].text
