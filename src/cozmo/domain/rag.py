"""RAG types - chunks and search hits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One slice of a file ready to embed."""

    id: str
    path: str
    start_line: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RetrievalCandidate:
    """One retrieval hit before/after reranking."""

    path: str
    start_line: int
    text: str
    score: float = 0.0
    chunk_id: str = ""


@dataclass(frozen=True)
class ExpandedHit:
    """Reranked hit with surrounding source lines."""

    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    original_text: str = ""
