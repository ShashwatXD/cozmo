"""
RAG types — chunks and search hits.

What: Chunk, SearchHit.
Why: one shape for indexer, store, and semantic_search tool.
Layer: domain.
Flutter: Freezed models for search results.
"""

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
