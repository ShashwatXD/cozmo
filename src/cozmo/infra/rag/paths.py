"""Helpers to locate/load the on-disk RAG index for a workdir."""

from __future__ import annotations

from pathlib import Path

from cozmo.domain.ports_rag import VectorStore
from cozmo.infra.rag.store import JsonVectorStore

INDEX_NAME = "index.json"


def index_path(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo" / INDEX_NAME


def chroma_dir(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo" / "chroma"


def load_store(workdir: Path) -> VectorStore:
    """Default JSON backend (CLI may use build_vector_store for chroma)."""
    return JsonVectorStore.load(index_path(workdir))
