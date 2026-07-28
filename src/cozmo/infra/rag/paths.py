"""Helpers to locate/load the on-disk RAG index for a workdir."""

from __future__ import annotations

from pathlib import Path

from cozmo.infra.rag import HashingEmbedder, VectorStore

INDEX_NAME = "index.json"


def index_path(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo" / INDEX_NAME


def load_store(workdir: Path) -> VectorStore:
    return VectorStore.load(index_path(workdir))


def default_embedder() -> HashingEmbedder:
    return HashingEmbedder()
