"""Helpers to locate the on-disk RAG index for a workdir."""

from __future__ import annotations

from pathlib import Path

INDEX_NAME = "index.json"


def index_path(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo" / INDEX_NAME


def chroma_dir(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo" / "chroma"
