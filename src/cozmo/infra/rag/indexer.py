"""Repository indexer: walk files, chunk, embed, store (incremental + resilient)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.chunking import chunk_text
from cozmo.infra.rag.store import VectorStore
from cozmo.infra.workspace.ignore import INDEXABLE_SUFFIXES, IgnoreFilter

logger = logging.getLogger(__name__)

_META_NAME = "rag_meta.json"
_BATCH = 32


@dataclass
class IndexReport:
    """Outcome of an index run (partial success allowed)."""

    chunks: int = 0
    files_seen: int = 0
    files_embedded: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    errors: list[str] = field(default_factory=list)
    partial: bool = False


class RepoIndexer:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def store(self) -> VectorStore:
        return self._store

    def index_dir(self, root: Path, *, incremental: bool = True) -> IndexReport:
        """
        Index text files under root.

        Incremental mode re-embeds only changed or new files and drops deleted
        paths. Per-file embed failures are recorded; other files still succeed.
        """
        root = root.resolve()
        report = IndexReport()
        filt = IgnoreFilter(root)
        meta_path = root / ".cozmo" / _META_NAME
        old_hashes = self._load_hashes(meta_path) if incremental else {}

        file_data: list[tuple[str, str, str]] = []
        for path in filt.iter_files(suffixes=INDEXABLE_SUFFIXES):
            rel = str(path.relative_to(root))
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                report.errors.append(f"{rel}: read failed ({exc})")
                report.partial = True
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            file_data.append((rel, text, digest))

        report.files_seen = len(file_data)
        current = {rel for rel, _, _ in file_data}
        store_empty = len(self._store) == 0
        do_full = (not incremental) or (not old_hashes) or store_empty

        if do_full:
            self._store.clear()
            to_embed = file_data
            report.files_unchanged = 0
        else:
            removed = set(old_hashes) - current
            if removed and hasattr(self._store, "drop_paths"):
                report.files_removed = int(self._store.drop_paths(removed))
            to_embed = [
                (rel, text, digest)
                for rel, text, digest in file_data
                if old_hashes.get(rel) != digest
            ]
            report.files_unchanged = report.files_seen - len(to_embed)
            changed_paths = {rel for rel, _, _ in to_embed}
            if changed_paths and hasattr(self._store, "drop_paths"):
                self._store.drop_paths(changed_paths)

        saved_hashes: dict[str, str] = {}
        if not do_full:
            for rel, _, digest in file_data:
                if old_hashes.get(rel) == digest and rel not in {
                    r for r, _, _ in to_embed
                }:
                    saved_hashes[rel] = digest

        for rel, text, digest in to_embed:
            try:
                self._embed_file(rel, text)
                report.files_embedded += 1
                saved_hashes[rel] = digest
            except Exception as exc:  # noqa: BLE001 - continue other files
                logger.warning("embed failed for %s: %s", rel, exc)
                report.errors.append(f"{rel}: embed failed ({exc})")
                report.partial = True

        report.chunks = len(self._store)
        self._save_hashes(meta_path, saved_hashes)
        return report

    def _embed_file(self, rel: str, text: str) -> int:
        chunks = chunk_text(rel, text)
        if not chunks:
            return 0
        total = 0
        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i : i + _BATCH]
            embeddings = self._embedder.embed_many([c.text for c in batch])
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"embed_many returned {len(embeddings)} for {len(batch)} chunks"
                )
            for chunk, emb in zip(batch, embeddings, strict=True):
                if not emb:
                    raise RuntimeError("empty embedding vector")
                self._store.add(chunk, emb)
            total += len(batch)
        return total

    @staticmethod
    def _load_hashes(path: Path) -> dict[str, str]:
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        files = raw.get("files") if isinstance(raw, dict) else None
        if not isinstance(files, dict):
            return {}
        return {str(k): str(v) for k, v in files.items()}

    @staticmethod
    def _save_hashes(path: Path, hashes: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "files": hashes}
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
