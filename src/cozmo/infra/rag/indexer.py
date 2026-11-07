"""Repository indexer: walk files, chunk, embed, store (incremental + resilient)."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from cozmo.domain.ports_rag import Embedder
from cozmo.domain.rag import Chunk
from cozmo.infra.rag.chunking import chunk_text
from cozmo.infra.rag.store import VectorStore
from cozmo.infra.workspace.ignore import INDEXABLE_SUFFIXES, IgnoreFilter

logger = logging.getLogger(__name__)

_META_NAME = "rag_meta.json"
# Chunks per embedding request. Batches span files, so a repo of many small
# files costs ceil(chunks / _BATCH) requests instead of one or more per file.
_BATCH = 64


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
    embed_requests: int = 0


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
                if old_hashes.get(rel) == digest and rel not in {r for r, _, _ in to_embed}:
                    saved_hashes[rel] = digest

        for rel, digest, ok, err in self._embed_files(to_embed, report):
            if ok:
                report.files_embedded += 1
                saved_hashes[rel] = digest
            else:
                logger.warning("embed failed for %s: %s", rel, err)
                report.errors.append(f"{rel}: embed failed ({err})")
                report.partial = True

        report.chunks = len(self._store)
        self._save_hashes(meta_path, saved_hashes)
        return report

    def _embed_files(
        self,
        files: list[tuple[str, str, str]],
        report: IndexReport,
    ) -> list[tuple[str, str, bool, str]]:
        """
        Embed chunks of many files in cross-file batches.

        A file is committed to the store only after all of its chunks are
        embedded. If a batch fails, the files it touched are retried one by one
        so a single bad file cannot fail the others.
        """
        pending: list[tuple[str, str, list[Chunk]]] = []
        for rel, text, digest in files:
            pending.append((rel, digest, chunk_text(rel, text)))

        flat: list[tuple[int, Chunk]] = [
            (fi, c) for fi, (_, _, chunks) in enumerate(pending) for c in chunks
        ]
        vectors: dict[str, list[float]] = {}
        failed: dict[int, str] = {}

        for i in range(0, len(flat), _BATCH):
            batch = flat[i : i + _BATCH]
            try:
                embs = self._request(report, [c.text for _, c in batch])
                for (_, chunk), emb in zip(batch, embs, strict=True):
                    vectors[chunk.id] = emb
            except Exception as exc:  # noqa: BLE001 - isolate per file below
                logger.info("batch embed failed (%s); retrying files individually", exc)
                for fi in sorted({fi for fi, _ in batch}):
                    if fi in failed:
                        continue
                    _, _, chunks = pending[fi]
                    todo = [c for c in chunks if c.id not in vectors]
                    try:
                        for emb_chunk, emb in zip(
                            todo, self._request(report, [c.text for c in todo]), strict=True
                        ):
                            vectors[emb_chunk.id] = emb
                    except Exception as file_exc:  # noqa: BLE001
                        failed[fi] = str(file_exc)

        results: list[tuple[str, str, bool, str]] = []
        for fi, (rel, digest, chunks) in enumerate(pending):
            if fi in failed:
                results.append((rel, digest, False, failed[fi]))
                continue
            for chunk in chunks:
                self._store.add(chunk, vectors[chunk.id])
            results.append((rel, digest, True, ""))
        return results

    def _request(self, report: IndexReport, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        report.embed_requests += 1
        embeddings = self._embedder.embed_many(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(f"embed_many returned {len(embeddings)} for {len(texts)} chunks")
        if any(not e for e in embeddings):
            raise RuntimeError("empty embedding vector")
        return embeddings

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
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
