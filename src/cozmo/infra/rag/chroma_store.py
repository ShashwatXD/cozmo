"""Optional Chroma-backed VectorStore (install chromadb for ANN)."""

from __future__ import annotations

import json
from pathlib import Path

from cozmo.domain.rag import Chunk, SearchHit
from cozmo.infra.rag.store import JsonVectorStore


class ChromaVectorStore:
    """
    Local Chroma collection under path parent / chroma/.

    Falls back to JsonVectorStore behavior when chromadb is not installed
    only if constructed via factory with backend=json; this class requires chromadb.
    """

    def __init__(self, persist_dir: Path, *, collection: str = "cozmo") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "vector_backend=chroma requires chromadb. "
                'Install with: pip install "cozmo-agent[vector]"'
            ) from exc

        self._persist_dir = persist_dir
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        # Mirror for save()/items() compatibility with existing indexer code.
        self._mirror = JsonVectorStore()

    def __len__(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        self._mirror.clear()

    def add(self, chunk: Chunk, embedding: list[float]) -> None:
        self._collection.upsert(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[
                {
                    "path": chunk.path,
                    "start_line": chunk.start_line,
                    "text": chunk.text[:2000],
                }
            ],
        )
        self._mirror.add(chunk, embedding)

    def items(self) -> list[tuple[Chunk, list[float]]]:
        return self._mirror.items()

    def search(self, query_embedding: list[float], *, top_k: int = 5) -> list[SearchHit]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self._collection.count())),
        )
        hits: list[SearchHit] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for i, doc_id in enumerate(ids):
            meta = metas[i] or {}
            dist = float(dists[i]) if i < len(dists) else 1.0
            # Chroma cosine distance: lower is better → convert to similarity-ish score.
            score = 1.0 - dist
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        id=str(doc_id),
                        path=str(meta.get("path", "")),
                        start_line=int(meta.get("start_line", 1)),
                        text=str(docs[i] if i < len(docs) else meta.get("text", "")),
                    ),
                    score=score,
                )
            )
        return hits

    def save(self, path: Path) -> None:
        """Also write JSON mirror so non-chroma tools can still read index.json."""
        self._mirror.save(path)
        meta = path.parent / "chroma_meta.json"
        meta.write_text(
            json.dumps({"backend": "chroma", "persist_dir": str(self._persist_dir)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, *, persist_dir: Path | None = None) -> ChromaVectorStore:
        chroma_dir = persist_dir or (path.parent / "chroma")
        store = cls(chroma_dir)
        # If chroma empty but JSON exists, hydrate once.
        if store._collection.count() == 0 and path.exists():
            json_store = JsonVectorStore.load(path)
            for chunk, emb in json_store.items():
                store.add(chunk, emb)
        else:
            # Rebuild mirror from JSON if present for items()/save compatibility.
            if path.exists():
                store._mirror = JsonVectorStore.load(path)
        return store
