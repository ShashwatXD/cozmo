"""JsonVectorStore still works as VectorStore alias."""

from pathlib import Path

from cozmo.domain.rag import Chunk
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.store import JsonVectorStore, VectorStore, cosine


def test_json_vector_store_roundtrip(tmp_path: Path) -> None:
    emb = StubEmbedder()
    store = JsonVectorStore()
    chunk = Chunk(id="1", path="a.py", start_line=1, text="def foo():\n  return 1\n")
    store.add(chunk, emb.embed(chunk.text))
    path = tmp_path / "index.json"
    store.save(path)
    loaded = JsonVectorStore.load(path)
    assert len(loaded) == 1
    hits = loaded.search(emb.embed("foo"), top_k=1)
    assert hits and hits[0].chunk.path == "a.py"


def test_vector_store_alias() -> None:
    assert VectorStore is JsonVectorStore
    assert cosine([1.0, 0.0], [1.0, 0.0]) > 0.99
