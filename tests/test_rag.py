"""RAG chunking + indexing + search (no network)."""

from pathlib import Path

from cozmo.infra.rag.chunking import chunk_text
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.indexer import RepoIndexer
from cozmo.infra.rag.store import VectorStore, cosine
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.domain.tools import ToolCall
import json


def test_chunk_text() -> None:
    chunks = chunk_text("a.py", "line1\nline2\nline3\n", chunk_size=20, overlap=5)
    assert len(chunks) >= 1
    assert chunks[0].path == "a.py"


def test_cosine_identical() -> None:
    v = [1.0, 0.0, 0.0]
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_index_and_search(tmp_path: Path) -> None:
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b + 1  # off-by-one\n",
        encoding="utf-8",
    )
    embedder = StubEmbedder()
    store = VectorStore()
    n = RepoIndexer(embedder, store).index_dir(tmp_path).chunks
    assert n >= 1
    hits = store.search(embedder.embed("off-by-one add function"), top_k=3)
    assert hits
    assert "math_utils" in hits[0].chunk.path


def test_semantic_tool(tmp_path: Path) -> None:
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b + 1  # off-by-one bug\n",
        encoding="utf-8",
    )
    embedder = StubEmbedder()
    store = VectorStore()
    RepoIndexer(embedder, store).index_dir(tmp_path)
    guard = WorkspaceGuard(tmp_path)
    reg = build_default_registry(guard, vector_store=store, embedder=embedder)
    ex = ToolExecutor(reg)
    result = ex.execute(
        ToolCall(
            id="1",
            name="semantic_search",
            arguments=json.dumps({"query": "off-by-one bug in add"}),
        )
    )
    assert not result.is_error
    assert "math_utils" in result.content
