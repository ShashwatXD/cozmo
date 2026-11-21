"""Search context is read from disk on demand, so edits made during a session show up."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cozmo.domain.rag import Chunk
from cozmo.domain.tools import ToolCall
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.live_sources import LiveSources
from cozmo.infra.rag.store import VectorStore
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.search.pipeline import RetrievalPipeline


def _touch_later(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    later = time.time() + 5
    os.utime(path, (later, later))


def test_reads_current_text_and_rereads_after_an_edit(tmp_path: Path) -> None:
    f = tmp_path / "math_utils.py"
    f.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    src = LiveSources(tmp_path)
    assert "a - b" in (src.get("math_utils.py") or "")
    _touch_later(f, "def add(a, b):\n    return a + b\n")
    assert "a + b" in (src.get("math_utils.py") or "")


def test_refuses_paths_outside_the_workspace(tmp_path: Path) -> None:
    (tmp_path.parent / "secret.txt").write_text("nope", encoding="utf-8")
    src = LiveSources(tmp_path)
    assert src.get("../secret.txt") is None
    assert src.get("missing.py", "fallback") == "fallback"


def test_flags_files_changed_after_indexing(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    src = LiveSources(tmp_path, indexed_at=time.time())
    assert not src.changed_since_index("a.py")
    _touch_later(f, "x = 2\n")
    assert src.changed_since_index("a.py")
    assert not LiveSources(tmp_path).changed_since_index("a.py"), "no index time, no flag"


def test_pipeline_expands_hits_with_the_edited_code(tmp_path: Path) -> None:
    f = tmp_path / "math_utils.py"
    f.write_text("# header\ndef add(a, b):\n    return a - b\n", encoding="utf-8")
    embedder = StubEmbedder(dim=64)
    store = VectorStore()
    store.add(
        Chunk(id="c1", path="math_utils.py", start_line=3, text="return a - b"),
        embedder.embed("return a - b"),
    )
    pipe = RetrievalPipeline(store, embedder, sources=LiveSources(tmp_path), top_k=1)
    _touch_later(f, "# header\ndef add(a, b):\n    return a + b  # fixed\n")
    hit = pipe.retrieve("a - b")[0]
    assert "# fixed" in hit.text, "expansion shows the file as it is now, not the indexed copy"


def test_semantic_search_marks_hits_from_changed_files(tmp_path: Path) -> None:
    f = tmp_path / "math_utils.py"
    f.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    embedder = StubEmbedder(dim=64)
    store = VectorStore()
    store.add(
        Chunk(id="c1", path="math_utils.py", start_line=2, text="return a - b"),
        embedder.embed("return a - b"),
    )
    sources = LiveSources(tmp_path, indexed_at=time.time())
    reg = build_default_registry(
        WorkspaceGuard(tmp_path), vector_store=store, embedder=embedder, sources=sources
    )

    def run() -> str:
        call = ToolCall(id="1", name="semantic_search", arguments=json.dumps({"query": "a - b"}))
        return ToolExecutor(reg).execute(call).content

    assert "changed since indexing" not in run()
    _touch_later(f, "def add(a, b):\n    return a + b\n")
    assert "changed since indexing" in run()
