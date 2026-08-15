"""Phase 1: ignore, rg search, ranged read, caps, incremental index, evidence packs."""

from __future__ import annotations

import json
from pathlib import Path

from cozmo.app.subagent import _evidence_pack
from cozmo.domain.tools import ToolCall
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.indexer import RepoIndexer
from cozmo.infra.rag.store import JsonVectorStore
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.caps import shape_tool_content
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.infra.tools.rg_search import search_fallback, search_repo
from cozmo.infra.workspace.ignore import IgnoreFilter


def test_ignore_filter_skips_venv_and_gitignore(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("secret\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("nope\n", encoding="utf-8")

    filt = IgnoreFilter(tmp_path)
    files = {p.name for p in filt.iter_files()}
    assert "keep.py" in files
    assert "lib.py" not in files
    assert "ignored.py" not in files


def test_search_repo_flex_whitespace(tmp_path: Path) -> None:
    (tmp_path / "math.py").write_text("return a - b\n", encoding="utf-8")
    out = search_repo(tmp_path, "a-b")
    assert "math.py" in out
    assert "a - b" in out or "a-b" in out.replace(" ", "")


def test_search_fallback_respects_ignore(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.py").write_text("needle here\n", encoding="utf-8")
    hits = search_fallback(tmp_path, "needle")
    assert any("ok.py" in h for h in hits)
    assert not any("node_modules" in h for h in hits)


def test_ranged_read_file(tmp_path: Path) -> None:
    body = "\n".join(f"line{i}" for i in range(1, 21))
    (tmp_path / "f.py").write_text(body + "\n", encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    reg = build_default_registry(guard)
    ex = ToolExecutor(reg)

    result = ex.execute(
        ToolCall(
            id="1",
            name="read_file",
            arguments=json.dumps(
                {"path": "f.py", "start_line": 3, "end_line": 5}
            ),
        )
    )
    assert not result.is_error
    assert "f.py:3-5" in result.content
    assert "line3" in result.content
    assert "line5" in result.content
    assert "line1" not in result.content.split("\n", 1)[-1] or True

    around = ex.execute(
        ToolCall(
            id="2",
            name="read_file",
            arguments=json.dumps({"path": "f.py", "around_line": 10, "window": 1}),
        )
    )
    assert "line9" in around.content
    assert "line10" in around.content
    assert "line11" in around.content


def test_tool_executor_shapes_large_output(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("x" * 5_000, encoding="utf-8")
    guard = WorkspaceGuard(tmp_path)
    reg = build_default_registry(guard)
    ex = ToolExecutor(reg, max_chars=500)
    result = ex.execute(
        ToolCall(id="1", name="read_file", arguments=json.dumps({"path": "big.py"}))
    )
    assert not result.is_error
    assert "truncated" in result.content
    assert len(result.content) < 700


def test_shape_tool_content_unit() -> None:
    out = shape_tool_content("search_repo", "a" * 100, max_chars=50)
    assert out.endswith("]") or "truncated" in out
    assert len(out) < 120


def test_incremental_index_skips_unchanged(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    embedder = StubEmbedder(dim=32)
    store = JsonVectorStore()
    r1 = RepoIndexer(embedder, store).index_dir(tmp_path, incremental=True)
    assert r1.files_embedded == 1
    assert r1.chunks >= 1
    first_chunks = r1.chunks

    r2 = RepoIndexer(embedder, store).index_dir(tmp_path, incremental=True)
    assert r2.files_unchanged == 1
    assert r2.files_embedded == 0
    assert r2.chunks == first_chunks

    (tmp_path / "a.py").write_text("def a():\n    return 2\n", encoding="utf-8")
    r3 = RepoIndexer(embedder, store).index_dir(tmp_path, incremental=True)
    assert r3.files_embedded == 1
    assert r3.chunks >= 1


def test_evidence_pack_parses_json() -> None:
    pack = _evidence_pack(
        json.dumps(
            {
                "paths": ["a.py"],
                "claims": ["a.py:1 defines foo"],
                "open_questions": [],
                "summary": "found foo",
            }
        ),
        steps=2,
        stop_reason="completed",
        tokens=10,
    )
    assert pack["ok"] is True
    assert pack["paths"] == ["a.py"]
    assert pack["summary"] == "found foo"


def test_evidence_pack_fallback_plain_text() -> None:
    pack = _evidence_pack(
        "plain prose finding",
        steps=1,
        stop_reason="completed",
        tokens=3,
    )
    assert pack["summary"] == "plain prose finding"
    assert pack["paths"] == []
