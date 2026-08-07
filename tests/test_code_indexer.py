"""Tests for the code indexer package.

Runs offline with no external dependencies. Uses pytest tmp_path fixture.
"""

from __future__ import annotations

from pathlib import Path

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import (
    FileSymbols,
    Location,
    SymbolKind,
    SymbolNode,
    Visibility,
)
from cozmo.indexer.incremental_indexer import IncrementalIndexer
from cozmo.indexer.repository_indexer import RepositoryIndexer


# ---------------------------------------------------------------------------
# IncrementalIndexer
# ---------------------------------------------------------------------------


class TestIncrementalIndexer:
    def test_detects_new_files(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("print('hi')", encoding="utf-8")

        idx = IncrementalIndexer()
        changed = idx.changed_files(tmp_path, [f])
        assert changed == [f]

    def test_unchanged_file_not_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        content = "print('hi')"
        f.write_text(content, encoding="utf-8")

        idx = IncrementalIndexer()
        idx.update_hash(Path("hello.py"), content)

        changed = idx.changed_files(tmp_path, [f])
        assert changed == []

    def test_changed_file_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.py"
        f.write_text("v1", encoding="utf-8")

        idx = IncrementalIndexer()
        idx.update_hash(Path("hello.py"), "v1")

        f.write_text("v2", encoding="utf-8")
        changed = idx.changed_files(tmp_path, [f])
        assert changed == [f]

    def test_save_and_load(self, tmp_path: Path) -> None:
        idx = IncrementalIndexer()
        idx.update_hash(Path("a.py"), "content_a")

        hashes_file = tmp_path / "hashes.json"
        idx.save(hashes_file)

        idx2 = IncrementalIndexer()
        idx2.load(hashes_file)

        f = tmp_path / "a.py"
        f.write_text("content_a", encoding="utf-8")
        assert idx2.changed_files(tmp_path, [f]) == []


# ---------------------------------------------------------------------------
# RepositoryIndexer
# ---------------------------------------------------------------------------


class TestRepositoryIndexer:
    def test_indexes_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "mod.py").write_text(
            "def greet():\n    '''Say hi.'''\n    print('hi')\n",
            encoding="utf-8",
        )
        (tmp_path / "util.py").write_text(
            "class Helper:\n    pass\n",
            encoding="utf-8",
        )
        # Should be skipped (unsupported ext)
        (tmp_path / "data.bin").write_bytes(b"\x00\x01")

        indexer = RepositoryIndexer()
        code_index = indexer.index(tmp_path)

        assert "mod.py" in code_index.files
        assert "util.py" in code_index.files
        assert "data.bin" not in code_index.files

        # Symbols extracted
        mod_syms = code_index.symbols_in_file("mod.py")
        assert any(s.name == "greet" for s in mod_syms)

    def test_skips_excluded_dirs(self, tmp_path: Path) -> None:
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("x = 1", encoding="utf-8")

        indexer = RepositoryIndexer()
        code_index = indexer.index(tmp_path)
        assert len(code_index.files) == 0

    def test_saves_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

        indexer = RepositoryIndexer()
        indexer.index(tmp_path)

        meta = tmp_path / ".cozmo" / "code_index.json"
        assert meta.exists()


# ---------------------------------------------------------------------------
# CodeIndex
# ---------------------------------------------------------------------------


class TestCodeIndex:
    def _make_symbol(self, name: str, kind: SymbolKind = SymbolKind.FUNCTION) -> SymbolNode:
        return SymbolNode(
            name=name,
            qualified_name=name,
            kind=kind,
            location=Location("test.py", 1, 5),
        )

    def test_all_symbols_includes_children(self) -> None:
        child = self._make_symbol("method", SymbolKind.METHOD)
        parent = SymbolNode(
            name="MyClass",
            qualified_name="MyClass",
            kind=SymbolKind.CLASS,
            location=Location("test.py", 1, 20),
            children=(child,),
        )
        fs = FileSymbols(path="test.py", symbols=(parent,), language="python")
        idx = CodeIndex(files={"test.py": fs})

        all_syms = idx.all_symbols
        names = [s.name for s in all_syms]
        assert "MyClass" in names
        assert "method" in names

    def test_symbols_in_file_missing(self) -> None:
        idx = CodeIndex()
        assert idx.symbols_in_file("nope.py") == ()
