"""Tests for the graphs module (dependency, call, import)."""

from __future__ import annotations

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import (
    FileSymbols,
    ImportInfo,
    Location,
    SymbolKind,
    SymbolNode,
    Visibility,
)
from cozmo.graphs.call_graph import CallGraph
from cozmo.graphs.dependency_graph import DependencyGraph
from cozmo.graphs.import_graph import ImportGraph


# ── helpers ──────────────────────────────────────────────────────────

def _make_index() -> CodeIndex:
    """Two-file index with cross-imports."""
    fs_a = FileSymbols(
        path="cozmo/alpha.py",
        symbols=(
            SymbolNode(
                name="greet",
                qualified_name="cozmo.alpha.greet",
                kind=SymbolKind.FUNCTION,
                location=Location("cozmo/alpha.py", 1, 3),
            ),
        ),
        imports=(
            ImportInfo(module="cozmo.beta", names=("helper",)),
            ImportInfo(module="os", names=("path",)),
        ),
        language="python",
    )
    fs_b = FileSymbols(
        path="cozmo/beta.py",
        symbols=(
            SymbolNode(
                name="helper",
                qualified_name="cozmo.beta.helper",
                kind=SymbolKind.FUNCTION,
                location=Location("cozmo/beta.py", 1, 5),
            ),
        ),
        imports=(
            ImportInfo(module="json"),
        ),
        language="python",
    )
    return CodeIndex(files={"cozmo/alpha.py": fs_a, "cozmo/beta.py": fs_b})


# ── DependencyGraph ─────────────────────────────────────────────────

class TestDependencyGraph:
    def test_forward_dependency(self) -> None:
        idx = _make_index()
        g = DependencyGraph()
        g.build(idx)
        assert "cozmo/beta.py" in g.dependencies("cozmo/alpha.py")

    def test_reverse_dependent(self) -> None:
        idx = _make_index()
        g = DependencyGraph()
        g.build(idx)
        assert "cozmo/alpha.py" in g.dependents("cozmo/beta.py")

    def test_external_import_not_resolved(self) -> None:
        idx = _make_index()
        g = DependencyGraph()
        g.build(idx)
        # os is external – should not appear in deps
        deps = g.dependencies("cozmo/alpha.py")
        assert all("os" not in d for d in deps)

    def test_to_dict(self) -> None:
        idx = _make_index()
        g = DependencyGraph()
        g.build(idx)
        d = g.to_dict()
        assert isinstance(d, dict)
        assert "cozmo/alpha.py" in d


# ── CallGraph ────────────────────────────────────────────────────────

class TestCallGraph:
    def test_simple_call(self) -> None:
        source_a = "def greet():\n    helper()\n    return 1\n"
        source_b = "def helper():\n    pass\n"
        idx = _make_index()
        cg = CallGraph()
        cg.build(idx, {"cozmo/alpha.py": source_a, "cozmo/beta.py": source_b})

        assert "cozmo.beta.helper" in cg.callees_of("cozmo.alpha.greet")

    def test_callers_of(self) -> None:
        source_a = "def greet():\n    helper()\n"
        source_b = "def helper():\n    pass\n"
        idx = _make_index()
        cg = CallGraph()
        cg.build(idx, {"cozmo/alpha.py": source_a, "cozmo/beta.py": source_b})

        assert "cozmo.alpha.greet" in cg.callers_of("cozmo.beta.helper")

    def test_to_dict(self) -> None:
        source_a = "def greet():\n    helper()\n"
        idx = _make_index()
        cg = CallGraph()
        cg.build(idx, {"cozmo/alpha.py": source_a})
        d = cg.to_dict()
        assert isinstance(d, dict)


# ── ImportGraph ──────────────────────────────────────────────────────

class TestImportGraph:
    def test_internal_classification(self) -> None:
        idx = _make_index()
        ig = ImportGraph()
        ig.build(idx)
        assert "cozmo.beta" in ig.internal_imports("cozmo/alpha.py")

    def test_external_classification(self) -> None:
        idx = _make_index()
        ig = ImportGraph()
        ig.build(idx)
        assert "os" in ig.external_imports("cozmo/alpha.py")
        assert "json" in ig.external_imports("cozmo/beta.py")

    def test_all_external_packages(self) -> None:
        idx = _make_index()
        ig = ImportGraph()
        ig.build(idx)
        pkgs = ig.all_external_packages()
        assert pkgs == {"os", "json"}

    def test_to_dict_structure(self) -> None:
        idx = _make_index()
        ig = ImportGraph()
        ig.build(idx)
        d = ig.to_dict()
        assert "internal" in d
        assert "external" in d
