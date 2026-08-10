"""Tests for the search package – symbol, reference, hybrid, and semantic search."""

from __future__ import annotations

import pytest

from cozmo.domain.index import CodeIndex
from cozmo.domain.rag import Chunk
from cozmo.domain.symbols import (
    FileSymbols,
    Location,
    SymbolKind,
    SymbolNode,
    Visibility,
)
from cozmo.infra.rag.embedder import HashingEmbedder
from cozmo.infra.rag.store import VectorStore
from cozmo.search.hybrid_search import HybridSearch
from cozmo.search.reference_search import ReferenceSearch
from cozmo.search.semantic_search import SemanticSearch
from cozmo.search.symbol_search import SymbolSearch




def _make_symbol(
    name: str,
    qname: str,
    kind: SymbolKind = SymbolKind.FUNCTION,
    path: str = "mod.py",
    start: int = 1,
    end: int = 5,
    doc: str = "",
) -> SymbolNode:
    return SymbolNode(
        name=name,
        qualified_name=qname,
        kind=kind,
        location=Location(path, start, end),
        docstring=doc,
        visibility=Visibility.PUBLIC,
    )


@pytest.fixture()
def index() -> CodeIndex:
    syms = (
        _make_symbol("calculate_total", "billing.calculate_total", doc="Sum line items"),
        _make_symbol("User", "models.User", SymbolKind.CLASS, "models.py", 10, 30, doc="User model"),
        _make_symbol("calculate_tax", "billing.calculate_tax", doc="Compute tax"),
        _make_symbol("send_email", "notify.send_email", doc="Send email notification"),
    )
    fs = FileSymbols(path="mod.py", symbols=syms, language="python")
    return CodeIndex(files={"mod.py": fs})


SOURCES: dict[str, str] = {
    "main.py": (
        "from billing import calculate_total\n"
        "\n"
        "def run():\n"
        "    total = calculate_total(items)\n"
        "    print(total)\n"
    ),
    "readme.txt": "Use calculate_total to get the sum.\n",
}




class TestSymbolSearch:
    def test_exact_match(self, index: CodeIndex) -> None:
        ss = SymbolSearch(index)
        results = ss.search("calculate_total")
        assert results
        assert results[0].match_type == "exact"
        assert results[0].symbol.name == "calculate_total"
        assert results[0].score == 1.0

    def test_prefix_match(self, index: CodeIndex) -> None:
        ss = SymbolSearch(index)
        results = ss.search("calc")
        prefix = [r for r in results if r.match_type == "prefix"]
        assert len(prefix) == 2
        names = {r.symbol.name for r in prefix}
        assert names == {"calculate_total", "calculate_tax"}

    def test_fuzzy_match(self, index: CodeIndex) -> None:
        ss = SymbolSearch(index)
        results = ss.search("calcualte_totl")  # typo
        assert results
        assert results[0].symbol.name == "calculate_total"

    def test_kind_filter(self, index: CodeIndex) -> None:
        ss = SymbolSearch(index)
        results = ss.search("User", kind=SymbolKind.CLASS)
        assert all(r.symbol.kind == SymbolKind.CLASS for r in results)

    def test_top_k(self, index: CodeIndex) -> None:
        ss = SymbolSearch(index)
        results = ss.search("c", top_k=2)
        assert len(results) <= 2




class TestReferenceSearch:
    def test_python_references(self, index: CodeIndex) -> None:
        rs = ReferenceSearch(index, SOURCES)
        refs = rs.find_references("calculate_total")
        py_refs = [r for r in refs if r.path == "main.py"]
        assert len(py_refs) >= 2  # import + usage

    def test_text_fallback(self, index: CodeIndex) -> None:
        rs = ReferenceSearch(index, SOURCES)
        refs = rs.find_references("calculate_total")
        txt_refs = [r for r in refs if r.path == "readme.txt"]
        assert len(txt_refs) == 1
        assert txt_refs[0].context == "Text match"




class TestHybridSearch:
    def test_rrf_fusion(self, index: CodeIndex) -> None:
        embedder = HashingEmbedder(dim=64)
        store = VectorStore()

        # Add chunks matching the symbols
        for sym in index.all_symbols:
            text = f"{sym.name} {sym.docstring}"
            chunk = Chunk(id=sym.qualified_name, path=sym.location.path, start_line=sym.location.start_line, text=text)
            store.add(chunk, embedder.embed(text))

        hs = HybridSearch(store, embedder, index)
        results = hs.search("calculate total")
        assert results
        # Top hit should be calculate_total (both BM25 and vector agree)
        assert "calculate" in results[0].path or "calculate" in results[0].text

    def test_has_both_ranks(self, index: CodeIndex) -> None:
        embedder = HashingEmbedder(dim=64)
        store = VectorStore()
        for sym in index.all_symbols:
            text = f"{sym.name} {sym.docstring}"
            chunk = Chunk(id=sym.qualified_name, path=sym.location.path, start_line=sym.location.start_line, text=text)
            store.add(chunk, embedder.embed(text))

        hs = HybridSearch(store, embedder, index)
        results = hs.search("calculate_total")
        # At least one result should have both ranks
        both = [r for r in results if r.bm25_rank is not None and r.vector_rank is not None]
        assert both




class TestSemanticSearch:
    def test_returns_hits(self) -> None:
        embedder = HashingEmbedder(dim=64)
        store = VectorStore()
        store.add(
            Chunk(id="c1", path="a.py", start_line=1, text="calculate the total price"),
            embedder.embed("calculate the total price"),
        )
        store.add(
            Chunk(id="c2", path="b.py", start_line=1, text="send email notification"),
            embedder.embed("send email notification"),
        )

        ss = SemanticSearch(store, embedder)
        hits = ss.search("total price calculation")
        assert hits
        assert hits[0].chunk.id == "c1"

    def test_top_k(self) -> None:
        embedder = HashingEmbedder(dim=64)
        store = VectorStore()
        for i in range(10):
            store.add(
                Chunk(id=f"c{i}", path="f.py", start_line=i, text=f"symbol_{i}"),
                embedder.embed(f"symbol_{i}"),
            )

        ss = SemanticSearch(store, embedder)
        hits = ss.search("symbol", top_k=3)
        assert len(hits) == 3
