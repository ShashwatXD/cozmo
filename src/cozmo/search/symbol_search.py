"""Symbol search – resolve queries to SymbolNode matches."""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import SymbolKind, SymbolNode

@dataclass(frozen=True)
class SymbolMatch:
    """A symbol matched by the search engine."""

    symbol: SymbolNode
    score: float
    match_type: str  # "exact" | "prefix" | "fuzzy"

class SymbolSearch:

    def __init__(self, index: CodeIndex) -> None:
        self._index = index

    def search(
        self,
        query: str,
        *,
        kind: SymbolKind | None = None,
        top_k: int = 10,
    ) -> list[SymbolMatch]:
        candidates = self._index.all_symbols
        if kind is not None:
            candidates = [s for s in candidates if s.kind == kind]

        matches: list[SymbolMatch] = []
        q = query.lower()

        for sym in candidates:
            name_l = sym.name.lower()
            qname_l = sym.qualified_name.lower()

            if q == name_l or q == qname_l:
                matches.append(SymbolMatch(sym, 1.0, "exact"))
                continue

            if name_l.startswith(q) or qname_l.startswith(q):
                ratio = len(q) / min(len(name_l), len(qname_l)) if name_l else 0.0
                matches.append(SymbolMatch(sym, 0.5 + 0.4 * ratio, "prefix"))
                continue

            best = max(
                difflib.SequenceMatcher(None, q, name_l).ratio(),
                difflib.SequenceMatcher(None, q, qname_l).ratio(),
            )
            if best >= 0.4:
                matches.append(SymbolMatch(sym, best * 0.8, "fuzzy"))

        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]
