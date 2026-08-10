"""Code index types."""

from __future__ import annotations

from dataclasses import dataclass, field

from cozmo.domain.symbols import FileSymbols, SymbolNode

@dataclass
class CodeIndex:
    """Complete code index for a repository."""

    files: dict[str, FileSymbols] = field(default_factory=dict)

    @property
    def all_symbols(self) -> list[SymbolNode]:
        symbols: list[SymbolNode] = []
        for fs in self.files.values():
            symbols.extend(fs.symbols)
            for sym in fs.symbols:
                symbols.extend(sym.children)
        return symbols

    def symbols_in_file(self, path: str) -> tuple[SymbolNode, ...]:
        fs = self.files.get(path)
        return fs.symbols if fs else ()
