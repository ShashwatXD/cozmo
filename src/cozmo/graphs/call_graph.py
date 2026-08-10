"""Call graph built by walking Python ASTs."""

from __future__ import annotations

import ast
from collections import defaultdict

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import FileSymbols, SymbolKind

class CallGraph:
    """Caller → callee relationships extracted from function bodies."""

    def __init__(self) -> None:
        self._calls: dict[str, set[str]] = defaultdict(set)


    def build(self, index: CodeIndex, sources: dict[str, str]) -> None:
        """Parse every file's AST and record call edges."""
        self._calls.clear()
        name_table = self._build_name_table(index)

        for path, source in sources.items():
            file_syms = index.files.get(path)
            if file_syms is None or file_syms.language != "python":
                continue
            try:
                tree = ast.parse(source, filename=path)
            except SyntaxError:
                continue
            self._walk_tree(tree, file_syms, name_table)


    def callees_of(self, name: str) -> set[str]:
        """Qualified names called by *name*."""
        return set(self._calls.get(name, set()))

    def callers_of(self, name: str) -> set[str]:
        """Qualified names that call *name*."""
        return {caller for caller, callees in self._calls.items() if name in callees}

    def to_dict(self) -> dict[str, list[str]]:
        """Serializable adjacency list."""
        return {k: sorted(v) for k, v in sorted(self._calls.items())}


    @staticmethod
    def _build_name_table(index: CodeIndex) -> dict[str, str]:
        """Map simple function/method names to their qualified name.

        When multiple symbols share a simple name the last one wins,
        which is acceptable for a best-effort static call graph.
        """
        table: dict[str, str] = {}
        for fs in index.files.values():
            for sym in fs.symbols:
                if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    table[sym.name] = sym.qualified_name
                for child in sym.children:
                    if child.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                        table[child.name] = child.qualified_name
        return table

    def _walk_tree(
        self,
        tree: ast.Module,
        file_syms: FileSymbols,
        name_table: dict[str, str],
    ) -> None:
        """Walk AST and record call edges for every function/method."""
        func_ranges: list[tuple[range, str]] = []
        for sym in file_syms.symbols:
            if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                func_ranges.append(
                    (range(sym.location.start_line, sym.location.end_line + 1), sym.qualified_name)
                )
            for child in sym.children:
                if child.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD):
                    func_ranges.append(
                        (range(child.location.start_line, child.location.end_line + 1), child.qualified_name)
                    )

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee_name = self._call_name(node)
            if callee_name is None:
                continue

            resolved = name_table.get(callee_name, callee_name)

            lineno = getattr(node, "lineno", None)
            if lineno is None:
                continue
            caller_qn: str | None = None
            for rng, qn in func_ranges:
                if lineno in rng:
                    caller_qn = qn
                    break
            if caller_qn is None:
                # top-level call – attribute to module
                caller_qn = f"<module:{file_syms.path}>"

            if resolved != caller_qn:
                self._calls[caller_qn].add(resolved)

    @staticmethod
    def _call_name(node: ast.Call) -> str | None:
        """Extract the simple or dotted name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            # e.g. obj.method – return just the attribute name
            return func.attr
        return None
