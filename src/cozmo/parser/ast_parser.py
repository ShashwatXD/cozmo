"""Python AST-based parser for symbol extraction."""

from __future__ import annotations

import ast
from typing import Sequence

from cozmo.domain.symbols import (
    FileSymbols,
    ImportInfo,
    Location,
    SymbolKind,
    SymbolNode,
    Visibility,
)


def _visibility(name: str) -> Visibility:
    return Visibility.PRIVATE if name.startswith("_") else Visibility.PUBLIC


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a class/function/module node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
        return ast.get_docstring(node) or ""
    return ""


def _end_line(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)


class PythonASTParser:
    """Parses Python source into SymbolNode and ImportInfo objects."""

    def parse(self, source: str, path: str = "<string>") -> FileSymbols:
        tree = ast.parse(source, filename=path)
        symbols = self._visit_body(tree.body, path, prefix="")
        imports = self._collect_imports(tree.body, path)
        return FileSymbols(
            path=path,
            symbols=tuple(symbols),
            imports=tuple(imports),
            language="python",
        )

    # ------------------------------------------------------------------
    # Symbol extraction
    # ------------------------------------------------------------------

    def _visit_body(
        self,
        stmts: Sequence[ast.stmt],
        path: str,
        prefix: str,
    ) -> list[SymbolNode]:
        symbols: list[SymbolNode] = []
        for node in stmts:
            if isinstance(node, ast.ClassDef):
                symbols.append(self._visit_class(node, path, prefix))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._visit_function(node, path, prefix, is_method=False))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                symbols.extend(self._visit_assignment(node, path, prefix))
        return symbols

    def _visit_class(self, node: ast.ClassDef, path: str, prefix: str) -> SymbolNode:
        qname = f"{prefix}{node.name}" if prefix else node.name
        children: list[SymbolNode] = []
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                children.append(self._visit_function(child, path, f"{qname}.", is_method=True))
            elif isinstance(child, ast.ClassDef):
                children.append(self._visit_class(child, path, f"{qname}."))
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                children.extend(self._visit_assignment(child, path, f"{qname}."))
        return SymbolNode(
            name=node.name,
            qualified_name=qname,
            kind=SymbolKind.CLASS,
            location=Location(path, node.lineno, _end_line(node)),
            docstring=_get_docstring(node),
            visibility=_visibility(node.name),
            children=tuple(children),
        )

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        path: str,
        prefix: str,
        *,
        is_method: bool,
    ) -> SymbolNode:
        qname = f"{prefix}{node.name}" if prefix else node.name
        return SymbolNode(
            name=node.name,
            qualified_name=qname,
            kind=SymbolKind.METHOD if is_method else SymbolKind.FUNCTION,
            location=Location(path, node.lineno, _end_line(node)),
            docstring=_get_docstring(node),
            visibility=_visibility(node.name),
        )

    def _visit_assignment(
        self,
        node: ast.Assign | ast.AnnAssign,
        path: str,
        prefix: str,
    ) -> list[SymbolNode]:
        names: list[str] = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        return [
            SymbolNode(
                name=n,
                qualified_name=f"{prefix}{n}" if prefix else n,
                kind=SymbolKind.VARIABLE,
                location=Location(path, node.lineno, _end_line(node)),
                visibility=_visibility(n),
            )
            for n in names
        ]

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _collect_imports(
        self,
        stmts: Sequence[ast.stmt],
        path: str,
    ) -> list[ImportInfo]:
        imports: list[ImportInfo] = []
        for node in stmts:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        ImportInfo(
                            module=alias.name,
                            alias=alias.asname or "",
                            location=Location(path, node.lineno, _end_line(node)),
                        )
                    )
            elif isinstance(node, ast.ImportFrom):
                imports.append(
                    ImportInfo(
                        module=node.module or "",
                        names=tuple(a.name for a in node.names),
                        alias="",
                        is_relative=bool(node.level),
                        level=node.level or 0,
                        location=Location(path, node.lineno, _end_line(node)),
                    )
                )
        return imports
