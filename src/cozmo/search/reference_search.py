"""Reference search – find all usages of a symbol across the codebase."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from cozmo.domain.index import CodeIndex

@dataclass(frozen=True)
class Reference:
    """A single usage site of a symbol."""

    path: str
    line: int
    line_text: str
    context: str  # surrounding context or node description

class ReferenceSearch:

    def __init__(self, index: CodeIndex, sources: dict[str, str]) -> None:
        self._index = index
        self._sources = sources

    def find_references(self, symbol_name: str) -> list[Reference]:
        refs: list[Reference] = []
        for path, source in self._sources.items():
            fs = self._index.files.get(path)
            lang = fs.language if fs else "unknown"
            if lang == "python":
                refs.extend(self._search_python(path, source, symbol_name))
            else:
                refs.extend(self._search_text(path, source, symbol_name))
        refs.sort(key=lambda r: (r.path, r.line))
        return refs

    def _search_python(
        self, path: str, source: str, name: str
    ) -> list[Reference]:
        results: list[Reference] = []
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return self._search_text(path, source, name)

        for node in ast.walk(tree):
            matched = False
            ctx = ""
            if isinstance(node, ast.Name) and node.id == name:
                matched = True
                ctx = f"Name reference"
            elif isinstance(node, ast.Attribute) and node.attr == name:
                matched = True
                ctx = f"Attribute access"
            elif isinstance(node, ast.FunctionDef) and node.name == name:
                matched = True
                ctx = "Function definition"
            elif isinstance(node, ast.ClassDef) and node.name == name:
                matched = True
                ctx = "Class definition"

            if matched and hasattr(node, "lineno"):
                lineno = node.lineno
                line_text = lines[lineno - 1] if lineno <= len(lines) else ""
                results.append(Reference(path, lineno, line_text.strip(), ctx))

        return results

    def _search_text(
        self, path: str, source: str, name: str
    ) -> list[Reference]:
        results: list[Reference] = []
        for i, line in enumerate(source.splitlines(), 1):
            if name in line:
                results.append(Reference(path, i, line.strip(), "Text match"))
        return results
