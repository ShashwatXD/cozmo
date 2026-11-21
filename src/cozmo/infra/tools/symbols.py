"""Python symbol index built from syntax trees, for exact "where is it / who calls it" answers.

Grep finds text and returns one line per hit, so the model needs another round to read the
code, and a fixed read window can stop mid-function. Parsing gives exact definitions with
their full line ranges and real call sites with the function they sit in.

The index lives in memory. It is built on first use (well under a second for a few hundred
files, no API calls) and each query re-parses only files whose size or modified time
changed, so answers are never stale. A file with a syntax error keeps its last good parse.
Only Python is parsed; other languages fall back to search_repo and semantic_search.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from cozmo.infra.workspace.ignore import IgnoreFilter

MAX_DEFINITION_LINES = 150
MAX_DEFINITIONS = 5
MAX_REFERENCES = 80


@dataclass(frozen=True)
class Symbol:
    name: str
    qualname: str  # "Class.method" for methods, the name otherwise
    kind: str  # "class" | "def" | "method"
    path: str
    start: int  # first line, decorators included
    end: int
    signature: str  # the def/class line as written


@dataclass(frozen=True)
class CallSite:
    name: str
    path: str
    line: int
    enclosing: str  # qualname of the function or class around the call, or "<module>"
    is_attribute: bool  # obj.name(...) — matched by name only, the type is unknown
    text: str


@dataclass
class _FileEntry:
    stamp: tuple[int, int]
    lines: list[str]
    symbols: list[Symbol] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    error: str | None = None
    # name bound at module level -> the import line that binds it ("from x import add")
    imports: dict[str, str] = field(default_factory=dict)


def _imports(tree: ast.Module, lines: list[str]) -> dict[str, str]:
    """Module-level import lines, keyed by the name each one binds."""
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            text = " ".join(
                part.strip() for part in lines[node.lineno - 1 : node.end_lineno or node.lineno]
            )
            for alias in node.names:
                found[(alias.asname or alias.name).split(".")[0]] = text
    return found


def _parse(rel: str, text: str) -> tuple[list[Symbol], list[CallSite]]:
    tree = ast.parse(text)
    lines = text.splitlines()
    symbols: list[Symbol] = []
    calls: list[CallSite] = []

    def visit(node: ast.AST, scope: list[tuple[str, str]]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                is_class = isinstance(child, ast.ClassDef)
                in_class = bool(scope) and scope[-1][1] == "class"
                kind = "class" if is_class else ("method" if in_class else "def")
                qual = ".".join([s[0] for s in scope] + [child.name])
                start = min([child.lineno] + [d.lineno for d in child.decorator_list])
                end = child.end_lineno or child.lineno
                sig = lines[child.lineno - 1].strip() if child.lineno <= len(lines) else child.name
                symbols.append(Symbol(child.name, qual, kind, rel, start, end, sig))
                visit(child, [*scope, (child.name, "class" if is_class else "def")])
                continue
            if isinstance(child, ast.Call):
                fn = child.func
                name = (
                    fn.id
                    if isinstance(fn, ast.Name)
                    else fn.attr
                    if isinstance(fn, ast.Attribute)
                    else None
                )
                if name:
                    line = child.lineno
                    calls.append(
                        CallSite(
                            name=name,
                            path=rel,
                            line=line,
                            enclosing=".".join(s[0] for s in scope) or "<module>",
                            is_attribute=isinstance(fn, ast.Attribute),
                            text=lines[line - 1].strip() if line <= len(lines) else "",
                        )
                    )
            visit(child, scope)

    visit(tree, [])
    return symbols, calls


class SymbolIndex:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._files: dict[str, _FileEntry] = {}

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self._root).as_posix()

    def _update(self, path: Path) -> None:
        rel = self._rel(path)
        try:
            st = path.stat()
        except OSError:
            self._files.pop(rel, None)
            return
        stamp = (st.st_mtime_ns, st.st_size)
        entry = self._files.get(rel)
        if entry is not None and entry.stamp == stamp:
            return
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            symbols, calls = _parse(rel, text)
            imports = _imports(ast.parse(text), text.splitlines())
        except SyntaxError as exc:
            # Mid-edit files often fail to parse; keep answering from the last good version.
            if entry is not None:
                entry.stamp, entry.error = stamp, f"syntax error at line {exc.lineno}"
            else:
                self._files[rel] = _FileEntry(stamp, [], error=f"syntax error at line {exc.lineno}")
            return
        except (OSError, ValueError):
            return
        self._files[rel] = _FileEntry(stamp, text.splitlines(), symbols, calls, imports=imports)

    def refresh(self) -> None:
        """Parse new or changed Python files and forget deleted ones."""
        present = IgnoreFilter(self._root).iter_files(suffixes=frozenset({".py"}))
        seen = set()
        for path in present:
            seen.add(self._rel(path))
            self._update(path)
        for rel in list(self._files):
            if rel not in seen:
                del self._files[rel]

    def definitions(self, name: str, *, path: str | None = None) -> list[Symbol]:
        self.refresh()
        out: list[Symbol] = []
        for rel, entry in self._files.items():
            if path and rel != path.lstrip("./"):
                continue
            out.extend(s for s in entry.symbols if name in (s.name, s.qualname))
        # Classes and top-level functions first, then by location.
        return sorted(out, key=lambda s: (s.kind == "method", s.path, s.start))

    def references(self, name: str) -> list[CallSite]:
        self.refresh()
        bare = name.rsplit(".", 1)[-1]
        out = [c for e in self._files.values() for c in e.calls if c.name == bare]
        return sorted(out, key=lambda c: (c.path, c.line))

    def outline(self, path: str) -> tuple[list[Symbol], int, str | None] | None:
        full = (self._root / path).resolve()
        if not full.is_relative_to(self._root) or not full.is_file():
            return None
        self._update(full)
        entry = self._files.get(self._rel(full))
        if entry is None:
            return None
        return sorted(entry.symbols, key=lambda s: s.start), len(entry.lines), entry.error

    def source(self, symbol: Symbol) -> list[str]:
        entry = self._files.get(symbol.path)
        return entry.lines[symbol.start - 1 : symbol.end] if entry else []

    def imports_used(self, symbol: Symbol) -> list[str]:
        """The file's import lines for names this symbol's body uses: where its helpers come from."""
        entry = self._files.get(symbol.path)
        if not entry or not entry.imports:
            return []
        words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", "\n".join(self.source(symbol))))
        return sorted({line for name, line in entry.imports.items() if name in words})


# ---- tool output -----------------------------------------------------------------------


def format_definitions(index: SymbolIndex, name: str, path: str | None = None) -> str:
    found = index.definitions(name, path=path)
    if not found:
        return (
            f"No Python definition named {name!r}. AST tools read .py files only: try "
            "search_repo for other languages or exact text, or semantic_search by meaning."
        )
    blocks: list[str] = []
    for sym in found[:MAX_DEFINITIONS]:
        body = index.source(sym)
        header = f"{sym.path}:{sym.start}-{sym.end}  {sym.kind} {sym.qualname}"
        used = index.imports_used(sym)
        if used:
            # Saves a read of the whole file just to learn where a helper comes from.
            header += "\n# imports it uses: " + "; ".join(used)
        if len(body) > MAX_DEFINITION_LINES:
            cut = sym.start + MAX_DEFINITION_LINES - 1
            body = [
                *body[:MAX_DEFINITION_LINES],
                f"...[{sym.kind} continues to line {sym.end}: read_file "
                f"start_line={cut + 1} end_line={sym.end} for the rest]",
            ]
        blocks.append(header + "\n" + "\n".join(body))
    if len(found) > MAX_DEFINITIONS:
        rest = ", ".join(f"{s.path}:{s.start} ({s.qualname})" for s in found[MAX_DEFINITIONS:])
        blocks.append(f"{len(found) - MAX_DEFINITIONS} more definition(s): {rest}")
    return "\n---\n".join(blocks)


def format_references(index: SymbolIndex, name: str, max_hits: int = MAX_REFERENCES) -> str:
    calls = index.references(name)
    if not calls:
        return (
            f"No calls to {name!r} in Python files. It may be used without being called "
            "(passed as a value, registered by string): try search_repo."
        )
    shown = calls[:max_hits]
    lines = [f"{c.path}:{c.line}  in {c.enclosing}  │ {c.text[:160]}" for c in shown]
    head = f"{len(calls)} call(s) to {name.rsplit('.', 1)[-1]}()"
    if len(calls) > len(shown):
        head += f", showing {len(shown)}; narrow with search_repo or raise max_hits"
    notes = []
    if any(c.is_attribute for c in shown):
        notes.append(
            "Calls written as obj.name(...) are matched by name only and may belong to "
            "another type with a method of the same name."
        )
    return "\n".join([head, *lines, *notes])


def format_outline(index: SymbolIndex, path: str) -> str:
    if not path.endswith(".py"):
        return f"outline reads Python files only; use read_file or search_repo for {path}."
    result = index.outline(path)
    if result is None:
        return f"Not a Python file in the workspace: {path}"
    symbols, total, error = result
    head = f"{path} ({total} lines)" + (f", {error}; showing the last good parse" if error else "")
    if not symbols:
        return f"{head}\n(no classes or functions)"
    rows = []
    for s in symbols:
        indent = "  " * s.qualname.count(".")
        rows.append(f"  {indent}{s.signature.rstrip(':')}   lines {s.start}-{s.end}")
    return "\n".join([head, *rows])
