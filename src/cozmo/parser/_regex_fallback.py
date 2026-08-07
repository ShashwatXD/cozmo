"""Regex-based fallback parser for non-Python languages."""

from __future__ import annotations

import re
from typing import Sequence

from cozmo.domain.symbols import (
    FileSymbols,
    ImportInfo,
    Location,
    SymbolKind,
    SymbolNode,
    Visibility,
)

# ---- Pattern sets per language -----------------------------------------

_JS_TS_PATTERNS: list[tuple[re.Pattern[str], SymbolKind]] = [
    (re.compile(r"^\s*(?:export\s+)?class\s+(\w+)"), SymbolKind.CLASS),
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"), SymbolKind.FUNCTION),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\("), SymbolKind.FUNCTION),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)"), SymbolKind.VARIABLE),
]

_JS_TS_IMPORT = re.compile(
    r"""^\s*import\s+(?:"""
    r"""(?:\{[^}]*\}\s+from\s+|"""
    r"""\*\s+as\s+\w+\s+from\s+|"""
    r"""\w+\s+from\s+)"""
    r""")['"]([^'"]+)['"]""",
)

_RUST_PATTERNS: list[tuple[re.Pattern[str], SymbolKind]] = [
    (re.compile(r"^\s*(?:pub(?:\(.*?\))?\s+)?struct\s+(\w+)"), SymbolKind.CLASS),
    (re.compile(r"^\s*(?:pub(?:\(.*?\))?\s+)?enum\s+(\w+)"), SymbolKind.CLASS),
    (re.compile(r"^\s*(?:pub(?:\(.*?\))?\s+)?(?:async\s+)?fn\s+(\w+)"), SymbolKind.FUNCTION),
    (re.compile(r"^\s*(?:pub(?:\(.*?\))?\s+)?trait\s+(\w+)"), SymbolKind.CLASS),
]

_RUST_IMPORT = re.compile(r"^\s*use\s+([\w:]+)")

_GO_PATTERNS: list[tuple[re.Pattern[str], SymbolKind]] = [
    (re.compile(r"^\s*type\s+(\w+)\s+struct\b"), SymbolKind.CLASS),
    (re.compile(r"^\s*type\s+(\w+)\s+interface\b"), SymbolKind.CLASS),
    (re.compile(r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\("), SymbolKind.FUNCTION),
]

_LANG_MAP: dict[str, tuple[list[tuple[re.Pattern[str], SymbolKind]], re.Pattern[str] | None]] = {
    "javascript": (_JS_TS_PATTERNS, _JS_TS_IMPORT),
    "typescript": (_JS_TS_PATTERNS, _JS_TS_IMPORT),
    "rust": (_RUST_PATTERNS, _RUST_IMPORT),
    "go": (_GO_PATTERNS, None),
}


def _visibility(name: str) -> Visibility:
    if name.startswith("_"):
        return Visibility.PRIVATE
    return Visibility.PUBLIC


def parse_with_regex(source: str, path: str, language: str) -> FileSymbols:
    """Extract symbols from *source* using regex heuristics."""
    sym_patterns, import_pat = _LANG_MAP.get(language, ([], None))

    symbols: list[SymbolNode] = []
    imports: list[ImportInfo] = []

    for lineno, line in enumerate(source.splitlines(), start=1):
        for pat, kind in sym_patterns:
            m = pat.match(line)
            if m:
                name = m.group(1)
                symbols.append(
                    SymbolNode(
                        name=name,
                        qualified_name=name,
                        kind=kind,
                        location=Location(path, lineno, lineno),
                        visibility=_visibility(name),
                    )
                )
                break  # first match wins for this line

        if import_pat:
            im = import_pat.match(line)
            if im:
                imports.append(
                    ImportInfo(
                        module=im.group(1),
                        location=Location(path, lineno, lineno),
                    )
                )

    return FileSymbols(
        path=path,
        symbols=tuple(symbols),
        imports=tuple(imports),
        language=language,
    )
