"""High-level symbol extractor — dispatches to the right parser."""

from __future__ import annotations

from cozmo.domain.symbols import FileSymbols
from cozmo.parser.ast_parser import PythonASTParser
from cozmo.parser import tree_sitter
from cozmo.parser._regex_fallback import parse_with_regex

_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
}

_python_parser = PythonASTParser()


class SymbolExtractor:
    """Unified entry point for symbol extraction.

    Accepts a file path, source text, and optional language hint.
    Dispatches to PythonASTParser for Python, tree-sitter when
    available, and a regex fallback otherwise.
    """

    def extract(
        self,
        path: str,
        source: str,
        language: str | None = None,
    ) -> FileSymbols:
        lang = language or self._detect_language(path)

        if lang == "python":
            return _python_parser.parse(source, path)

        # Try tree-sitter first for non-Python languages
        ts_result = tree_sitter.parse_file(path, source, lang)
        if ts_result is not None:
            return ts_result

        # Regex fallback
        return parse_with_regex(source, path, lang)

    @staticmethod
    def _detect_language(path: str) -> str:
        for ext, lang in _EXT_TO_LANG.items():
            if path.endswith(ext):
                return lang
        return "unknown"
