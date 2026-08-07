"""Optional tree-sitter bridge with graceful fallback.

tree-sitter is NOT a required dependency — we fall back to regex
when it is unavailable.
"""

from __future__ import annotations

from cozmo.domain.symbols import FileSymbols


def is_available() -> bool:
    """Return True if tree-sitter is installed and usable."""
    try:
        import tree_sitter  # noqa: F401
        return True
    except ImportError:
        return False


def parse_file(path: str, text: str, language: str) -> FileSymbols | None:
    """Parse *text* with tree-sitter. Returns None when unavailable."""
    if not is_available():
        return None
    # Full tree-sitter implementation deferred; return None for now
    # so callers fall through to the regex / AST path.
    return None
