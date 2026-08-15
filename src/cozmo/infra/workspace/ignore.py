"""Workspace path filtering: default skips + .gitignore."""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".cozmo",
        "dist",
        "build",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",
        "coverage",
        ".next",
        ".nuxt",
    }
)

_TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".txt",
        ".toml",
        ".yml",
        ".yaml",
        ".json",
        ".dart",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".css",
        ".scss",
        ".html",
        ".xml",
        ".sh",
        ".bash",
        ".zsh",
        ".sql",
        ".graphql",
        ".proto",
    }
)


def is_default_skipped(path: Path, *, root: Path | None = None) -> bool:
    """True if any path component is in DEFAULT_SKIP_DIRS."""
    parts = path.parts
    if root is not None:
        try:
            parts = path.resolve().relative_to(root.resolve()).parts
        except ValueError:
            parts = path.parts
    return any(part in DEFAULT_SKIP_DIRS for part in parts)


def is_indexable_suffix(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


class IgnoreFilter:
    """
    Combine hardcoded skip dirs with root .gitignore patterns.

    Supports a practical subset: blank/comment lines, !, **, *, ?, and
    directory patterns ending in /. Nested gitignores are not loaded
    (ripgrep still applies full gitignore when used for search).
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._rules: list[tuple[bool, re.Pattern[str], bool]] = []
        self._load_gitignore(self.root / ".gitignore")

    def _load_gitignore(self, path: Path) -> None:
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negate = line.startswith("!")
            if negate:
                line = line[1:]
            if not line:
                continue
            dir_only = line.endswith("/")
            if dir_only:
                line = line[:-1]
            try:
                pattern = self._glob_to_re(line)
            except re.error:
                continue
            self._rules.append((negate, pattern, dir_only))

    @staticmethod
    def _glob_to_re(glob: str) -> re.Pattern[str]:
        """Translate a gitignore-style glob to a regex matched on relative paths."""
        # Anchored to repo-relative path using / separators.
        i = 0
        out: list[str] = []
        if glob.startswith("/"):
            glob = glob[1:]
            out.append("^")
        elif not glob.startswith("**/"):
            # Match in any directory unless pattern has a slash.
            if "/" not in glob.rstrip("/"):
                out.append("(?:^|/)")
            else:
                out.append("^")
        while i < len(glob):
            c = glob[i]
            if glob.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if c == "*":
                out.append("[^/]*")
                i += 1
                continue
            if c == "?":
                out.append("[^/]")
                i += 1
                continue
            out.append(re.escape(c))
            i += 1
        out.append("(?:/.*)?$")
        return re.compile("".join(out))

    def ignored(self, path: Path) -> bool:
        """Return True if *path* should be excluded from walk/index/fallback search."""
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return True
        if is_default_skipped(rel):
            return True
        rel_s = rel.as_posix()
        ignored = False
        for negate, pattern, _dir_only in self._rules:
            if pattern.search(rel_s):
                ignored = not negate
        return ignored

    def iter_files(self, *, suffixes: frozenset[str] | None = None) -> list[Path]:
        """List files under root that pass ignore rules (and optional suffixes)."""
        found: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if self.ignored(path):
                continue
            if suffixes is not None and path.suffix.lower() not in suffixes:
                continue
            found.append(path)
        return found


# Public alias used by indexer
INDEXABLE_SUFFIXES = _TEXT_SUFFIXES
