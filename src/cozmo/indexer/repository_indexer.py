"""
Repository indexer - walks workspace, parses symbols, builds CodeIndex.

What: Orchestrates file scanning, symbol extraction, and index persistence.
Why: Single entry point for building a complete code index of a repository.
Layer: indexer (app-level orchestration over domain + infra).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import FileSymbols
from cozmo.indexer.incremental_indexer import IncrementalIndexer
from cozmo.parser import PythonASTParser

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cozmo", "dist", "build"}

_SUPPORTED_SUFFIXES = {
    ".py", ".ts", ".js", ".rs", ".go", ".md", ".toml", ".yml", ".yaml", ".json", ".dart",
}

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".md": "markdown",
    ".toml": "toml",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".dart": "dart",
}


class RepositoryIndexer:
    """Walks a workspace and builds a CodeIndex of all symbols."""

    def __init__(self, incremental: IncrementalIndexer | None = None) -> None:
        self._incremental = incremental or IncrementalIndexer()
        self._parser = PythonASTParser()

    def index(self, root: Path) -> CodeIndex:
        """Scan *root*, extract symbols, return a CodeIndex."""
        root = root.resolve()
        code_index = CodeIndex()

        all_files = self._collect_files(root)

        # Incremental: only process changed files
        hashes_path = root / ".cozmo" / "file_hashes.json"
        self._incremental.load(hashes_path)
        changed = self._incremental.changed_files(root, all_files)

        for fpath in changed:
            rel = str(fpath.relative_to(root))
            lang = _LANGUAGE_MAP.get(fpath.suffix.lower(), "unknown")
            try:
                text = fpath.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            fs = self._parse_file(rel, text, lang)
            code_index.files[rel] = fs
            self._incremental.update_hash(Path(rel), text)

        self._incremental.save(hashes_path)
        self._save_index(root, code_index)
        return code_index

    def _collect_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            files.append(path)
        return files

    def _parse_file(self, rel_path: str, text: str, language: str) -> FileSymbols:
        """Parse a file into FileSymbols. Only Python gets full AST parsing."""
        if language == "python":
            return self._parser.parse(text, path=rel_path)
        # Non-Python: return a minimal FileSymbols (no symbol extraction yet)
        return FileSymbols(path=rel_path, language=language)

    def _save_index(self, root: Path, index: CodeIndex) -> None:
        out = root / ".cozmo" / "code_index.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            path: {
                "path": fs.path,
                "language": fs.language,
                "symbol_count": len(fs.symbols),
                "import_count": len(fs.imports),
            }
            for path, fs in index.files.items()
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
