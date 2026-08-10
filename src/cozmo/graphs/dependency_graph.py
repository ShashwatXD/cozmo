"""File-to-file dependency graph built from import statements."""

from __future__ import annotations

from collections import defaultdict

from cozmo.domain.index import CodeIndex
from cozmo.domain.symbols import ImportInfo

class DependencyGraph:
    """Directed graph of file-level dependencies derived from imports."""

    def __init__(self) -> None:
        self._deps: dict[str, set[str]] = defaultdict(set)


    def build(self, index: CodeIndex) -> None:
        """Populate the graph from a :class:`CodeIndex`."""
        self._deps.clear()
        known_modules = self._module_map(index)

        for path, file_syms in index.files.items():
            # ensure every file appears even if it has no imports
            self._deps.setdefault(path, set())
            for imp in file_syms.imports:
                resolved = self._resolve(imp, known_modules)
                if resolved and resolved != path:
                    self._deps[path].add(resolved)


    def dependencies(self, path: str) -> set[str]:
        """Files that *path* depends on (forward edges)."""
        return set(self._deps.get(path, set()))

    def dependents(self, path: str) -> set[str]:
        """Files that depend on *path* (reverse edges)."""
        return {src for src, targets in self._deps.items() if path in targets}

    def to_dict(self) -> dict[str, list[str]]:
        """Serializable adjacency list."""
        return {k: sorted(v) for k, v in sorted(self._deps.items())}


    @staticmethod
    def _module_map(index: CodeIndex) -> dict[str, str]:
        """Map dotted module names to file paths in the index.

        ``cozmo/domain/tools.py`` ➜ ``cozmo.domain.tools``
        ``cozmo/domain/__init__.py`` ➜ ``cozmo.domain``
        """
        mapping: dict[str, str] = {}
        for path in index.files:
            mod = path.replace("/", ".").replace("\\", ".")
            if mod.endswith(".py"):
                mod = mod[:-3]
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            mapping[mod] = path
        return mapping

    @staticmethod
    def _resolve(imp: ImportInfo, module_map: dict[str, str]) -> str | None:
        """Resolve an :class:`ImportInfo` to a file path if known."""
        mod = imp.module
        if not mod:
            return None
        if mod in module_map:
            return module_map[mod]
        # ``from cozmo.domain.tools import X`` – module is the path
        # try progressively shorter prefixes (package __init__)
        parts = mod.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in module_map:
                return module_map[prefix]
        return None
