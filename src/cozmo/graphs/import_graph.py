"""Import graph separating internal vs external dependencies."""

from __future__ import annotations

from collections import defaultdict

from cozmo.domain.index import CodeIndex


class ImportGraph:
    """Per-file classification of imports as internal or external."""

    def __init__(self) -> None:
        self._internal: dict[str, set[str]] = defaultdict(set)
        self._external: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, index: CodeIndex) -> None:
        """Classify every import in the index."""
        self._internal.clear()
        self._external.clear()

        known_roots = self._known_roots(index)

        for path, file_syms in index.files.items():
            self._internal.setdefault(path, set())
            self._external.setdefault(path, set())
            for imp in file_syms.imports:
                mod = imp.module
                if not mod:
                    continue
                top = mod.split(".")[0]
                if top in known_roots or imp.is_relative:
                    self._internal[path].add(mod)
                else:
                    self._external[path].add(top)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def internal_imports(self, path: str) -> set[str]:
        """Internal (project) modules imported by *path*."""
        return set(self._internal.get(path, set()))

    def external_imports(self, path: str) -> set[str]:
        """External packages imported by *path*."""
        return set(self._external.get(path, set()))

    def all_external_packages(self) -> set[str]:
        """Union of all external top-level package names."""
        pkgs: set[str] = set()
        for s in self._external.values():
            pkgs |= s
        return pkgs

    def to_dict(self) -> dict:
        """Serializable representation."""
        return {
            "internal": {k: sorted(v) for k, v in sorted(self._internal.items())},
            "external": {k: sorted(v) for k, v in sorted(self._external.items())},
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _known_roots(index: CodeIndex) -> set[str]:
        """Derive top-level package names from file paths in the index."""
        roots: set[str] = set()
        for path in index.files:
            top = path.replace("\\", "/").split("/")[0]
            roots.add(top)
        return roots
