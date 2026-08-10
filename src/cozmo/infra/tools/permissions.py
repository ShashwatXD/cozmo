"""Workspace permissions - keep tools inside the project."""

from __future__ import annotations

from pathlib import Path

class PermissionError_(Exception):
    """Raised when a tool tries to leave the sandbox or use a disabled capability."""

class WorkspaceGuard:
    """Path and capability checks before tool side-effects."""

    def __init__(
        self,
        workdir: Path,
        *,
        allow_write: bool = True,
        allow_shell: bool = False,
    ) -> None:
        self.workdir = workdir.resolve()
        self.allow_write = allow_write
        self.allow_shell = allow_shell

    def resolve(self, path: str | Path) -> Path:
        """Resolve path relative to workdir; reject escapes (.., absolute outside)."""
        raw = Path(path)
        candidate = (self.workdir / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            candidate.relative_to(self.workdir)
        except ValueError as exc:
            raise PermissionError_(
                f"Path '{path}' is outside workdir {self.workdir}"
            ) from exc
        return candidate

    def require_write(self) -> None:
        if not self.allow_write:
            raise PermissionError_("Write tools disabled (COZMO_ALLOW_WRITE=false)")

    def require_shell(self) -> None:
        if not self.allow_shell:
            raise PermissionError_(
                "Shell tool disabled. Set COZMO_ALLOW_SHELL=true to enable."
            )
