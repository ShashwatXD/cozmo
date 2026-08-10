"""Small Cozmo activity animations for thinking / tool calls."""

from __future__ import annotations

from cozmo.cli import theme

_EYE_FRAMES: tuple[str, ...] = (
    "◉  ◉",
    "◎  ◉",
    "◉  ◎",
    "◡  ◡",
)

_LOGO_FRAMES: tuple[str, ...] = (
    "c····",
    "co···",
    "coz··",
    "cozm·",
    "cozmo",
    "·ozmo",
    "··zmo",
    "···mo",
)

_TOOL_LABELS: dict[str, str] = {
    "search_repo": "finding files",
    "semantic_search": "searching code",
    "read_file": "reading",
    "write_file": "writing",
    "symbol_search": "finding symbols",
    "find_references": "tracing refs",
    "get_codebase_graph": "mapping graph",
    "run_shell": "running shell",
    "git_status": "checking git",
    "git_diff": "diffing",
}


def _ensure_spinners() -> None:
    try:
        from rich.spinner import SPINNERS

        SPINNERS.setdefault(
            "cozmo_eyes",
            {"interval": 140, "frames": list(_EYE_FRAMES)},
        )
        SPINNERS.setdefault(
            "cozmo_logo",
            {"interval": 90, "frames": list(_LOGO_FRAMES)},
        )
    except Exception:
        pass


def tool_label(name: str) -> str:
    return _TOOL_LABELS.get(name, name.replace("_", " "))


class Activity:
    """Cyan Rich Status with Cozmo eye / logo spinners."""

    def __init__(self) -> None:
        self._status = None
        self._console = None

    def __enter__(self) -> "Activity":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._status is not None:
            return
        _ensure_spinners()
        try:
            from rich.console import Console
            from rich.status import Status

            self._console = Console(stderr=True)
            if not self._console.is_terminal:
                return
            self._status = Status(
                f"[{theme.CYAN}]thinking…[/]",
                console=self._console,
                spinner="cozmo_eyes",
                spinner_style=theme.CYAN_BOLD,
            )
            self._status.start()
        except Exception:
            self._status = None

    def stop(self) -> None:
        if self._status is None:
            return
        try:
            self._status.stop()
        except Exception:
            pass
        self._status = None

    def thinking(self) -> None:
        self._show("cozmo_eyes", "thinking")

    def tool(self, name: str) -> None:
        self._show("cozmo_logo", tool_label(name))

    def _show(self, spinner: str, label: str) -> None:
        if self._status is None:
            self.start()
        if self._status is None:
            return
        try:
            # Recreate status to swap spinner frames (Rich Status spinner is fixed at start)
            self.stop()
            from rich.status import Status

            assert self._console is not None
            self._status = Status(
                f"[{theme.CYAN}]{label}…[/]",
                console=self._console,
                spinner=spinner,
                spinner_style=theme.CYAN_BOLD,
            )
            self._status.start()
        except Exception:
            self._status = None
