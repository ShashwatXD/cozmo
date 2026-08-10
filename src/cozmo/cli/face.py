"""Cozmo splash — animated wordmark only (Hermes-style)."""

from __future__ import annotations

import sys
import time

from cozmo import __version__
from cozmo.cli import theme

# Block logo rows (painted with a cyan gradient during animation).
_LOGO_ROWS: tuple[str, ...] = (
    r"  ██████╗ ██████╗ ███████╗███╗   ███╗ ██████╗ ",
    r" ██╔════╝██╔═══██╗╚══███╔╝████╗ ████║██╔═══██╗",
    r" ██║     ██║   ██║  ███╔╝ ██╔████╔██║██║   ██║",
    r" ██║     ██║   ██║ ███╔╝  ██║╚██╔╝██║██║   ██║",
    r" ╚██████╗╚██████╔╝███████╗██║ ╚═╝ ██║╚██████╔╝",
    r"  ╚═════╝ ╚═════╝ ╚══════╝╚═╝     ╚═╝ ╚═════╝ ",
)

_ROW_STYLES: tuple[str, ...] = (
    theme.CYAN_BOLD,
    theme.CYAN_BOLD,
    theme.CYAN,
    theme.CYAN,
    theme.CYAN_DIM,
    theme.CYAN_DIM,
)


def render_face_plain() -> str:
    return "\n".join(_LOGO_ROWS)


def _paint_logo(console) -> None:
    from rich.text import Text

    for style, row in zip(_ROW_STYLES, _LOGO_ROWS, strict=True):
        console.print(Text(row, style=style))


def _animate_logo(console) -> None:
    """Reveal the wordmark row-by-row, then a quick cyan pulse."""
    from rich.text import Text

    # Row reveal
    for i, (style, row) in enumerate(zip(_ROW_STYLES, _LOGO_ROWS, strict=True)):
        console.print(Text(row, style=style))
        if console.is_terminal:
            time.sleep(0.045)

    if not console.is_terminal:
        return

    # Pulse: flash bold → settle back to gradient
    time.sleep(0.08)
    # Move cursor up over logo and redraw brighter, then final gradient
    n = len(_LOGO_ROWS)
    sys.stdout.write(f"\033[{n}A")
    sys.stdout.flush()
    for row in _LOGO_ROWS:
        console.print(Text(row, style=theme.CYAN_BOLD))
    time.sleep(0.12)
    sys.stdout.write(f"\033[{n}A")
    sys.stdout.flush()
    _paint_logo(console)


def print_face(*, clear: bool = True, animate: bool = True) -> None:
    try:
        from rich.console import Console
        from rich.text import Text

        console = Console()
        if clear and console.is_terminal:
            console.clear()

        console.print()
        if animate and console.is_terminal:
            _animate_logo(console)
        else:
            _paint_logo(console)

        console.print()
        meta = Text("  ")
        meta.append("coding agent", style=theme.CYAN)
        meta.append(f"  v{__version__}", style=theme.CYAN_DIM)
        console.print(meta)
        console.print(Text("  how can i help?  ·  /help  /exit", style=theme.MUTED))
        console.print()
    except Exception:
        import typer

        if clear:
            typer.echo("\033[2J\033[H", nl=False)
        typer.secho(render_face_plain(), fg="cyan")
        typer.secho(f"  coding agent  v{__version__}", fg="cyan")
        typer.secho("  how can i help?  ·  /help  /exit", fg="bright_black")
        typer.echo()
