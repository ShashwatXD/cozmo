"""Cozmo splash — ASCII mascot + cyan wordmark.

Mascot art lives in mascot_ascii.py (outline density art of the
headphones / cube-head / skateboard mascot).
"""

from __future__ import annotations

import sys
import time

from cozmo import __version__
from cozmo.cli import theme
from cozmo.cli.mascot_ascii import MASCOT_ASCII

_BOT_H = len(MASCOT_ASCII)

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
    return "\n".join((*MASCOT_ASCII, "", *_LOGO_ROWS))


def _cursor_up(n: int) -> None:
    sys.stdout.write(f"\033[{n}A")
    sys.stdout.flush()


def _char_style(ch: str) -> str:
    """Density → brand color (open eyes stay dark, face cyan, cans purple)."""
    if ch in " .":
        return theme.MUTED
    if ch in ":":
        return theme.MUTED
    if ch in "-=":
        return theme.SHELL_DIM
    if ch in "+":
        return theme.SHELL_DIM
    if ch in "*":
        return theme.HEADPHONE
    if ch in "#":
        return theme.CYAN
    if ch == "%":
        return theme.CYAN
    if ch == "@":
        return theme.CYAN_BOLD
    return theme.CYAN


def _paint_mascot(console) -> None:
    from rich.text import Text

    for row in MASCOT_ASCII:
        text = Text()
        for ch in row:
            text.append(ch, style=_char_style(ch))
        console.print(text)


def _animate_mascot(console) -> None:
    """Reveal top → bottom, then a quick cyan pulse redraw."""
    from rich.text import Text

    for i, row in enumerate(MASCOT_ASCII):
        text = Text()
        for ch in row:
            text.append(ch, style=_char_style(ch))
        console.print(text)
        if i < 8 or i % 3 == 0:
            time.sleep(0.008)

    time.sleep(0.12)
    # pulse: redraw all bold cyan, then settle
    _cursor_up(_BOT_H)
    for row in MASCOT_ASCII:
        text = Text()
        for ch in row:
            if ch in ". ":
                text.append(ch, style=theme.MUTED)
            else:
                text.append(ch, style=theme.CYAN_BOLD)
        console.print(text)
    time.sleep(0.10)
    _cursor_up(_BOT_H)
    _paint_mascot(console)


def _paint_logo(console) -> None:
    from rich.text import Text

    for style, row in zip(_ROW_STYLES, _LOGO_ROWS, strict=True):
        console.print(Text(row, style=style))


def _animate_logo(console) -> None:
    from rich.text import Text

    for style, row in zip(_ROW_STYLES, _LOGO_ROWS, strict=True):
        console.print(Text(row, style=style))
        time.sleep(0.04)
    time.sleep(0.06)
    n = len(_LOGO_ROWS)
    _cursor_up(n)
    for row in _LOGO_ROWS:
        console.print(Text(row, style=theme.CYAN_BOLD))
    time.sleep(0.10)
    _cursor_up(n)
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
            _animate_mascot(console)
            console.print()
            _animate_logo(console)
        else:
            _paint_mascot(console)
            console.print()
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
