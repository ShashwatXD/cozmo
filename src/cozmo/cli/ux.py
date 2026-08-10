"""CLI banner, selects, and status output."""

from __future__ import annotations

from pathlib import Path

import typer

from cozmo import __version__

_SELECT_STYLE = None

def _style():
    global _SELECT_STYLE
    if _SELECT_STYLE is None:
        from questionary import Style

        _SELECT_STYLE = Style(
            [
                ("qmark", "fg:cyan bold"),
                ("question", "bold"),
                ("answer", "fg:cyan bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan bold"),
                ("selected", "fg:cyan"),
                ("instruction", "fg:#808080"),
            ]
        )
    return _SELECT_STYLE

def select(
    prompt: str,
    choices: list[str] | list[tuple[str, str]],
    default: str | None = None,
) -> str:
    import questionary
    from questionary import Choice

    q_choices: list[Choice] = []
    values: list[str] = []
    for item in choices:
        if isinstance(item, tuple):
            value, title = item
            q_choices.append(Choice(title=title, value=value))
            values.append(value)
        else:
            q_choices.append(Choice(title=item, value=item))
            values.append(item)
    default = default if default in values else values[0]
    result = questionary.select(
        prompt,
        choices=q_choices,
        default=default,
        style=_style(),
        use_indicator=True,
        use_shortcuts=False,
        instruction="(↑/↓ · Enter)",
    ).ask()
    if result is None:
        raise typer.Exit(code=1)
    return str(result)

def confirm(prompt: str, default: bool = False) -> bool:
    import questionary

    result = questionary.confirm(prompt, default=default, style=_style()).ask()
    if result is None:
        raise typer.Exit(code=1)
    return bool(result)

def prompt_secret(prompt: str, *, required: bool = False, existing: str = "") -> str | None:
    import questionary

    while True:
        message = prompt
        if existing:
            message = f"{prompt} (Enter keeps existing key)"
        raw = questionary.password(message, style=_style()).ask()
        if raw is None:
            raise typer.Exit(code=1)
        raw = raw.strip()
        if raw:
            return raw
        if existing:
            return existing
        if not required:
            return None
        typer.secho("  ✗ API key required — try again", fg="red")

def print_banner() -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console(stderr=False)
        title = Text()
        title.append("cozmo", style="bold cyan")
        title.append(f"  v{__version__}", style="dim")
        body = Text("coding agent · understands your repo\n", style="white")
        body.append("type a task · /help for commands · /exit to quit", style="dim")
        console.print(Panel(body, title=title, border_style="cyan", padding=(0, 2)))
    except Exception:
        typer.secho(f"cozmo v{__version__}", fg="cyan", bold=True)
        typer.secho("coding agent · /help · /exit", fg="bright_black")

def print_session_status(
    *,
    provider: str,
    model: str,
    workdir: Path,
    rag_chunks: int,
    allow_write: bool,
    allow_shell: bool,
) -> None:
    try:
        from rich.console import Console

        console = Console()
        bits = [
            f"[cyan]{provider}[/]/{model}",
            f"[dim]{workdir}[/]",
        ]
        if rag_chunks:
            bits.append(f"[dim]rag:{rag_chunks}[/]")
        else:
            bits.append("[yellow]rag:off[/]")
        flags = []
        if allow_write:
            flags.append("write")
        if allow_shell:
            flags.append("shell")
        if flags:
            bits.append("[dim]" + "+".join(flags) + "[/]")
        console.print(" · ".join(bits))
    except Exception:
        typer.secho(
            f"{provider}/{model} · {workdir}",
            fg="bright_black",
        )

def print_ok(msg: str) -> None:
    typer.secho(f"  ✓ {msg}", fg="green")

def print_err(msg: str) -> None:
    typer.secho(f"  ✗ {msg}", fg="red")

def print_dim(msg: str) -> None:
    typer.secho(msg, fg="bright_black")

def print_setup_summary(path: Path, data: dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("provider", str(data.get("provider")))
        table.add_row("model", str(data.get("model")))
        table.add_row("api_key", "set" if data.get("api_key") else "—")
        if data.get("base_url"):
            table.add_row("base_url", str(data.get("base_url")))
        table.add_row("embeddings", "auto (from provider)")
        table.add_row("config", str(path))
        console.print()
        console.print("[bold green]Ready.[/] Launch with [cyan]cozmo[/]")
        console.print(table)
    except Exception:
        print_ok(f"saved {path}")
        typer.secho("Run: cozmo", fg="cyan")
