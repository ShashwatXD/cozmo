"""
CLI View — terminal I/O only.

Commands:
  cozmo chat   — plain LLM (multi-turn if no -m)
  cozmo agent  — ReAct + tools (multi-turn if no -m)
  cozmo index  — build RAG index for a workdir

Layer: View. Flutter: Widget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cozmo import __version__
from cozmo.app.agent import AgentRunner
from cozmo.app.chat import ChatUseCase
from cozmo.domain.completion import Usage
from cozmo.domain.cost import format_cost_line
from cozmo.domain.memory import ConversationMemory
from cozmo.infra.llm.factory import build_llm
from cozmo.infra.rag import HashingEmbedder, RepoIndexer, VectorStore
from cozmo.infra.rag.paths import default_embedder, index_path, load_store
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.settings import Settings, load_settings

app = typer.Typer(
    name="cozmo",
    help="Cozmo — CLI coding assistant.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version"),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("chat")
def chat(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="One-shot message (omit for multi-turn REPL)",
    ),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    json_mode: bool = typer.Option(False, "--json"),
) -> None:
    """Plain chat (no tools). Multi-turn when -m is omitted."""
    settings = load_settings()
    llm = build_llm(settings)
    memory = ConversationMemory(max_messages=settings.memory_max_messages)
    use_case = ChatUseCase(llm, memory=memory, temperature=settings.temperature)

    if message is not None:
        _chat_once(
            use_case, message, stream=stream, json_mode=json_mode, settings=settings
        )
        return

    typer.secho(
        "chat REPL — memory on. /exit /clear",
        fg="bright_black",
    )
    while True:
        try:
            text = typer.prompt("you")
        except typer.Abort:
            break
        if text.strip() in {"/exit", "/quit", "exit", "quit"}:
            break
        if text.strip() == "/clear":
            memory.clear()
            typer.secho("memory cleared", fg="bright_black")
            continue
        _chat_once(use_case, text, stream=stream, json_mode=False, settings=settings)
        typer.secho(f"[memory messages={len(memory)}]", fg="bright_black")


@app.command("agent")
def agent(
    message: Optional[str] = typer.Option(
        None,
        "--message",
        "-m",
        help="One-shot task (omit for multi-turn REPL)",
    ),
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        "-w",
        help="Workspace root (default: COZMO_WORKDIR)",
    ),
) -> None:
    """Coding agent with tools + conversation memory + optional RAG."""
    settings = load_settings()
    root_dir = (workdir or settings.workdir).resolve()
    guard = WorkspaceGuard(
        root_dir,
        allow_write=settings.allow_write,
        allow_shell=settings.allow_shell,
    )
    embedder = default_embedder()
    store = load_store(root_dir)
    registry = build_default_registry(
        guard, vector_store=store, embedder=embedder
    )
    executor = ToolExecutor(registry)
    llm = build_llm(settings)
    memory = ConversationMemory(max_messages=settings.memory_max_messages)
    runner = AgentRunner(
        llm,
        registry,
        executor,
        memory=memory,
        temperature=settings.temperature,
        max_steps=settings.max_agent_steps,
    )

    typer.secho(f"workdir={root_dir}", fg="bright_black")
    if len(store) > 0:
        typer.secho(f"rag chunks={len(store)}", fg="bright_black")
    else:
        typer.secho(
            "rag: no index (optional: cozmo index -w ...)",
            fg="bright_black",
        )

    if message is not None:
        _agent_once(runner, message, settings)
        return

    typer.secho("agent REPL — memory on. /exit /clear", fg="bright_black")
    while True:
        try:
            text = typer.prompt("task")
        except typer.Abort:
            break
        if text.strip() in {"/exit", "/quit", "exit", "quit"}:
            break
        if text.strip() == "/clear":
            memory.clear()
            typer.secho("memory cleared", fg="bright_black")
            continue
        _agent_once(runner, text, settings)
        typer.secho(f"[memory messages={len(memory)}]", fg="bright_black")


@app.command("index")
def index_cmd(
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        "-w",
        help="Workspace to index (default: COZMO_WORKDIR)",
    ),
) -> None:
    """
    Build a local RAG index under <workdir>/.cozmo/index.json.

    Flutter: precompute a search index offline, then query at runtime.
    """
    settings = load_settings()
    root_dir = (workdir or settings.workdir).resolve()
    embedder = HashingEmbedder()
    store = VectorStore()
    indexer = RepoIndexer(embedder, store)
    n = indexer.index_dir(root_dir)
    out = index_path(root_dir)
    store.save(out)
    typer.echo(f"Indexed {n} chunks → {out}")


def _chat_once(
    use_case: ChatUseCase,
    text: str,
    *,
    stream: bool,
    json_mode: bool,
    settings: Settings,
) -> None:
    if json_mode:
        result = use_case.run(text, json_mode=True)
        typer.echo(result.content)
        _print_meter(settings.provider, settings.model, result.usage)
        return
    if stream:
        for chunk in use_case.stream(text):
            typer.echo(chunk, nl=False)
        typer.echo("")
        return
    result = use_case.run(text)
    typer.echo(result.content)
    _print_meter(settings.provider, settings.model, result.usage)


def _agent_once(runner: AgentRunner, text: str, settings: Settings) -> None:
    for event in runner.run_events(text):
        if event.kind == "tool_call":
            typer.secho(f"→ tool {event.tool_name}({event.text})", fg="cyan")
        elif event.kind == "tool_result":
            preview = event.text if len(event.text) < 400 else event.text[:400] + "…"
            typer.secho(f"← {event.tool_name}: {preview}", fg="bright_black")
        elif event.kind == "assistant":
            typer.echo(event.text)
        elif event.kind == "done" and event.text:
            typer.echo(event.text)
    result = runner.last_result
    if result:
        _print_meter(settings.provider, settings.model, result.usage)
        typer.secho(f"steps={result.steps}", fg="bright_black")


def _print_meter(provider: str, model: str, usage: Usage) -> None:
    line = format_cost_line(provider=provider, model=model, usage=usage)
    if line:
        typer.secho(f"\n{line}", fg="bright_black")


if __name__ == "__main__":
    app()
