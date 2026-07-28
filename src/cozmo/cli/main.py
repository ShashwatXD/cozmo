"""
Cozmo CLI - chat, agent, index, eval.

Commands:
  cozmo chat   - plain LLM (multi-turn if no -m)
  cozmo agent  - ReAct coding agent with tools + RAG + memory
  cozmo index  - build RAG index for a workdir
  cozmo eval   - run golden-task regression suite
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cozmo import __version__
from cozmo.app.agent import AgentRunner
from cozmo.app.chat import ChatUseCase
from cozmo.app.eval_runner import run_eval
from cozmo.domain.completion import Usage
from cozmo.domain.cost import format_cost_line
from cozmo.domain.memory import ConversationMemory
from cozmo.infra.llm.factory import build_llm
from cozmo.infra.rag import RepoIndexer, VectorStore
from cozmo.infra.rag.factory import build_embedder
from cozmo.infra.rag.paths import index_path, load_store
from cozmo.infra.telemetry.tracer import Tracer
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.settings import Settings, load_settings

app = typer.Typer(
    name="cozmo",
    help="Cozmo - production-style CLI coding agent.",
    no_args_is_help=True,
)


def _apply_provider(
    settings: Settings,
    provider: Optional[str],
    model: Optional[str],
) -> Settings:
    data = settings.model_dump()
    if provider:
        data["provider"] = provider
    if model:
        data["model"] = model
    return Settings(**data)


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
    message: Optional[str] = typer.Option(None, "--message", "-m"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    json_mode: bool = typer.Option(False, "--json"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="stub | openai | ollama (overrides .env)"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model id override"),
) -> None:
    """Plain chat (no tools). Multi-turn when -m is omitted."""
    settings = _apply_provider(load_settings(), provider, model)
    llm = build_llm(settings)
    memory = ConversationMemory(max_messages=settings.memory_max_messages)
    use_case = ChatUseCase(llm, memory=memory, temperature=settings.temperature)
    typer.secho(f"provider={settings.provider} model={settings.model}", fg="bright_black")

    if message is not None:
        _chat_once(
            use_case, message, stream=stream, json_mode=json_mode, settings=settings
        )
        return

    typer.secho("chat REPL - /exit /clear", fg="bright_black")
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
    message: Optional[str] = typer.Option(None, "--message", "-m"),
    workdir: Optional[Path] = typer.Option(None, "--workdir", "-w"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="stub | openai | ollama (overrides .env)"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model id override"),
) -> None:
    """Coding agent: ReAct + tools + memory + RAG + traces."""
    settings = _apply_provider(load_settings(), provider, model)
    root_dir = (workdir or settings.workdir).resolve()
    guard = WorkspaceGuard(
        root_dir,
        allow_write=settings.allow_write,
        allow_shell=settings.allow_shell,
    )
    embedder = build_embedder(settings)
    store = load_store(root_dir)
    registry = build_default_registry(guard, vector_store=store, embedder=embedder)
    executor = ToolExecutor(registry)
    llm = build_llm(settings)
    memory = ConversationMemory(max_messages=settings.memory_max_messages)
    trace_path = root_dir / ".cozmo" / "traces.jsonl"
    tracer = Tracer(trace_path if settings.trace_enabled else None, enabled=settings.trace_enabled)
    runner = AgentRunner(
        llm,
        registry,
        executor,
        memory=memory,
        tracer=tracer,
        temperature=settings.temperature,
        max_steps=settings.max_agent_steps,
    )

    typer.secho(
        f"provider={settings.provider} model={settings.model} workdir={root_dir}",
        fg="bright_black",
    )
    if len(store) > 0:
        typer.secho(
            f"rag chunks={len(store)} embedder={settings.embedder}",
            fg="bright_black",
        )
    else:
        typer.secho("rag: no index (cozmo index -w ...)", fg="bright_black")
    if settings.trace_enabled:
        typer.secho(f"traces → {trace_path}", fg="bright_black")

    if message is not None:
        _agent_once(runner, message, settings)
        return

    typer.secho("agent REPL - /exit /clear", fg="bright_black")
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
    workdir: Optional[Path] = typer.Option(None, "--workdir", "-w"),
    embedder_name: Optional[str] = typer.Option(
        None,
        "--embedder",
        help="hash | openai | ollama (overrides COZMO_EMBEDDER)",
    ),
) -> None:
    """Build RAG index at <workdir>/.cozmo/index.json."""
    settings = load_settings()
    if embedder_name:
        data = settings.model_dump()
        data["embedder"] = embedder_name
        settings = Settings(**data)
    root_dir = (workdir or settings.workdir).resolve()
    embedder = build_embedder(settings)
    store = VectorStore()
    n = RepoIndexer(embedder, store).index_dir(root_dir)
    out = index_path(root_dir)
    store.save(out)
    typer.echo(
        f"Indexed {n} chunks with embedder={settings.embedder} → {out}"
    )


@app.command("eval")
def eval_cmd(
    workdir: Optional[Path] = typer.Option(
        None,
        "--workdir",
        "-w",
        help="Fixture repo (default: tests/fixtures/tiny_repo)",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Use configured LLM provider instead of stub scripts",
    ),
) -> None:
    """Run golden-task eval suite (CI-safe stub by default)."""
    settings = load_settings()
    root = workdir or Path("tests/fixtures/tiny_repo")
    root = root.resolve()
    results = run_eval(root, settings=settings, live=live)
    passed = sum(1 for r in results if r.passed)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        typer.secho(f"[{mark}] {r.case_id}: {r.detail[:120]}", fg="green" if r.passed else "red")
    typer.echo(f"\n{passed}/{len(results)} passed")
    if passed != len(results):
        raise typer.Exit(code=1)


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
            typer.secho(f"-> tool {event.tool_name}({event.text})", fg="cyan")
        elif event.kind == "tool_result":
            preview = event.text if len(event.text) < 400 else event.text[:400] + "..."
            typer.secho(f"<- {event.tool_name}: {preview}", fg="bright_black")
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
