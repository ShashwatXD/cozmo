"""Cozmo CLI entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from cozmo import __version__
from cozmo.app.agent import AgentRunner
from cozmo.app.chat import ChatUseCase
from cozmo.app.eval_runner import run_eval
from cozmo.domain.completion import Usage
from cozmo.domain.cost import format_cost_line
from cozmo.domain.memory import ConversationMemory
from cozmo.infra.config.paths import (
    project_config_path,
    user_config_dir,
    user_config_path,
    workspace_dir,
)
from cozmo.cli import ux
from cozmo.cli.setup_wizard import run_setup
from cozmo.infra.config.store import load_user_config, save_user_config
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
    help="Cozmo — coding agent. Run `cozmo` to start.",
    no_args_is_help=False,
    add_completion=False,
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

def _settings_to_user_dict(settings: Settings) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "model": settings.model,
        "workdir": str(settings.workdir),
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "temperature": settings.temperature,
        "timeout_s": settings.timeout_s,
        "max_retries": settings.max_retries,
        "max_tokens": settings.max_tokens,
        "allow_write": settings.allow_write,
        "allow_shell": settings.allow_shell,
        "max_agent_steps": settings.max_agent_steps,
        "memory_max_messages": settings.memory_max_messages,
        "embedder": settings.embedder,
        "embedding_model": settings.embedding_model,
        "trace_enabled": settings.trace_enabled,
        "log_level": settings.log_level,
    }

def _ensure_config() -> None:
    if user_config_path().is_file():
        return
    if os.environ.get("COZMO_SKIP_SETUP") == "1":
        save_user_config(_settings_to_user_dict(load_settings()))
        return
    if os.environ.get("COZMO_PROVIDER") or os.environ.get("COZMO_API_KEY"):
        save_user_config(_settings_to_user_dict(load_settings()))
        return
    if not sys.stdin.isatty():
        run_setup(non_interactive=True)
        return
    run_setup(non_interactive=False)

def _build_indexes(root_dir: Path, settings: Settings, *, quiet: bool = False) -> None:
    embedder = build_embedder(settings)
    store = VectorStore()
    n = RepoIndexer(embedder, store).index_dir(root_dir)
    out = index_path(root_dir)
    store.save(out)
    if not quiet:
        typer.secho(f"indexed {n} chunks → {out}", fg="bright_black")

    try:
        from cozmo.indexer.repository_indexer import RepositoryIndexer as CodeRepoIndexer

        code_index = CodeRepoIndexer().index(root_dir)
        if not quiet:
            typer.secho(
                f"code-index {len(code_index.files)} files → {root_dir / '.cozmo' / 'code_index.json'}",
                fg="bright_black",
            )
    except Exception as exc:
        if not quiet:
            typer.secho(f"code-index skipped: {exc}", fg="bright_black")

def _ensure_index(root_dir: Path, settings: Settings, *, auto_index: bool) -> None:
    if not auto_index:
        return
    rag = index_path(root_dir)
    code = root_dir / ".cozmo" / "code_index.json"
    if rag.is_file() and code.is_file():
        return
    ux.print_dim("First time in this repo — indexing…")
    _build_indexes(root_dir, settings, quiet=False)

def _run_agent_session(
    *,
    settings: Settings,
    root_dir: Path,
    message: Optional[str],
    auto_index: bool = True,
) -> None:
    _ensure_index(root_dir, settings, auto_index=auto_index)

    guard = WorkspaceGuard(
        root_dir,
        allow_write=settings.allow_write,
        allow_shell=settings.allow_shell,
    )
    embedder = build_embedder(settings)
    store = load_store(root_dir)

    code_index = None
    sources: dict[str, str] = {}
    code_index_path = root_dir / ".cozmo" / "code_index.json"
    if code_index_path.is_file():
        try:
            from cozmo.domain.index import CodeIndex

            code_index = CodeIndex.load(code_index_path)
            for rel_path in code_index.files:
                full = root_dir / rel_path
                if full.is_file():
                    try:
                        sources[rel_path] = full.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except OSError:
                        pass
        except Exception:
            pass

    registry = build_default_registry(
        guard,
        vector_store=store,
        embedder=embedder,
        code_index=code_index,
        sources=sources,
    )
    executor = ToolExecutor(registry)
    llm = build_llm(settings)
    memory = ConversationMemory(max_messages=settings.memory_max_messages)
    trace_path = root_dir / ".cozmo" / "traces.jsonl"
    tracer = Tracer(
        trace_path if settings.trace_enabled else None,
        enabled=settings.trace_enabled,
    )
    runner = AgentRunner(
        llm,
        registry,
        executor,
        memory=memory,
        tracer=tracer,
        temperature=settings.temperature,
        max_steps=settings.max_agent_steps,
    )

    typer.echo()
    ux.print_banner()
    ux.print_session_status(
        provider=settings.provider,
        model=settings.model,
        workdir=root_dir,
        rag_chunks=len(store),
        allow_write=settings.allow_write,
        allow_shell=settings.allow_shell,
    )
    typer.echo()

    if message is not None:
        _agent_once(runner, message, settings)
        return

    ux.print_dim("Ask anything about this repo.")
    ux.print_dim("/help  /clear  /config  /setup  /exit")
    while True:
        try:
            text = typer.prompt("❯")
        except typer.Abort:
            break
        raw = text.strip()
        if raw in {"/exit", "/quit", "exit", "quit"}:
            break
        if raw in {"/help", "help"}:
            typer.echo(
                "  /clear   reset conversation memory\n"
                "  /config  show config paths\n"
                "  /setup   re-run provider wizard\n"
                "  /exit    quit"
            )
            continue
        if raw == "/clear":
            memory.clear()
            ux.print_dim("memory cleared")
            continue
        if raw == "/config":
            typer.echo(f"  {user_config_path()}")
            continue
        if raw == "/setup":
            run_setup(non_interactive=False)
            ux.print_dim("Restart cozmo to use new settings.")
            continue
        if not raw:
            continue
        _agent_once(runner, text, settings)

@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version"),
    message: Optional[str] = typer.Option(
        None, "--message", "-m", help="One-shot task (skip REPL)"
    ),
    workdir: Optional[Path] = typer.Option(
        None, "--workdir", "-w", help="Repo root (default: cwd)"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="stub | openai | anthropic | openrouter | ollama"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model id"),
    no_index: bool = typer.Option(
        False, "--no-index", help="Skip auto-index on first launch"
    ),
) -> None:
    """Start the coding agent (default). Subcommands: setup, doctor, index, …"""
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is not None:
        return

    _ensure_config()
    settings = _apply_provider(load_settings(), provider, model)
    root_dir = (workdir or Path.cwd()).resolve()
    _run_agent_session(
        settings=settings,
        root_dir=root_dir,
        message=message,
        auto_index=not no_index,
    )

@app.command("chat")
def chat(
    message: Optional[str] = typer.Option(None, "--message", "-m"),
    stream: bool = typer.Option(True, "--stream/--no-stream"),
    json_mode: bool = typer.Option(False, "--json"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="stub | openai | anthropic | openrouter | ollama"
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
        None, "--provider", "-p", help="stub | openai | anthropic | openrouter | ollama"
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model id override"),
    no_index: bool = typer.Option(False, "--no-index"),
) -> None:
    """Same as bare `cozmo` — kept as an explicit alias."""
    _ensure_config()
    settings = _apply_provider(load_settings(), provider, model)
    root_dir = (workdir or Path.cwd()).resolve()
    _run_agent_session(
        settings=settings,
        root_dir=root_dir,
        message=message,
        auto_index=not no_index,
    )

@app.command("index")
def index_cmd(
    workdir: Optional[Path] = typer.Option(None, "--workdir", "-w"),
    embedder_name: Optional[str] = typer.Option(
        None,
        "--embedder",
        help="hash | openai | ollama (overrides COZMO_EMBEDDER)",
    ),
) -> None:
    """Rebuild RAG + code index (usually automatic on first `cozmo`)."""
    settings = load_settings()
    if embedder_name:
        data = settings.model_dump()
        data["embedder"] = embedder_name
        settings = Settings(**data)
    root_dir = (workdir or Path.cwd()).resolve()
    _build_indexes(root_dir, settings, quiet=False)

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
        typer.secho(
            f"[{mark}] {r.case_id}: {r.detail[:120]}",
            fg="green" if r.passed else "red",
        )
    typer.echo(f"\n{passed}/{len(results)} passed")
    if passed != len(results):
        raise typer.Exit(code=1)

@app.command("setup")
def setup_cmd(
    non_interactive: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Write defaults without prompts",
    ),
) -> None:
    """Create or update ~/.cozmo/config.json (also runs automatically on first `cozmo`)."""
    path = run_setup(non_interactive=non_interactive)
    if non_interactive:
        typer.secho(f"Wrote {path}", fg="green")
        typer.secho("Edit it anytime, or run: cozmo config", fg="bright_black")

@app.command("config")
def config_cmd(
    init_project: bool = typer.Option(
        False,
        "--project",
        help="Create editable .cozmo/config.json in the current repo",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        help="Print config JSON (API key masked)",
    ),
) -> None:
    """
    Show where config files live — edit them in any editor.

      ~/.cozmo/config.json           global (provider, keys, defaults)
      <repo>/.cozmo/config.json      optional per-project overrides
    """
    import json

    user_path = user_config_path()
    proj_path = project_config_path(Path.cwd())

    if init_project:
        proj_path.parent.mkdir(parents=True, exist_ok=True)
        if not proj_path.is_file():
            data = {
                "allow_shell": False,
                "allow_write": True,
                "max_agent_steps": 8,
                "embedder": "hash",
            }
            proj_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            typer.secho(f"Created {proj_path}", fg="green")
        else:
            typer.secho(f"Already exists: {proj_path}", fg="yellow")
        typer.echo("Edit that file, then re-run cozmo.")
        return

    if not user_path.is_file():
        typer.secho("No user config yet — creating one…", fg="bright_black")
        run_setup(non_interactive=not sys.stdin.isatty())

    typer.echo("Config files (edit with any editor):\n")
    typer.echo(f"  global:  {user_path}")
    typer.echo(f"           ({'exists' if user_path.is_file() else 'missing'})")
    typer.echo(f"  project: {proj_path}")
    typer.echo(f"           ({'exists' if proj_path.is_file() else 'optional — cozmo config --project'})")
    typer.echo("")
    typer.secho(
        "Load order: CLI > COZMO_* env > cwd .env > project config > global config > defaults",
        fg="bright_black",
    )
    typer.echo("")
    typer.echo("Examples:")
    typer.echo('  nano ~/.cozmo/config.json')
    typer.echo('  cozmo config --show')
    typer.echo('  cozmo config --project   # create per-repo overrides')
    typer.echo('  cozmo setup              # interactive re-configure')

    if show and user_path.is_file():
        data = load_user_config(user_path)
        if data.get("api_key"):
            data["api_key"] = "***"
        typer.echo("\n--- global config ---")
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
        if proj_path.is_file():
            typer.echo("\n--- project config ---")
            typer.echo(proj_path.read_text(encoding="utf-8"))

@app.command("doctor")
def doctor_cmd() -> None:
    """Print config locations and effective settings (never prints secrets)."""
    cfg_path = user_config_path()
    cfg_dir = user_config_dir()
    proj = project_config_path(Path.cwd())
    cwd_env = Path.cwd() / ".env"
    settings = load_settings()
    work = Path.cwd().resolve()
    ws = workspace_dir(work)

    typer.echo(f"version:          {__version__}")
    typer.echo(f"user config dir:  {cfg_dir}")
    typer.echo(
        f"global config:    {cfg_path} "
        f"({'exists' if cfg_path.is_file() else 'missing'})"
    )
    typer.echo(
        f"project config:   {proj} "
        f"({'exists' if proj.is_file() else 'none'})"
    )
    typer.echo(f"cwd .env:         {cwd_env} ({'exists' if cwd_env.is_file() else 'none'})")
    typer.echo(f"workdir:          {work}")
    typer.echo(f"workspace data:   {ws}/")
    typer.echo("")
    from cozmo.infra.rag.factory import resolve_embedder

    backend, emb_model = resolve_embedder(settings)
    typer.echo(f"provider:         {settings.provider}")
    typer.echo(f"model:            {settings.model}")
    typer.echo(f"embedder:         {backend} · {emb_model}")
    typer.echo(f"base_url:         {settings.base_url or '(default)'}")
    typer.echo(f"api_key:          {'set' if settings.api_key else 'not set'}")
    typer.echo(f"allow_write:      {settings.allow_write}")
    typer.echo(f"allow_shell:      {settings.allow_shell}")
    if (Path.cwd() / ".env").is_file():
        typer.echo("")
        typer.secho(
            "Note: cwd .env overrides ~/.cozmo/config.json when both set the same keys.",
            fg="yellow",
        )
    typer.echo("")
    typer.secho("Edit config:  cozmo config", fg="cyan")
    typer.secho("Start agent:  cozmo", fg="cyan")

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

def _friendly_llm_error(exc: BaseException, settings: Settings) -> str:
    name = type(exc).__name__
    msg = str(exc) or name
    low = msg.lower()
    if "402" in msg or "more credits" in low or "can only afford" in low:
        return (
            "OpenRouter: not enough credits for this request.\n"
            f"  → add credits:  https://openrouter.ai/settings/credits\n"
            f"  → or lower max_tokens in ~/.cozmo/config.json "
            f"(current default {settings.max_tokens})\n"
            "  → or pick a cheaper model:  cozmo setup"
        )
    if settings.provider == "ollama":
        base = settings.base_url or "http://127.0.0.1:11434/v1"
        return (
            f"Cannot reach Ollama at {base}\n"
            "  → start it:  ollama serve\n"
            "  → pull model: ollama pull {0}\n"
            "  → or run:     cozmo setup   (pick openai / anthropic / openrouter)"
        ).format(settings.model)
    if settings.provider in {"openai", "openrouter", "anthropic"}:
        return (
            f"LLM request failed ({name}): {msg[:200]}\n"
            "  → check api_key / base_url:  cozmo doctor\n"
            "  → reconfigure:               cozmo setup"
        )
    return f"LLM error ({name}): {msg[:300]}"


def _agent_once(runner: AgentRunner, text: str, settings: Settings) -> None:
    try:
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
    except Exception as exc:  # noqa: BLE001 — surface as UX, don't dump traceback
        from cozmo.cli import ux as _ux

        _ux.print_err(_friendly_llm_error(exc, settings))
        return
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
