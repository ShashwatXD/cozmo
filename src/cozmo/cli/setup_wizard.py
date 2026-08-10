"""Interactive setup: provider → API key → model list."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from cozmo.cli import ux
from cozmo.infra.config.paths import user_config_path
from cozmo.infra.config.store import example_config, load_user_config, save_user_config
from cozmo.infra.llm.model_catalog import validate_and_list

def _write_example(directory: Path) -> None:
    import json

    path = directory / "config.example.json"
    path.write_text(
        json.dumps(example_config(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

def _pick_model(
    provider: str,
    api_key: str,
    *,
    base_url: str | None,
    previous: str | None,
) -> str:
    ux.print_dim("  Checking API key + fetching models…")
    catalog = validate_and_list(provider, api_key, base_url=base_url)
    if not catalog.ok:
        ux.print_err(catalog.error or "Could not validate key")
        if not ux.confirm("Continue anyway and type a model id manually?", default=False):
            raise typer.Exit(code=1)
        return typer.prompt("Model id", default=previous or "").strip()

    ux.print_ok("API key valid")
    models = list(catalog.models)
    if not models:
        return typer.prompt("Model id", default=previous or "").strip()

    fav_set = set(catalog.favorites)
    choices: list[tuple[str, str]] = []
    for mid in models:
        label = f"★ {mid}" if mid in fav_set else mid
        choices.append((mid, label))
    if len(choices) > 60:
        choices = choices[:60]
        choices.append(("__manual__", "… type a model id manually"))

    default = previous if previous in {c[0] for c in choices} else choices[0][0]
    picked = ux.select("Model", choices, default=default)
    if picked == "__manual__":
        return typer.prompt("Model id", default=previous or choices[0][0]).strip()
    return picked

def run_setup(*, non_interactive: bool = False) -> Path:
    existing = load_user_config()
    data: dict[str, Any] = dict(existing)

    if non_interactive:
        if not data:
            data = {
                "provider": "stub",
                "model": "stub-model",
                "embedder": "auto",
                "allow_write": True,
                "allow_shell": False,
            }
        path = save_user_config(data)
        _write_example(path.parent)
        return path

    typer.echo()
    try:
        from rich.console import Console

        Console().print("[bold cyan]cozmo setup[/]  [dim]BYOK · key stays on your machine[/]")
    except Exception:
        typer.secho("cozmo setup", fg="cyan", bold=True)
    ux.print_dim(f"Config → {user_config_path()}\n")

    provider = ux.select(
        "Provider",
        [
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic (Claude)"),
            ("openrouter", "OpenRouter"),
            ("ollama", "Ollama (local)"),
            ("custom", "Custom OpenAI-compatible URL"),
            ("stub", "Stub (offline tests)"),
        ],
        default=str(data.get("provider") or "openai"),
    )

    if provider == "stub":
        data = {
            **data,
            "provider": "stub",
            "model": "stub-model",
            "embedder": "auto",
            "allow_write": True,
            "allow_shell": False,
        }
        data.pop("api_key", None)
        data.pop("base_url", None)
        path = save_user_config(data)
        _write_example(path.parent)
        ux.print_setup_summary(path, data)
        return path

    if provider == "ollama":
        base = typer.prompt(
            "Ollama URL",
            default=str(data.get("base_url") or "http://127.0.0.1:11434/v1"),
        ).strip()
        data["provider"] = "ollama"
        data["base_url"] = base
        data.pop("api_key", None)
        catalog = validate_and_list("ollama", "", base_url=base)
        if catalog.ok and catalog.models:
            ux.print_ok(f"Found {len(catalog.models)} local model(s)")
            choices = [(m, m) for m in catalog.models]
            choices.append(("__manual__", "… type model name"))
            prev = str(data.get("model") or "")
            default = prev if prev in catalog.models else catalog.models[0]
            picked = ux.select("Model", choices, default=default)
            data["model"] = (
                typer.prompt("Model", default=default).strip()
                if picked == "__manual__"
                else picked
            )
        else:
            ux.print_err(catalog.error or "No models")
            data["model"] = typer.prompt(
                "Model (pull first: ollama pull qwen2.5:3b)",
                default=str(data.get("model") or "qwen2.5:3b"),
            ).strip()
    elif provider == "custom":
        base = typer.prompt(
            "Base URL",
            default=str(data.get("base_url") or ""),
        ).strip()
        if not base:
            ux.print_err("Base URL required")
            raise typer.Exit(code=1)
        key = ux.prompt_secret(
            "API key",
            required=True,
            existing=str(data.get("api_key") or ""),
        )
        data["provider"] = "openai"
        data["base_url"] = base
        data["api_key"] = key
        data["model"] = _pick_model(
            "openai",
            key or "",
            base_url=base,
            previous=str(data.get("model") or "") or None,
        )
    else:
        key = ux.prompt_secret(
            "API key",
            required=True,
            existing=str(data.get("api_key") or ""),
        )
        data["provider"] = provider
        data["api_key"] = key
        if provider == "openrouter":
            data["base_url"] = "https://openrouter.ai/api/v1"
        elif provider == "openai":
            if data.get("base_url") and "openai.com" not in str(data.get("base_url")):
                data["base_url"] = None
        elif provider == "anthropic":
            data.pop("base_url", None)

        data["model"] = _pick_model(
            provider,
            key or "",
            base_url=data.get("base_url"),
            previous=str(data.get("model") or "") or None,
        )

    data["embedder"] = "auto"
    if data.get("provider") == "ollama":
        data["embedding_model"] = "nomic-embed-text"
    elif data.get("provider") in {"openai", "openrouter"}:
        data["embedding_model"] = "text-embedding-3-small"

    data["allow_write"] = ux.confirm(
        "Allow file writes?", default=bool(data.get("allow_write", True))
    )
    data["allow_shell"] = ux.confirm(
        "Allow shell commands?", default=bool(data.get("allow_shell", False))
    )

    path = save_user_config(data)
    _write_example(path.parent)
    ux.print_setup_summary(path, data)
    return path
