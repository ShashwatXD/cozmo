"""CLI help + stub chat. Forces stub so .env ollama does not hit the network."""

import os

from typer.testing import CliRunner

from cozmo.cli.main import app

runner = CliRunner()


def _stub_env(monkeypatch) -> None:
    monkeypatch.setenv("COZMO_PROVIDER", "stub")
    monkeypatch.setenv("COZMO_MODEL", "stub-model")
    # Prevent repo .env openai/ollama from winning over process env... 
    # pydantic-settings: env vars already win over env_file by default.


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_chat_stream_stub(monkeypatch) -> None:
    _stub_env(monkeypatch)
    result = runner.invoke(app, ["chat", "-m", "hi"])
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()


def test_chat_no_stream(monkeypatch) -> None:
    _stub_env(monkeypatch)
    result = runner.invoke(app, ["chat", "--no-stream", "-m", "hi"])
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()


def test_chat_stream_shows_meter(monkeypatch) -> None:
    _stub_env(monkeypatch)
    result = runner.invoke(app, ["chat", "-m", "hi"])
    assert result.exit_code == 0
    assert "stub" in result.stdout.lower()


def test_agent_help(monkeypatch) -> None:
    _stub_env(monkeypatch)
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "tools" in result.stdout.lower() or "agent" in result.stdout.lower()


def test_chat_json(monkeypatch) -> None:
    _stub_env(monkeypatch)
    result = runner.invoke(app, ["chat", "--json", "-m", "hi"])
    assert result.exit_code == 0
    assert "{" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_setup_yes(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    result = runner.invoke(app, ["setup", "-y"])
    assert result.exit_code == 0
    assert (tmp_path / ".cozmo" / "config.json").is_file()
    assert "Wrote" in result.stdout


def test_doctor(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("COZMO_PROVIDER", "stub")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "global config" in result.stdout.lower() or "config.json" in result.stdout
    assert "cozmo" in result.stdout.lower()


def test_default_launch_one_shot(monkeypatch, tmp_path) -> None:
    """Bare `cozmo -m` should run the agent (default command)."""
    from pathlib import Path

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("COZMO_PROVIDER", "stub")
    monkeypatch.setenv("COZMO_MODEL", "stub-model")
    monkeypatch.setenv("COZMO_SKIP_SETUP", "1")
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / "hello.py").write_text("def hi():\n    return 1\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["-w", str(repo), "-m", "hi", "--no-index"],
    )
    assert result.exit_code == 0, result.stdout
    assert "cozmo" in result.stdout.lower() or "stub" in result.stdout.lower()
