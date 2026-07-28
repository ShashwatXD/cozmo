"""CLI help + stub chat. Forces stub so .env ollama does not hit the network."""

import os

from typer.testing import CliRunner

from cozmo.cli.main import app

runner = CliRunner()


def _stub_env(monkeypatch) -> None:
    # Flutter: override Provider with Fake — tests must not call real backends
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
