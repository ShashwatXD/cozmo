"""Drive the interactive REPL and the index / eval / config commands end to end (stub provider)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cozmo.cli.main import app

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for key, value in {
        "COZMO_PROVIDER": "stub",
        "COZMO_MODEL": "stub-model",
        "COZMO_SKIP_SETUP": "1",
        "COZMO_VECTOR_BACKEND": "json",
        "COZMO_MCP_DISABLED": "1",
    }.items():
        monkeypatch.setenv(key, value)
    return tmp_path


@pytest.fixture
def repo(home: Path) -> Path:
    root = home / "proj"
    root.mkdir()
    (root / "math_utils.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    return root


def test_repl_slash_commands(repo: Path) -> None:
    script = "\n".join(
        [
            "/help",
            "/ask",
            "explain add",
            "/agent",
            "",
            "/sessions",
            "/export md",
            "/export json",
            "/continue",
            "/continue nope",
            "/clear",
            "/compact",
            "/config",
            "/exit",
        ]
    )
    result = runner.invoke(app, ["-w", str(repo), "--no-index"], input=script + "\n")
    out = result.stdout
    assert result.exit_code == 0, out
    assert "/continue [id]" in out  # /help
    assert "mode → ask" in out and "mode → agent" in out
    assert "memory cleared" in out
    assert "session not found: nope" in out
    sessions = list((repo / ".cozmo" / "history").glob("*.jsonl"))
    assert sessions, "REPL turns are persisted as JSONL session events"


def test_index_then_semantic_session(repo: Path) -> None:
    result = runner.invoke(app, ["index", "-w", str(repo), "--embedder", "stub"])
    assert result.exit_code == 0, result.stdout
    assert (repo / ".cozmo" / "index.json").is_file()
    meta = json.loads((repo / ".cozmo" / "rag_meta.json").read_text(encoding="utf-8"))
    assert "math_utils.py" in meta["files"]

    again = runner.invoke(app, ["index", "-w", str(repo), "--embedder", "stub"])
    assert again.exit_code == 0


def test_eval_command_passes_on_fixture(home: Path) -> None:
    result = runner.invoke(app, ["eval", "-w", str(FIXTURE)])
    assert result.exit_code == 0, result.stdout
    assert "FAIL" not in result.stdout


def test_config_show_masks_key(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COZMO_API_KEY", "sk-supersecretvalue")
    runner.invoke(app, ["setup", "-y"])
    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0, result.stdout
    assert "sk-supersecretvalue" not in result.stdout


def test_ask_flag_one_shot(repo: Path) -> None:
    result = runner.invoke(app, ["-w", str(repo), "--ask", "-m", "what does add do?", "--no-index"])
    assert result.exit_code == 0, result.stdout
