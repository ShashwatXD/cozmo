"""User config paths + JSON store + settings merge order."""

from __future__ import annotations

import json
from pathlib import Path

from cozmo.infra.config.paths import user_config_dir, user_config_path
from cozmo.infra.config.store import load_user_config, save_user_config
from cozmo.settings import Settings, load_settings


def test_user_config_dir_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert user_config_dir() == tmp_path / ".cozmo"
    assert user_config_path() == tmp_path / ".cozmo" / "config.json"


def test_user_config_dir_xdg(monkeypatch, tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert user_config_dir() == xdg / "cozmo"


def test_save_and_load_user_config(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    save_user_config(
        {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "sk-test",
            "unknown_key": "drop-me",
        },
        path=path,
    )
    data = load_user_config(path)
    assert data["provider"] == "openai"
    assert data["api_key"] == "sk-test"
    assert "unknown_key" not in data
    assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_openai_api_key_normalized(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": "openai", "openai_api_key": "sk-old"}),
        encoding="utf-8",
    )
    data = load_user_config(path)
    assert data["api_key"] == "sk-old"
    assert "openai_api_key" not in data


def test_load_settings_reads_user_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    # Clear env overrides that CI/dev .env might set
    for key in list(__import__("os").environ):
        if key.startswith("COZMO_"):
            monkeypatch.delenv(key, raising=False)

    cfg = tmp_path / ".cozmo" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        json.dumps({"provider": "ollama", "model": "qwen2.5:3b", "embedder": "auto"}),
        encoding="utf-8",
    )

    # Avoid picking up a real cwd .env provider during the test
    monkeypatch.chdir(tmp_path)

    s = load_settings()
    assert s.provider == "ollama"
    assert s.model == "qwen2.5:3b"


def test_env_overrides_user_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / ".cozmo" / "config.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({"provider": "ollama", "model": "from-json"}), encoding="utf-8")

    monkeypatch.setenv("COZMO_PROVIDER", "stub")
    monkeypatch.setenv("COZMO_MODEL", "from-env")
    s = load_settings()
    assert s.provider == "stub"
    assert s.model == "from-env"


def test_project_config_overrides_user(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for key in list(__import__("os").environ):
        if key.startswith("COZMO_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    user = tmp_path / ".cozmo" / "config.json"
    user.parent.mkdir(parents=True)
    user.write_text(json.dumps({"provider": "ollama", "model": "from-user"}), encoding="utf-8")

    proj = tmp_path / ".cozmo" / "config.json"
    # same path as user when HOME==cwd — use nested project instead
    proj_root = tmp_path / "repo"
    proj_root.mkdir()
    monkeypatch.chdir(proj_root)
    # re-home user config under tmp_path/.cozmo (already written)
    pcfg = proj_root / ".cozmo" / "config.json"
    pcfg.parent.mkdir(parents=True)
    pcfg.write_text(json.dumps({"model": "from-project"}), encoding="utf-8")

    s = load_settings()
    assert s.model == "from-project"
    assert s.provider == "ollama"


def test_cli_init_overrides_env() -> None:
    s = Settings(provider="openai", model="cli-model")
    assert s.provider == "openai"
    assert s.model == "cli-model"


def test_config_cmd(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner

    from cozmo.cli.main import app

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("COZMO_PROVIDER", "stub")
    monkeypatch.setenv("COZMO_SKIP_SETUP", "1")
    runner = CliRunner()
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "global:" in result.stdout
    assert "config.json" in result.stdout
