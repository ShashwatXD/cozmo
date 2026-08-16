"""User and workspace config paths."""

from __future__ import annotations

import os
from pathlib import Path

CONFIG_FILENAME = "config.json"


def user_config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "cozmo"
    return Path.home() / ".cozmo"


def user_config_path() -> Path:
    return user_config_dir() / CONFIG_FILENAME


def workspace_dir(workdir: Path) -> Path:
    return workdir.resolve() / ".cozmo"


def permissions_path(workdir: Path | None = None) -> Path:
    """Project-level allow/deny rules for mutating tools."""
    root = (workdir or Path.cwd()).resolve()
    return workspace_dir(root) / "permissions.json"


def project_config_path(workdir: Path | None = None) -> Path:
    root = (workdir or Path.cwd()).resolve()
    return workspace_dir(root) / CONFIG_FILENAME


def workspace_history_dir(workdir: Path | None = None) -> Path:
    root = (workdir or Path.cwd()).resolve()
    return workspace_dir(root) / "history"


def session_events_path(workdir: Path, session_id: str) -> Path:
    return workspace_history_dir(workdir) / f"{session_id}.jsonl"


def sessions_index_path(workdir: Path | None = None) -> Path:
    return workspace_history_dir(workdir) / "sessions.jsonl"


def history_index_path(workdir: Path | None = None) -> Path:
    """Vector index over past session turns (separate from code `index.json`)."""
    root = (workdir or Path.cwd()).resolve()
    return workspace_dir(root) / "history_index.json"


def history_rag_meta_path(workdir: Path | None = None) -> Path:
    root = (workdir or Path.cwd()).resolve()
    return workspace_dir(root) / "history_rag_meta.json"
