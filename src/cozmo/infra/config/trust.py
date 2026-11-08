"""Hash-pinned trust for files that ship inside a repository.

A cloned repo is untrusted input. Its config can name a shell to enable, an endpoint
that would receive the API key, or an MCP command to start. Those settings only apply
once the user trusts the exact bytes of the file (`cozmo trust`). If the file changes
later, for example after a `git pull`, it has to be trusted again.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cozmo.infra.config.paths import project_config_path, user_config_dir, user_config_path

# Project config keys that cannot leak data, run code, or relax a guardrail.
PROJECT_SAFE_KEYS = frozenset(
    {
        "model",
        "worker_model",
        "temperature",
        "max_tokens",
        "memory_max_messages",
        "max_messages_before_compact",
        "context_token_budget",
        "max_tool_result_chars",
        "embedder",
        "embedding_model",
        "vector_backend",
        "history_enabled",
        "history_max_sessions",
        "history_max_events_per_session",
        "history_rag",
        "trace_enabled",
    }
)

# Settings a repo-level .env must not change without trust.
DOTENV_SENSITIVE_KEYS = frozenset(
    {"base_url", "openai_base_url", "allow_shell", "allow_write", "mcp_servers"}
)


def _trust_path() -> Path:
    override = os.environ.get("COZMO_TRUST_FILE", "").strip()
    return Path(override) if override else user_config_dir() / "trust.json"


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


class TrustStore:
    """Absolute file path -> sha256 of the bytes the user approved."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _trust_path()

    def _load(self) -> dict[str, str]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        files = raw.get("files") if isinstance(raw, dict) else None
        return {str(k): str(v) for k, v in files.items()} if isinstance(files, dict) else {}

    def _save(self, files: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "files": files}, indent=2) + "\n", "utf-8")
        tmp.replace(self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def is_trusted(self, file: Path) -> bool:
        digest = _digest(file)
        return digest is not None and self._load().get(str(file.resolve())) == digest

    def trust(self, file: Path) -> None:
        digest = _digest(file)
        if digest is None:
            return
        files = self._load()
        files[str(file.resolve())] = digest
        self._save(files)

    def revoke(self, file: Path) -> None:
        files = self._load()
        if files.pop(str(file.resolve()), None) is not None:
            self._save(files)


def filter_project_config(data: dict[str, Any], workdir: Path | None = None) -> dict[str, Any]:
    """Drop sensitive project settings unless this exact config file is trusted."""
    if not data or TrustStore().is_trusted(project_config_path(workdir)):
        return data
    return {k: v for k, v in data.items() if k in PROJECT_SAFE_KEYS}


def filter_dotenv(data: dict[str, Any], env_file: Path) -> dict[str, Any]:
    if not data or TrustStore().is_trusted(env_file):
        return data
    return {k: v for k, v in data.items() if k not in DOTENV_SENSITIVE_KEYS}


def ignored_project_keys(workdir: Path | None = None) -> list[str]:
    """Project settings currently being ignored, for the startup warning."""
    from cozmo.infra.config.store import load_project_config

    data = load_project_config(workdir)
    path = project_config_path(workdir)
    if not data or path == user_config_path() or TrustStore().is_trusted(path):
        return []
    return sorted(k for k in data if k not in PROJECT_SAFE_KEYS)
