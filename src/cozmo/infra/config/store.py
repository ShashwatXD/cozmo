"""Load/save user and project config JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from cozmo.infra.config.paths import user_config_path

USER_CONFIG_KEYS = frozenset(
    {
        "provider",
        "model",
        "worker_model",
        "workdir",
        "api_key",
        "base_url",
        "temperature",
        "timeout_s",
        "max_retries",
        "max_tokens",
        "allow_write",
        "allow_shell",
        "max_agent_steps",
        "memory_max_messages",
        "max_messages_before_compact",
        "context_token_budget",
        "max_tool_calls_per_session",
        "max_cost_usd",
        "session_timeout_s",
        "max_subagent_depth",
        "max_subagent_steps",
        "shell_timeout_s",
        "embedder",
        "embedding_model",
        "vector_backend",
        "history_enabled",
        "history_max_sessions",
        "history_max_events_per_session",
        "trace_enabled",
    }
)

_LEGACY_KEY_MAP = {
    "openai_api_key": "api_key",
    "openai_base_url": "base_url",
}

def load_user_config(path: Path | None = None) -> dict[str, Any]:
    return _load_json_config(path or user_config_path())

def load_project_config(workdir: Path | None = None) -> dict[str, Any]:
    from cozmo.infra.config.paths import project_config_path

    return _load_json_config(project_config_path(workdir))

def load_merged_file_config(workdir: Path | None = None) -> dict[str, Any]:
    merged = load_user_config()
    merged.update(load_project_config(workdir))
    return merged

def _normalize_keys(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        mapped = _LEGACY_KEY_MAP.get(key, key)
        if mapped in USER_CONFIG_KEYS:
            if mapped in out and key in _LEGACY_KEY_MAP:
                continue
            out[mapped] = value
    return out

def _load_json_config(cfg_path: Path) -> dict[str, Any]:
    if not cfg_path.is_file():
        return {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return _normalize_keys(raw)

def example_config() -> dict[str, Any]:
    return {
        "provider": "ollama",
        "model": "qwen2.5:3b",
        "api_key": None,
        "base_url": "http://127.0.0.1:11434/v1",
        "embedder": "auto",
        "embedding_model": "text-embedding-3-small",
        "temperature": 0.2,
        "timeout_s": 120.0,
        "max_retries": 3,
        "max_tokens": 2048,
        "allow_write": True,
        "allow_shell": False,
        "max_agent_steps": 8,
        "memory_max_messages": 40,
        "trace_enabled": True,
    }

def save_user_config(data: dict[str, Any], path: Path | None = None) -> Path:
    cfg_path = path or user_config_path()
    directory = cfg_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass

    normalized = _normalize_keys(data)
    filtered = {
        k: v for k, v in normalized.items() if k in USER_CONFIG_KEYS and v is not None
    }
    if "workdir" in filtered and isinstance(filtered["workdir"], Path):
        filtered["workdir"] = str(filtered["workdir"])

    tmp = cfg_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(cfg_path)
    try:
        os.chmod(cfg_path, 0o600)
    except OSError:
        pass
    return cfg_path
