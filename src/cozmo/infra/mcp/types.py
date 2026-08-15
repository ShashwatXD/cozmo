"""MCP server config models (optional dependency)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_NAME_OK = re.compile(r"^[A-Za-z0-9_-]+$")


class McpServerConfig(BaseModel):
    """One MCP server entry from config JSON / Settings.mcp_servers."""

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    enabled: bool = True
    timeout_s: float = 60.0

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name or not _NAME_OK.match(name):
            raise ValueError(
                "mcp server name must be non-empty [A-Za-z0-9_-]+ "
                f"(got {value!r})"
            )
        return name


def parse_mcp_servers(raw: Any) -> list[McpServerConfig]:
    """Best-effort parse of config list; skip invalid entries."""
    if not raw:
        return []
    if not isinstance(raw, list):
        return []
    out: list[McpServerConfig] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            cfg = McpServerConfig.model_validate(item)
        except Exception:
            continue
        if cfg.enabled:
            out.append(cfg)
    return out


def mcp_tool_name(server: str, tool: str) -> str:
    """Stable Cozmo tool id for an MCP tool."""
    safe_tool = re.sub(r"[^A-Za-z0-9_-]+", "_", tool).strip("_") or "tool"
    return f"mcp__{server}__{safe_tool}"


def is_mcp_tool(name: str) -> bool:
    return name.startswith("mcp__")
