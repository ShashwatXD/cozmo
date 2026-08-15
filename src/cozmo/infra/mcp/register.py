"""Register MCP tools onto a Cozmo ToolRegistry."""

from __future__ import annotations

from typing import Any

from cozmo.domain.tools import ToolSpec
from cozmo.infra.mcp.client import McpManager, McpToolInfo
from cozmo.infra.tools.registry import ToolRegistry


def register_mcp_tools(registry: ToolRegistry, manager: McpManager) -> int:
    """Attach discovered MCP tools. Returns count registered."""
    count = 0
    for info in manager.tools:
        if registry.get(info.local_name) is not None:
            continue
        registry.register(
            ToolSpec(
                name=info.local_name,
                description=f"[mcp:{info.server}] {info.description}",
                parameters=info.parameters
                if isinstance(info.parameters, dict)
                else {"type": "object", "properties": {}},
            ),
            _make_handler(manager, info),
        )
        count += 1
    return count


def _make_handler(manager: McpManager, info: McpToolInfo):
    def handler(args: dict[str, Any]) -> str:
        return manager.call_tool(info.server, info.remote_name, args)

    return handler
