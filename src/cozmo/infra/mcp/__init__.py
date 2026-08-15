"""Optional MCP client hooks for Cozmo tools."""

from cozmo.infra.mcp.client import McpManager
from cozmo.infra.mcp.register import register_mcp_tools
from cozmo.infra.mcp.types import (
    McpServerConfig,
    is_mcp_tool,
    mcp_tool_name,
    parse_mcp_servers,
)

__all__ = [
    "McpManager",
    "McpServerConfig",
    "is_mcp_tool",
    "mcp_tool_name",
    "parse_mcp_servers",
    "register_mcp_tools",
]
