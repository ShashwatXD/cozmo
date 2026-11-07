"""End-to-end MCP over stdio: spawn a real server, discover tools, call through the agent registry."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from cozmo.domain.tools import ToolCall
from cozmo.infra.mcp.client import McpManager
from cozmo.infra.mcp.register import register_mcp_tools
from cozmo.infra.mcp.types import McpServerConfig
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry

SERVER = Path(__file__).parent / "fixtures" / "mcp" / "echo_server.py"


@pytest.fixture
def manager():
    mgr = McpManager(call_timeout_s=30)
    yield mgr
    mgr.close()


def test_stdio_server_tools_are_callable(manager: McpManager) -> None:
    warnings = manager.connect_servers(
        [
            McpServerConfig(
                name="echo", transport="stdio", command=sys.executable, args=[str(SERVER)]
            ),
            McpServerConfig(name="broken", transport="stdio", command="/nonexistent/binary"),
        ]
    )
    # A dead server fails soft with a warning; the good one still connects.
    assert len(warnings) == 1 and "broken" in warnings[0]
    names = {t.local_name for t in manager.tools}
    assert {"mcp__echo__add", "mcp__echo__shout"} <= names

    reg = ToolRegistry()
    assert register_mcp_tools(reg, manager) == 2
    executor = ToolExecutor(reg)

    res = executor.execute(ToolCall(id="1", name="mcp__echo__add", arguments='{"a": 2, "b": 40}'))
    assert not res.is_error and "42" in res.content
    res = executor.execute(ToolCall(id="2", name="mcp__echo__shout", arguments='{"text": "hi"}'))
    assert "HI" in res.content
    res = executor.execute(ToolCall(id="3", name="mcp__echo__add", arguments='{"a": "x"}'))
    assert res.is_error or "error" in res.content.lower()
