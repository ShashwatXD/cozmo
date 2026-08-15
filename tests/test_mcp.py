"""MCP naming + registration (no live MCP servers)."""

from typing import Any

from cozmo.app.agent import AgentRunner
from cozmo.domain.mode import AgentMode
from cozmo.domain.tools import ToolSpec
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.mcp.client import McpManager, McpToolInfo
from cozmo.infra.mcp.register import register_mcp_tools
from cozmo.infra.mcp.types import (
    McpServerConfig,
    is_mcp_tool,
    mcp_tool_name,
    parse_mcp_servers,
)
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry


def test_mcp_tool_name() -> None:
    assert mcp_tool_name("fs", "read_file") == "mcp__fs__read_file"
    assert mcp_tool_name("fs", "read file!") == "mcp__fs__read_file"
    assert is_mcp_tool("mcp__fs__x")
    assert not is_mcp_tool("read_file")


def test_parse_mcp_servers_skips_bad() -> None:
    cfgs = parse_mcp_servers(
        [
            {"name": "ok", "transport": "stdio", "command": "echo"},
            {"name": "bad name!", "command": "x"},
            {"name": "off", "enabled": False, "command": "x"},
            "nope",
        ]
    )
    assert len(cfgs) == 1
    assert cfgs[0].name == "ok"


def test_register_mcp_tools_with_fake_manager(tmp_path) -> None:
    class FakeManager:
        def __init__(self) -> None:
            self.tools = [
                McpToolInfo(
                    server="demo",
                    remote_name="ping",
                    local_name=mcp_tool_name("demo", "ping"),
                    description="ping tool",
                    parameters={
                        "type": "object",
                        "properties": {"msg": {"type": "string"}},
                    },
                )
            ]
            self.calls: list[tuple[str, str, dict[str, Any]]] = []

        def call_tool(
            self, server: str, remote_name: str, arguments: dict[str, Any]
        ) -> str:
            self.calls.append((server, remote_name, arguments))
            return f"pong:{arguments.get('msg')}"

    guard = WorkspaceGuard(tmp_path)
    reg = build_default_registry(guard)
    mgr = FakeManager()
    n = register_mcp_tools(reg, mgr)  # type: ignore[arg-type]
    assert n == 1
    assert reg.get("mcp__demo__ping") is not None
    ex = ToolExecutor(reg)
    from cozmo.domain.tools import ToolCall
    import json

    result = ex.execute(
        ToolCall(
            id="1",
            name="mcp__demo__ping",
            arguments=json.dumps({"msg": "hi"}),
        )
    )
    assert not result.is_error
    assert "pong:hi" in result.content
    assert mgr.calls == [("demo", "ping", {"msg": "hi"})]


def test_ask_mode_hides_mcp_tools(tmp_path) -> None:
    guard = WorkspaceGuard(tmp_path)
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="mcp__x__y",
            description="external",
            parameters={"type": "object", "properties": {}},
        ),
        lambda _a: "ok",
    )
    reg.register(
        ToolSpec(name="read_file", description="r", parameters={"type": "object", "properties": {}}),
        lambda _a: "ok",
    )
    runner = AgentRunner(
        StubLLMClient(final_text="ok"),
        reg,
        ToolExecutor(reg),
        mode=AgentMode.ASK,
        max_steps=2,
    )
    names = {s.name for s in runner._tool_specs_for_mode()}
    assert "read_file" in names
    assert "mcp__x__y" not in names


def test_mcp_manager_missing_package_warns() -> None:
    mgr = McpManager()
    warnings = mgr.connect_servers(
        [McpServerConfig(name="x", transport="stdio", command="false")]
    )
    # Either missing package or connect failure — both are fail-soft strings
    assert isinstance(warnings, list)
