"""MCP session manager: connect servers, list/call tools, fail soft."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from cozmo.infra.mcp.types import McpServerConfig, mcp_tool_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpToolInfo:
    server: str
    remote_name: str
    local_name: str
    description: str
    parameters: dict[str, Any]


class McpManager:
    """
    Long-lived MCP client connections on a background event loop.

    Sync call_tool() bridges into the loop so ToolExecutor stays sync.
    """

    def __init__(self, *, call_timeout_s: float = 120.0) -> None:
        self._call_timeout_s = call_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, Any] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        self._tools: list[McpToolInfo] = []
        self._lock = threading.Lock()

    @property
    def tools(self) -> list[McpToolInfo]:
        return list(self._tools)

    def start(self) -> None:
        if self._loop is not None:
            return
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=self._run_loop,
            args=(loop,),
            name="cozmo-mcp",
            daemon=True,
        )
        self._loop = loop
        self._thread = thread
        thread.start()

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._aclose_all(), loop)
            fut.result(timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mcp close: %s", exc)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._sessions.clear()
        self._stacks.clear()
        self._tools.clear()

    def connect_servers(self, servers: list[McpServerConfig]) -> list[str]:
        """
        Connect each server and collect tools. Returns warning strings.
        Fail-soft: one bad server does not block others.
        """
        warnings: list[str] = []
        if not servers:
            return warnings
        try:
            import mcp  # noqa: F401
        except ImportError:
            return [
                "mcp package missing — install with: pip install 'cozmo-agent[mcp]'"
            ]

        self.start()
        assert self._loop is not None
        for cfg in servers:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._aconnect(cfg), self._loop
                )
                tools = fut.result(timeout=max(30.0, cfg.timeout_s + 10))
                with self._lock:
                    self._tools.extend(tools)
            except Exception as exc:  # noqa: BLE001
                msg = f"mcp server '{cfg.name}' failed: {exc}"
                logger.warning(msg)
                warnings.append(msg)
        return warnings

    def call_tool(self, server: str, remote_name: str, arguments: dict[str, Any]) -> str:
        if self._loop is None:
            raise RuntimeError("MCP manager not started")
        fut = asyncio.run_coroutine_threadsafe(
            self._acall(server, remote_name, arguments), self._loop
        )
        return fut.result(timeout=self._call_timeout_s)

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
        loop.close()

    async def _aconnect(self, cfg: McpServerConfig) -> list[McpToolInfo]:
        from mcp import ClientSession

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if cfg.transport == "stdio":
                if not cfg.command:
                    raise ValueError("stdio transport requires command")
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client

                params = StdioServerParameters(
                    command=cfg.command,
                    args=list(cfg.args or []),
                    env=dict(cfg.env) if cfg.env else None,
                )
                read, write = await stack.enter_async_context(stdio_client(params))
            elif cfg.transport == "http":
                if not cfg.url:
                    raise ValueError("http transport requires url")
                try:
                    from mcp.client.streamable_http import streamable_http_client
                except ImportError as exc:
                    raise ImportError(
                        "streamable HTTP client unavailable in this mcp version"
                    ) from exc
                streams = await stack.enter_async_context(
                    streamable_http_client(cfg.url)
                )
                read, write = streams[0], streams[1]
            else:
                raise ValueError(f"unsupported transport: {cfg.transport}")

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
            tools: list[McpToolInfo] = []
            for tool in listed.tools:
                remote = str(tool.name)
                schema = getattr(tool, "inputSchema", None) or {
                    "type": "object",
                    "properties": {},
                }
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                tools.append(
                    McpToolInfo(
                        server=cfg.name,
                        remote_name=remote,
                        local_name=mcp_tool_name(cfg.name, remote),
                        description=(tool.description or remote).strip()
                        or f"MCP tool {remote} via {cfg.name}",
                        parameters=schema,
                    )
                )
            self._stacks[cfg.name] = stack
            self._sessions[cfg.name] = session
            return tools
        except Exception:
            await stack.aclose()
            raise

    async def _acall(
        self, server: str, remote_name: str, arguments: dict[str, Any]
    ) -> str:
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server not connected: {server}")
        result = await session.call_tool(remote_name, arguments=arguments or {})
        text = _flatten_content(result)
        if getattr(result, "isError", False):
            raise RuntimeError(text or "MCP tool returned isError")
        return text or "(empty MCP result)"

    async def _aclose_all(self) -> None:
        for name, stack in list(self._stacks.items()):
            try:
                await stack.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.warning("mcp close %s: %s", name, exc)
        self._stacks.clear()
        self._sessions.clear()


def _flatten_content(result: Any) -> str:
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None)
        if btype == "text" or hasattr(block, "text"):
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        elif hasattr(block, "data") and getattr(block, "mimeType", None):
            parts.append(f"[binary {block.mimeType} {len(block.data)} bytes]")
    structured = getattr(result, "structuredContent", None)
    if structured and not parts:
        parts.append(str(structured))
    return "\n".join(parts).strip()
