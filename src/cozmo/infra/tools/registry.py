"""Tool registry + executor with output shaping."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cozmo.domain.tools import ToolCall, ToolResult, ToolSpec
from cozmo.infra.tools.caps import DEFAULT_MAX_TOOL_CHARS, shape_tool_content


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], str]) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)


class ToolExecutor:
    """Runs ToolCalls; errors become ToolResult(is_error=True) for the model."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        max_chars: int = DEFAULT_MAX_TOOL_CHARS,
    ) -> None:
        self._registry = registry
        self._max_chars = max(256, max_chars)

    def execute(self, call: ToolCall) -> ToolResult:
        registered = self._registry.get(call.name)
        if registered is None:
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Unknown tool: {call.name}",
                is_error=True,
            )
        try:
            args = json.loads(call.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be a JSON object")
            content = registered.handler(args)
            content = shape_tool_content(
                call.name, content, max_chars=self._max_chars
            )
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=content,
                is_error=False,
            )
        except Exception as exc:  # noqa: BLE001 - feed error back to the model
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"Tool error: {exc}",
                is_error=True,
            )
