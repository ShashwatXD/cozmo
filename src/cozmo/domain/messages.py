"""Chat message shapes used across memory, LLM, and the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cozmo.domain.tools import ToolCall

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    name: str | None = None
