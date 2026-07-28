"""
Chat message shapes.

What: Role + Message (incl. tool_calls / tool results).
Why: one shared type for memory, LLM calls, and agent loop — DRY.
Layer: domain (Model).
Flutter: like a Freezed ChatMessage model.
"""

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
    """Flutter: immutable Freezed chat message (may carry tool calls)."""

    role: Role
    content: str = ""
    # Assistant requested tools
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    # Tool role: which call this result answers
    tool_call_id: str | None = None
    name: str | None = None
