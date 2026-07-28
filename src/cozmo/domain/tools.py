"""
Tool types — schemas the model can call.

What: ToolSpec, ToolCall, ToolResult.
Why: one contract for registry, LLM, and executor (DRY).
Layer: domain.
Flutter: like a Freezed command / use-case descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Flutter: documentation for a use-case the Cubit can invoke by name."""

    name: str
    description: str
    # JSON Schema object for arguments
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_openai(self) -> dict[str, Any]:
        """OpenAI / Ollama function-calling shape."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """Model asked to run a tool — arguments are a JSON string from the provider."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    """Outcome of executing one ToolCall."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
