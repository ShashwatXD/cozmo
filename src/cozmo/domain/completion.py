"""
Completion results shared across providers.

What: CompletionResult + token Usage (+ optional tool_calls).
Why: one return shape for stub/openai/ollama (DRY).
Layer: domain.
"""

from dataclasses import dataclass, field

from cozmo.domain.tools import ToolCall


@dataclass(frozen=True)
class Usage:
    """Token counts from the provider (0 if unknown / stub)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merged(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class CompletionResult:
    """Full non-stream reply (text and/or tool calls)."""

    content: str = ""
    usage: Usage = field(default_factory=Usage)
    # stop | length | tool_calls | ...
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
