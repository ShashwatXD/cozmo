"""Completion results shared across providers."""

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
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def truncated(self) -> bool:
        """True when the provider stopped because it hit the output limit."""
        return self.finish_reason in {"length", "max_tokens"}


class TransientLLMError(RuntimeError):
    """Provider failure worth retrying (rate limit, overload, 5xx, network)."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s
