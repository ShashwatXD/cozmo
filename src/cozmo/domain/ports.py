"""LLM ports - ViewModel depends on these abstractions, not vendor SDKs."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from cozmo.domain.completion import CompletionResult
from cozmo.domain.messages import Message
from cozmo.domain.tools import ToolSpec

@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        """Return a full assistant reply (text and/or tool_calls)."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Yield text chunks (no tools; agent uses complete())."""
        ...
