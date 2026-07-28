"""
Ports — contracts the app depends on.

What: LLMClient Protocol (complete + stream + tools).
Why: ViewModel talks to an abstraction; OpenAI/Ollama are adapters.
Layer: domain (Model). Must NOT import openai/httpx.
Flutter: like `abstract class AuthRepository`.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from cozmo.domain.completion import CompletionResult
from cozmo.domain.messages import Message
from cozmo.domain.tools import ToolSpec


@runtime_checkable
class LLMClient(Protocol):
    """Flutter: AuthRepository — app depends on this, not on a vendor SDK."""

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
        """
        Yield text chunks as they arrive (no tools — agent uses complete()).

        Flutter: like a Stream<String> the UI listens to (StreamBuilder).
        """
        ...
