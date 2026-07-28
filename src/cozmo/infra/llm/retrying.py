"""
Retry wrapper around any LLMClient.

What: retries transient failures (timeout, connection, rate limit).
Why: production gateways don't fail the user on one blip.
Layer: infra decorator.
"""

from __future__ import annotations

from collections.abc import Iterator

from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cozmo.domain.completion import CompletionResult
from cozmo.domain.messages import Message
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolSpec

_RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    TimeoutError,
    ConnectionError,
)


class RetryingLLMClient:
    """Decorator that retries transient failures on an inner LLMClient."""

    def __init__(self, inner: LLMClient, *, max_attempts: int = 3) -> None:
        self._inner = inner
        self._max_attempts = max(1, max_attempts)

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _call() -> CompletionResult:
            return self._inner.complete(
                messages,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools,
            )

        return _call()

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type(_RETRYABLE),
        )
        def _open() -> Iterator[str]:
            return self._inner.stream(messages, temperature=temperature)

        return _open()
