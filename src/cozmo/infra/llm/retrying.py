"""Retry wrapper around any LLMClient."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from cozmo.domain.completion import CompletionResult, TransientLLMError
from cozmo.domain.messages import Message
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolSpec

_RETRYABLE = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    TransientLLMError,
    TimeoutError,
    ConnectionError,
)

_MAX_RETRY_AFTER_S = 30.0

T = TypeVar("T")


def _retry_after_hint(exc: BaseException | None) -> float | None:
    """Read retry-after from our own error type or an SDK response header."""
    hint = getattr(exc, "retry_after_s", None)
    if hint is not None:
        return hint
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


class RetryingLLMClient:
    """Decorator that retries transient failures on an inner LLMClient."""

    def __init__(
        self,
        inner: LLMClient,
        *,
        max_attempts: int = 3,
        sleep: Callable[[float], None] | None = None,
        on_retry: Callable[[], None] | None = None,
    ) -> None:
        self._inner = inner
        self._max_attempts = max(1, max_attempts)
        self._sleep = sleep
        self._on_retry = on_retry
        # Jitter stops many clients from retrying in lockstep after an outage.
        self._backoff = wait_exponential(multiplier=0.5, min=0.5, max=8) + wait_random(0, 0.5)

    def _wait(self, state: RetryCallState) -> float:
        """Honor a provider ``retry-after`` hint, else exponential backoff."""
        exc = state.outcome.exception() if state.outcome else None
        hint = _retry_after_hint(exc)
        if isinstance(hint, (int, float)) and hint > 0:
            return min(float(hint), _MAX_RETRY_AFTER_S)
        return self._backoff(state)

    def _before_sleep(self, _state: RetryCallState) -> None:
        if self._on_retry is not None:
            self._on_retry()

    def _retry(self, fn: Callable[[], T]) -> T:
        retrying = Retrying(
            reraise=True,
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type(_RETRYABLE),
            sleep=self._sleep or time.sleep,
            before_sleep=self._before_sleep,
        )
        return retrying(fn)

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        return self._retry(
            lambda: self._inner.complete(
                messages,
                temperature=temperature,
                json_mode=json_mode,
                tools=tools,
            )
        )

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """
        Retry until the first chunk arrives, then stream the rest.

        Provider streams are lazy generators: calling ``stream()`` sends no
        request, so failures surface on the first ``next()``. Retrying the bare
        call (as before) never caught them. Once text has been yielded we do not
        retry, because replaying would duplicate output.
        """

        def _open() -> tuple[Iterator[str], list[str]]:
            it = iter(self._inner.stream(messages, temperature=temperature))
            try:
                return it, [next(it)]
            except StopIteration:
                return it, []

        it, first = self._retry(_open)
        yield from first
        yield from it
