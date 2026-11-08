"""Measure every model call in one place: span, tokens, cost, retries."""

from __future__ import annotations

from collections.abc import Iterator

from cozmo.domain.completion import CompletionResult
from cozmo.domain.cost import UsageLedger
from cozmo.domain.messages import Message
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolSpec
from cozmo.infra.telemetry.tracer import Tracer


class MeteredLLM:
    """Decorator for LLMClient. Anything that calls a model is counted automatically."""

    def __init__(
        self, inner: LLMClient, *, model: str, role: str, ledger: UsageLedger, tracer: Tracer
    ) -> None:
        self._inner = inner
        self._model = model
        self._role = role
        self._ledger = ledger
        self._tracer = tracer

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        with self._tracer.span("llm", model=self._model, role=self._role) as span:
            retries_before = self._ledger.retries
            result = self._inner.complete(
                messages, temperature=temperature, json_mode=json_mode, tools=tools
            )
            cost = self._ledger.record(self._model, result.usage)
            span.set(
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                cost_usd=round(cost, 6),
                finish_reason=result.finish_reason,
                tool_calls=[tc.name for tc in result.tool_calls],
                retries=self._ledger.retries - retries_before,
            )
            return result

    def stream(self, messages: list[Message], *, temperature: float = 0.2) -> Iterator[str]:
        return self._inner.stream(messages, temperature=temperature)
