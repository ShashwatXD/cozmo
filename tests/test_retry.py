"""Retry wrapper retries then succeeds."""

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.infra.llm.retrying import RetryingLLMClient


class _Flaky:
    def __init__(self, fails: int) -> None:
        self.fails = fails
        self.calls = 0

    def complete(self, messages, *, temperature=0.2, json_mode=False, tools=None):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("boom")
        return CompletionResult(content="ok", usage=Usage())

    def stream(self, messages, *, temperature=0.2):
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("boom")
        yield "ok"


def test_retry_then_success() -> None:
    inner = _Flaky(fails=2)
    llm = RetryingLLMClient(inner, max_attempts=3)
    result = llm.complete([Message(role=Role.USER, content="hi")])
    assert result.content == "ok"
    assert inner.calls == 3
