"""Message + completion domain types."""

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.prompts.loader import load_system_prompt


def test_message() -> None:
    m = Message(role=Role.USER, content="hi")
    assert m.role == Role.USER


def test_usage_total() -> None:
    u = Usage(prompt_tokens=3, completion_tokens=7)
    assert u.total_tokens == 10


def test_completion_result() -> None:
    r = CompletionResult(content="ok", usage=Usage(1, 2))
    assert r.content == "ok"


def test_default_prompt_loads() -> None:
    text = load_system_prompt("default")
    assert "Cozmo" in text
