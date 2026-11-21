"""Old tool results become one-line notes; the model calls the tool again if it needs them."""

from __future__ import annotations

import json

from cozmo.app.agent import AgentRunner
from cozmo.app.context_editing import CLEARED_PREFIX, clear_old_tool_results
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry


def _read_turn(n: int, size: int = 2_000) -> list[Message]:
    call = ToolCall(
        id=f"c{n}", name="read_file", arguments=json.dumps({"path": f"f{n}.py", "start_line": 1})
    )
    return [
        Message(role=Role.USER, content=f"look at f{n}"),
        Message(role=Role.ASSISTANT, content="", tool_calls=(call,)),
        Message(role=Role.TOOL, content="x" * size, tool_call_id=f"c{n}", name="read_file"),
        Message(role=Role.ASSISTANT, content=f"f{n} looks fine"),
    ]


def test_keeps_the_newest_results_and_clears_the_rest() -> None:
    history = [m for n in range(1, 5) for m in _read_turn(n)]
    out, report = clear_old_tool_results(history, keep_recent=2)
    tools = [m for m in out if m.role == Role.TOOL]
    assert report.cleared == 2
    assert [t.content.startswith(CLEARED_PREFIX) for t in tools] == [True, True, False, False]
    assert 'read_file(path="f1.py", start_line=1)' in tools[0].content
    assert "Call it again" in tools[0].content
    assert report.chars_freed > 3_000
    # Structure is untouched: same count, same tool_call ids, so providers still accept it.
    assert len(out) == len(history)
    assert [m.tool_call_id for m in out] == [m.tool_call_id for m in history]


def test_short_results_and_already_cleared_ones_are_left_alone() -> None:
    history = [*_read_turn(1, size=50), *_read_turn(2), *_read_turn(3)]
    once, first = clear_old_tool_results(history, keep_recent=1)
    assert first.cleared == 1, "the 50-char result is too small to be worth clearing"
    twice, second = clear_old_tool_results(once, keep_recent=1)
    assert second.cleared == 0 and twice == once


def test_large_write_payloads_are_not_repeated_in_the_note() -> None:
    call = ToolCall(
        id="w", name="write_file", arguments=json.dumps({"path": "a.py", "content": "y" * 5_000})
    )
    history = [
        Message(role=Role.ASSISTANT, content="", tool_calls=(call,)),
        Message(role=Role.TOOL, content="z" * 1_000, tool_call_id="w", name="write_file"),
        *_read_turn(2),
    ]
    out, _ = clear_old_tool_results(history, keep_recent=1)
    note = out[1].content
    assert 'write_file(path="a.py")' in note
    assert "yyyy" not in note


class _ScriptedLLM:
    """Answers each turn without tools and records what it was sent."""

    def __init__(self) -> None:
        self.seen: list[list[Message]] = []

    def complete(self, messages, *, temperature=0.2, tools=None, **_):
        self.seen.append(list(messages))
        return CompletionResult(content="ok", tool_calls=(), usage=Usage(), finish_reason="stop")


def test_agent_clears_old_results_before_the_next_turn() -> None:
    memory = ConversationMemory(max_messages=100)
    memory.extend([m for n in range(1, 5) for m in _read_turn(n)])
    llm = _ScriptedLLM()
    reg = ToolRegistry()
    runner = AgentRunner(
        llm,
        reg,
        ToolExecutor(reg),
        memory=memory,
        policy=AgentPolicy(keep_tool_results=1, context_token_budget=1_000_000),
    )
    events = list(runner.run_events("next question"))
    assert any(e.kind == "context" and "cleared 3" in e.text for e in events)
    sent_tools = [m for m in llm.seen[0] if m.role == Role.TOOL]
    assert sum(m.content.startswith(CLEARED_PREFIX) for m in sent_tools) == 3
    assert sent_tools[-1].content == "x" * 2_000, "the newest result is sent as is"
