"""Stuck detector heuristics + agent loop integration."""

import json
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.app.stuck_detector import (
    StuckStep,
    detect_stuck,
    step_from_tool_call,
)
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.stub import StubLLMClient, _messages_since_last_user
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


def _repeat_step(*, error: bool = False) -> StuckStep:
    return step_from_tool_call(
        tool_name="search_repo",
        args={"query": "missing"},
        result="no hits" if not error else "ERROR: denied",
        is_error=error,
    )


def test_not_stuck_with_few_steps() -> None:
    assert detect_stuck([_repeat_step()]).stuck is False


def test_repeating_action_observation() -> None:
    steps = [_repeat_step() for _ in range(4)]
    verdict = detect_stuck(steps)
    assert verdict.stuck is True
    assert verdict.reason == "repeating_action_observation"
    assert "same tool call and result" in verdict.message


def test_repeating_action_error() -> None:
    steps = [_repeat_step(error=True) for _ in range(4)]
    verdict = detect_stuck(steps)
    assert verdict.stuck is True
    assert verdict.reason == "repeating_action_error"


def test_alternating_pattern() -> None:
    a = step_from_tool_call(
        tool_name="read_file", args={"path": "a.py"}, result="content a"
    )
    b = step_from_tool_call(
        tool_name="read_file", args={"path": "b.py"}, result="content b"
    )
    steps = [a, b, a, b, a, b]
    verdict = detect_stuck(steps)
    assert verdict.stuck is True
    assert verdict.reason == "alternating_pattern"


class LoopingStubLLM:
    """Always returns the same tool call until the harness injects a stuck hint."""

    def __init__(self, call: ToolCall) -> None:
        self._call = call

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools=None,
    ) -> CompletionResult:
        after_user = _messages_since_last_user(messages)
        if any("[SYSTEM — STUCK DETECTOR]" in m.content for m in messages):
            return CompletionResult(
                content="Breaking the loop and summarizing findings.",
                usage=Usage(),
                finish_reason="stop",
            )
        if any(m.role == Role.TOOL for m in after_user):
            return CompletionResult(
                content="",
                usage=Usage(),
                finish_reason="tool_calls",
                tool_calls=(self._call,),
            )
        return CompletionResult(
            content="",
            usage=Usage(),
            finish_reason="tool_calls",
            tool_calls=(self._call,),
        )

    def stream(self, messages, *, temperature: float = 0.2):
        yield self.complete(messages, temperature=temperature).content or ""


def test_agent_injects_stuck_hint_after_repeated_tools() -> None:
    call = ToolCall(
        id="call_1",
        name="search_repo",
        arguments=json.dumps({"query": "nothing_here"}),
    )
    llm = LoopingStubLLM(call)
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    runner = AgentRunner(llm, reg, ToolExecutor(reg), max_steps=10)

    events = list(runner.run_events("Find the missing symbol"))
    stuck_events = [e for e in events if e.kind == "stuck"]
    assert len(stuck_events) == 1
    assert stuck_events[0].stop_reason == "repeating_action_observation"

    result = runner.last_result
    assert result is not None
    assert "Breaking the loop" in result.final_text
