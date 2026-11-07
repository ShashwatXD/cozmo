"""ReAct loop hard stops: max steps, tool-call budget, cost budget, timeout, stuck detector."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozmo.app import agent as agent_mod
from cozmo.app.agent import AgentRunner
from cozmo.app.history import SessionHistory
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.events import EventKind
from cozmo.domain.guardrails import AgentPolicy, StopReason
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall
from cozmo.infra.history.null import NullEventStore
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


class LoopingLLM:
    """Always asks for the same tool call: the pathological agent."""

    def __init__(self, *, usage: Usage | None = None, final: str = "gave up") -> None:
        self.usage = usage or Usage()
        self.final = final
        self.calls = 0
        self.seen: list[list[Message]] = []

    def complete(self, messages, *, temperature=0.2, json_mode=False, tools=None):
        self.calls += 1
        self.seen.append(list(messages))
        if tools is None:  # final synthesis after max steps
            return CompletionResult(content=self.final, usage=self.usage)
        call = ToolCall(
            id=f"c{self.calls}", name="read_file", arguments='{"path": "math_utils.py"}'
        )
        return CompletionResult(content="", usage=self.usage, tool_calls=(call,))

    def stream(self, messages, *, temperature=0.2):
        yield self.final


class _Events(NullEventStore):
    def __init__(self) -> None:
        self.events: list = []

    def append(self, event) -> None:
        self.events.append(event)


def _runner(llm, policy: AgentPolicy, **kw) -> AgentRunner:
    reg = build_default_registry(WorkspaceGuard(FIXTURE))
    return AgentRunner(llm, reg, ToolExecutor(reg), policy=policy, system_prompt="sys", **kw)


def test_max_steps_forces_final_synthesis_without_tools() -> None:
    llm = LoopingLLM(final="best effort answer")
    store = _Events()
    out = _runner(llm, AgentPolicy(max_agent_steps=3), history=SessionHistory(store)).run("go")
    assert out.stop_reason is StopReason.MAX_ITERATIONS
    assert out.final_text == "best effort answer"
    assert out.steps == 3
    assert llm.calls == 4  # 3 tool steps + 1 synthesis
    kinds = [e.kind for e in store.events]
    assert EventKind.STOPPED in kinds and EventKind.ASSISTANT_TURN in kinds


def test_tool_call_budget_is_a_hard_stop() -> None:
    llm = LoopingLLM()
    out = _runner(llm, AgentPolicy(max_agent_steps=20, max_tool_calls_per_session=2)).run("go")
    assert out.stop_reason is StopReason.MAX_TOOL_CALLS
    assert llm.calls <= 4


def test_cost_budget_is_a_hard_stop() -> None:
    llm = LoopingLLM(usage=Usage(prompt_tokens=1_000_000, completion_tokens=0))
    out = _runner(
        llm,
        AgentPolicy(max_agent_steps=20, max_cost_usd=0.10),
        model_name="gpt-4o-mini",
    ).run("go")
    assert out.stop_reason is StopReason.MAX_COST
    assert llm.calls == 1  # $0.15 spent on the first step trips the $0.10 cap


def test_session_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(agent_mod.time, "monotonic", lambda: next(clock))
    out = _runner(LoopingLLM(), AgentPolicy(max_agent_steps=5, session_timeout_s=10)).run("go")
    assert out.stop_reason is StopReason.SESSION_TIMEOUT


def test_stuck_detector_injects_hint_once() -> None:
    llm = LoopingLLM()
    events = list(_runner(llm, AgentPolicy(max_agent_steps=8)).run_events("go"))
    hints = [
        m
        for m in llm.seen[-1]
        if m.role == Role.USER and m.content.startswith("[SYSTEM — STUCK DETECTOR]")
    ]
    assert len(hints) == 1
    assert any(e.kind == "stuck" for e in events)
