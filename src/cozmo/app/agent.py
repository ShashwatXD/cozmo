"""Agent runner - ReAct loop + guardrails + memory + optional history/tracing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from cozmo.app.compaction import compact_memory, needs_compaction
from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.app.permissions import PermissionGate
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.cost import estimate_cost_usd
from cozmo.domain.guardrails import AgentPolicy, StopReason
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.mode import MUTATING_TOOLS, READ_ONLY_MODES, AgentMode, prompt_name_for_mode
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolResult
from cozmo.infra.telemetry.tracer import Tracer
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt


@dataclass(frozen=True)
class AgentEvent:
    kind: str  # thinking | assistant | tool_call | tool_result | permission | compact | done | stopped
    text: str = ""
    tool_name: str = ""
    stop_reason: str = ""


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    usage: Usage
    steps: int
    stop_reason: StopReason = StopReason.COMPLETED


class AgentRunner:
    """
    ReAct loop: prior memory + user message -> LLM (+ tools) ->
    execute tool_calls -> observe -> repeat until final answer or policy kill.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        *,
        models: ModelRouter | None = None,
        memory: ConversationMemory | None = None,
        tracer: Tracer | None = None,
        history: SessionHistory | None = None,
        policy: AgentPolicy | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_steps: int | None = None,
        model_name: str = "stub-model",
        depth: int = 0,
        permission_gate: PermissionGate | None = None,
        mode: AgentMode = AgentMode.AGENT,
    ) -> None:
        if models is None:
            if llm is None:
                raise ValueError("AgentRunner requires llm or models")
            models = ModelRouter.from_single(llm)
        if registry is None or executor is None:
            raise ValueError("AgentRunner requires registry and executor")

        self._models = models
        self._llm = models.worker
        self._registry = registry
        self._executor = executor
        self._memory = memory if memory is not None else ConversationMemory()
        self._tracer = tracer
        self._history = history
        self._policy = policy or AgentPolicy(
            max_agent_steps=max_steps if max_steps is not None else 8
        )
        if max_steps is not None:
            from dataclasses import replace

            self._policy = replace(self._policy, max_agent_steps=max_steps)
        self._mode = mode
        self._permission_gate = permission_gate
        if self._permission_gate is not None:
            self._permission_gate.set_mode(mode)
        self._system_prompt = system_prompt or load_system_prompt(
            prompt_name_for_mode(mode)
        )
        self._temperature = temperature
        self._model_name = model_name
        self._depth = depth
        self._tool_calls = 0
        self._session_cost = 0.0
        self._session_started = time.monotonic()
        self.last_result: AgentResult | None = None

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def policy(self) -> AgentPolicy:
        return self._policy

    @property
    def mode(self) -> AgentMode:
        return self._mode

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def set_mode(self, mode: AgentMode) -> None:
        """Switch ask/plan/agent (prompt + tool visibility + permission gate)."""
        self._mode = mode
        self._system_prompt = load_system_prompt(prompt_name_for_mode(mode))
        if self._permission_gate is not None:
            self._permission_gate.set_mode(mode)

    def run(self, user_text: str) -> AgentResult:
        for _ in self.run_events(user_text):
            pass
        assert self.last_result is not None
        return self.last_result

    def run_events(self, user_text: str) -> Iterator[AgentEvent]:
        if self._history:
            self._history.user_turn(user_text)

        # Soft guardrail: compact before building the prompt when over budget.
        if needs_compaction(self._memory, self._policy):
            summary = compact_memory(
                self._memory,
                self._models.orchestrator,
                self._policy,
                temperature=self._temperature,
            )
            if summary:
                if self._history:
                    self._history.compact(summary)
                yield AgentEvent(kind="compact", text=summary[:200])

        messages: list[Message] = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        tools = self._tool_specs_for_mode()
        total_usage = Usage()
        self.last_result = None
        max_steps = self._policy.max_agent_steps

        for step in range(1, max_steps + 1):
            stop = self._check_hard_limits()
            if stop is not None:
                yield from self._finish_stopped(messages, total_usage, step - 1, stop)
                return

            yield AgentEvent(kind="thinking")
            t0 = time.perf_counter()
            result: CompletionResult = self._llm.complete(
                messages,
                temperature=self._temperature,
                tools=tools,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            total_usage = total_usage.merged(result.usage)
            self._accumulate_cost(result.usage)
            if self._tracer:
                self._tracer.emit(
                    "llm",
                    step=step,
                    latency_ms=round(latency_ms, 1),
                    tokens=result.usage.total_tokens,
                    tool_calls=[tc.name for tc in result.tool_calls],
                    finish_reason=result.finish_reason,
                )

            stop = self._policy.check_cost(self._session_cost)
            if stop is not None:
                yield from self._finish_stopped(
                    messages, total_usage, step, stop, draft=result.content or ""
                )
                return

            if result.content and result.has_tool_calls:
                yield AgentEvent(kind="assistant", text=result.content)

            if not result.has_tool_calls:
                messages.append(
                    Message(role=Role.ASSISTANT, content=result.content or "")
                )
                self._commit_memory(messages)
                text = result.content or ""
                self.last_result = AgentResult(
                    final_text=text,
                    usage=total_usage,
                    steps=step,
                    stop_reason=StopReason.COMPLETED,
                )
                if self._tracer:
                    self._tracer.emit(
                        "agent_done",
                        steps=step,
                        tokens=total_usage.total_tokens,
                    )
                if self._history:
                    self._history.assistant_turn(text, steps=step)
                    self._history.stopped(StopReason.COMPLETED, steps=step)
                yield AgentEvent(kind="done", text=text)
                return

            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=result.content or "",
                    tool_calls=result.tool_calls,
                )
            )

            for call in result.tool_calls:
                self._tool_calls += 1
                stop = self._policy.check_tool_calls(self._tool_calls)
                if stop is not None:
                    yield from self._finish_stopped(messages, total_usage, step, stop)
                    return

                yield AgentEvent(
                    kind="tool_call",
                    text=call.arguments,
                    tool_name=call.name,
                )
                t1 = time.perf_counter()
                tool_result = self._execute_gated(call)
                if tool_result.is_error and tool_result.content.startswith(
                    "Permission denied"
                ):
                    yield AgentEvent(
                        kind="permission",
                        text=tool_result.content,
                        tool_name=call.name,
                    )
                if self._tracer:
                    self._tracer.emit(
                        "tool",
                        name=call.name,
                        is_error=tool_result.is_error,
                        latency_ms=round((time.perf_counter() - t1) * 1000, 1),
                    )
                if self._history:
                    self._history.tool(
                        call.name,
                        is_error=tool_result.is_error,
                        preview=tool_result.content,
                    )
                yield AgentEvent(
                    kind="tool_result",
                    text=tool_result.content,
                    tool_name=call.name,
                )
                messages.append(
                    Message(
                        role=Role.TOOL,
                        content=tool_result.content,
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )

        # Max iterations — one last synthesis without tools, then kill reason.
        yield AgentEvent(kind="thinking")
        result = self._llm.complete(messages, temperature=self._temperature, tools=None)
        total_usage = total_usage.merged(result.usage)
        self._accumulate_cost(result.usage)
        text = result.content or "(max steps reached)"
        messages.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(messages)
        self.last_result = AgentResult(
            final_text=text,
            usage=total_usage,
            steps=max_steps,
            stop_reason=StopReason.MAX_ITERATIONS,
        )
        if self._tracer:
            self._tracer.emit(
                "agent_done",
                steps=max_steps,
                tokens=total_usage.total_tokens,
                max_steps=True,
            )
        if self._history:
            self._history.assistant_turn(text, steps=max_steps)
            self._history.stopped(StopReason.MAX_ITERATIONS, steps=max_steps)
        yield AgentEvent(
            kind="stopped",
            text=text,
            stop_reason=StopReason.MAX_ITERATIONS.value,
        )
        yield AgentEvent(kind="done", text=text)

    def _tool_specs_for_mode(self) -> list:
        specs = self._registry.specs()
        if self._mode in READ_ONLY_MODES:
            return [s for s in specs if s.name not in MUTATING_TOOLS]
        return specs

    def _execute_gated(self, call) -> ToolResult:
        if self._permission_gate is not None and call.name in MUTATING_TOOLS:
            decision = self._permission_gate.decide(call)
            if not decision.allowed:
                return self._permission_gate.deny_result(call, decision)
        return self._executor.execute(call)

    def _check_hard_limits(self) -> StopReason | None:
        elapsed = time.monotonic() - self._session_started
        return (
            self._policy.check_session_timeout(elapsed)
            or self._policy.check_tool_calls(self._tool_calls)
            or self._policy.check_cost(self._session_cost)
        )

    def _accumulate_cost(self, usage: Usage) -> None:
        cost = estimate_cost_usd(self._model_name, usage)
        if cost:
            self._session_cost += cost

    def _finish_stopped(
        self,
        messages: list[Message],
        total_usage: Usage,
        steps: int,
        reason: StopReason,
        *,
        draft: str = "",
    ) -> Iterator[AgentEvent]:
        text = draft or f"(stopped: {reason.value})"
        if draft:
            messages.append(Message(role=Role.ASSISTANT, content=draft))
            self._commit_memory(messages)
        self.last_result = AgentResult(
            final_text=text,
            usage=total_usage,
            steps=max(steps, 0),
            stop_reason=reason,
        )
        if self._history:
            self._history.stopped(reason, steps=steps)
        if self._tracer:
            self._tracer.emit("agent_done", steps=steps, stop_reason=reason.value)
        yield AgentEvent(kind="stopped", text=text, stop_reason=reason.value)
        yield AgentEvent(kind="done", text=text)

    def _commit_memory(self, messages: list[Message]) -> None:
        self._memory.replace_history(
            [m for m in messages if m.role != Role.SYSTEM]
        )
