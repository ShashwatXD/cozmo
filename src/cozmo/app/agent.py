"""Agent runner - ReAct loop + guardrails + memory + optional history/tracing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from cozmo.app.compaction import compact_memory, needs_compaction
from cozmo.app.context_editing import clear_old_tool_results
from cozmo.app.groundedness import contexts_from_messages, score_groundedness
from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.app.permissions import PermissionGate
from cozmo.app.stuck_detector import detect_stuck, step_from_tool_call
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.cost import estimate_cost_usd
from cozmo.domain.guardrails import AgentPolicy, StopReason
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.mode import MUTATING_TOOLS, READ_ONLY_MODES, AgentMode, prompt_name_for_mode
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolResult
from cozmo.infra.mcp.types import is_mcp_tool
from cozmo.infra.telemetry.tracer import Tracer, clip, snapshot_messages
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt


@dataclass(frozen=True)
class AgentEvent:
    kind: str  # thinking | assistant | tool_call | tool_result | permission | context | compact | done | stopped
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
        self._system_prompt = system_prompt or load_system_prompt(prompt_name_for_mode(mode))
        self._temperature = temperature
        self._model_name = model_name
        self._depth = depth
        self._tool_calls = 0
        self._session_cost = 0.0
        self._turn_cost = 0.0
        self._turn_started = 0.0
        self.last_result: AgentResult | None = None
        self._stuck_steps: list = []
        self._stuck_warned = False

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
        self._turn_started = time.monotonic()
        self._turn_cost = 0.0
        if self._history:
            self._history.user_turn(user_text)
        if self._tracer:
            self._tracer.start_turn(
                user_text, mode=self._mode.value, model=self._model_name, depth=self._depth
            )

        # Cheapest first: old tool results become one-line notes (no model call).
        trimmed, report = clear_old_tool_results(
            self._memory.snapshot(), keep_recent=self._policy.keep_tool_results
        )
        if report.cleared:
            self._memory.replace_history(trimmed)
            if self._tracer:
                self._tracer.emit(
                    "context_cleared", results=report.cleared, chars=report.chars_freed
                )
            yield AgentEvent(
                kind="context",
                text=f"cleared {report.cleared} old tool result(s), ~{report.chars_freed // 4:,} tokens",
            )

        # Then, only if still over budget: summarize older turns with the orchestrator model.
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
                if self._tracer:
                    self._tracer.emit("compact", summary=clip(summary, 4_000))
                yield AgentEvent(kind="compact", text=summary[:200])

        messages: list[Message] = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        tools = self._tool_specs_for_mode()
        total_usage = Usage()
        self.last_result = None
        self._stuck_steps = []
        self._stuck_warned = False
        max_steps = self._policy.max_agent_steps

        for step in range(1, max_steps + 1):
            stop = self._check_hard_limits()
            if stop is not None:
                yield from self._finish_stopped(messages, total_usage, step - 1, stop)
                return

            if not self._stuck_warned:
                verdict = detect_stuck(self._stuck_steps)
                if verdict.stuck:
                    self._stuck_warned = True
                    hint = f"[SYSTEM — STUCK DETECTOR]\n{verdict.message}"
                    messages.append(Message(role=Role.USER, content=hint))
                    if self._tracer:
                        self._tracer.emit(
                            "stuck",
                            reason=verdict.reason,
                            message=verdict.message,
                        )
                    yield AgentEvent(
                        kind="stuck",
                        text=verdict.message,
                        stop_reason=verdict.reason,
                    )

            yield AgentEvent(kind="thinking")
            sent = snapshot_messages(messages) if self._tracer else []
            t0 = time.perf_counter()
            result: CompletionResult = self._llm.complete(
                messages,
                temperature=self._temperature,
                tools=tools,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            total_usage = total_usage.merged(result.usage)
            cost = self._accumulate_cost(result.usage)
            self._trace_llm(step, sent, tools, result, latency_ms, cost)

            stop = self._policy.check_cost(self._session_cost)
            if stop is not None:
                yield from self._finish_stopped(
                    messages, total_usage, step, stop, draft=result.content or ""
                )
                return

            if result.content and result.has_tool_calls:
                yield AgentEvent(kind="assistant", text=result.content)

            if not result.has_tool_calls:
                messages.append(Message(role=Role.ASSISTANT, content=result.content or ""))
                self._commit_memory(messages)
                text = result.content or ""
                self.last_result = AgentResult(
                    final_text=text,
                    usage=total_usage,
                    steps=step,
                    stop_reason=StopReason.COMPLETED,
                )
                self._trace_done(StopReason.COMPLETED, step, total_usage, text, messages)
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
                if tool_result.is_error and tool_result.content.startswith("Permission denied"):
                    yield AgentEvent(
                        kind="permission",
                        text=tool_result.content,
                        tool_name=call.name,
                    )
                if self._tracer:
                    self._tracer.emit(
                        "tool",
                        step=step,
                        name=call.name,
                        arguments=clip(call.arguments, 4_000),
                        result=clip(tool_result.content),
                        result_chars=len(tool_result.content),
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
                self._stuck_steps.append(
                    step_from_tool_call(
                        tool_name=call.name,
                        args=call.arguments,
                        result=tool_result.content[:500],
                        is_error=tool_result.is_error,
                    )
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
        sent = snapshot_messages(messages) if self._tracer else []
        t0 = time.perf_counter()
        result = self._llm.complete(messages, temperature=self._temperature, tools=None)
        latency_ms = (time.perf_counter() - t0) * 1000
        total_usage = total_usage.merged(result.usage)
        cost = self._accumulate_cost(result.usage)
        self._trace_llm(max_steps + 1, sent, None, result, latency_ms, cost)
        text = result.content or "(max steps reached)"
        messages.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(messages)
        self.last_result = AgentResult(
            final_text=text,
            usage=total_usage,
            steps=max_steps,
            stop_reason=StopReason.MAX_ITERATIONS,
        )
        self._trace_done(StopReason.MAX_ITERATIONS, max_steps, total_usage, text, messages)
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
            return [s for s in specs if s.name not in MUTATING_TOOLS and not is_mcp_tool(s.name)]
        return specs

    def _execute_gated(self, call) -> ToolResult:
        if self._permission_gate is not None and call.name in MUTATING_TOOLS:
            decision = self._permission_gate.decide(call)
            if not decision.allowed:
                return self._permission_gate.deny_result(call, decision)
        return self._executor.execute(call)

    def _check_hard_limits(self) -> StopReason | None:
        elapsed = time.monotonic() - self._turn_started
        return (
            self._policy.check_session_timeout(elapsed)
            or self._policy.check_tool_calls(self._tool_calls)
            or self._policy.check_cost(self._session_cost)
        )

    def _accumulate_cost(self, usage: Usage) -> float:
        cost = estimate_cost_usd(self._model_name, usage) or 0.0
        self._session_cost += cost
        self._turn_cost += cost
        return cost

    def _trace_llm(self, step, sent, tools, result, latency_ms: float, cost: float) -> None:
        """One model request with everything sent and received."""
        if not self._tracer:
            return
        self._tracer.emit(
            "llm",
            step=step,
            model=self._model_name,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            tokens=result.usage.total_tokens,
            cost_usd=round(cost, 6),
            finish_reason=result.finish_reason,
            messages=sent,
            tools=[t.name for t in tools or []],
            response=clip(result.content),
            tool_calls=[
                {"name": tc.name, "arguments": clip(tc.arguments, 2_000)}
                for tc in result.tool_calls
            ],
        )

    def _trace_done(self, reason, steps: int, usage: Usage, text: str, messages) -> None:
        """How the turn ended, with a groundedness check of the answer against what was read."""
        if not self._tracer:
            return
        data: dict = {
            "stop_reason": reason.value,
            "steps": steps,
            "tokens": usage.total_tokens,
            "cost_usd": round(self._turn_cost, 6),
            "duration_s": round(time.monotonic() - self._turn_started, 2),
            "answer": clip(text, 4_000),
        }
        # Only this turn's tool results count as evidence for this turn's answer.
        last_user = max((i for i, m in enumerate(messages) if m.role == Role.USER), default=0)
        contexts = contexts_from_messages(messages[last_user:])
        if contexts and text:
            report = score_groundedness(answer=text, contexts=contexts)
            invalid = report.citations.get("invalid", [])
            data["groundedness"] = {
                "score": report.score,
                "summary": report.summary,
                "unsupported": [clip(u, 300) for u in report.unsupported[:5]],
                "invalid_citations": len(invalid) if isinstance(invalid, list) else 0,
            }
        self._tracer.emit("agent_done", **data)

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
        self._trace_done(reason, steps, total_usage, text, messages)
        yield AgentEvent(kind="stopped", text=text, stop_reason=reason.value)
        yield AgentEvent(kind="done", text=text)

    def _commit_memory(self, messages: list[Message]) -> None:
        self._memory.replace_history([m for m in messages if m.role != Role.SYSTEM])
