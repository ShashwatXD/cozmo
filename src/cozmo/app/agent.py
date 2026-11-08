"""Agent runner - ReAct loop + guardrails + memory + optional history/tracing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from cozmo.app.compaction import compact_memory, needs_compaction, shrink_tool_results
from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.app.permissions import PermissionGate
from cozmo.app.stuck_detector import detect_stuck, step_from_tool_call
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.cost import UsageLedger
from cozmo.domain.guardrails import AgentPolicy, StopReason
from cozmo.domain.memory import ConversationMemory, drop_incomplete_tool_group
from cozmo.domain.messages import Message, Role
from cozmo.domain.mode import (
    READ_ONLY_MODES,
    AgentMode,
    prompt_name_for_mode,
    requires_approval,
)
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolResult
from cozmo.infra.history.checkpoint import CheckpointStore
from cozmo.infra.telemetry.tracer import Tracer, safe_args
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt

# After the stuck hint, the model gets this many tool calls to change course.
_STUCK_GRACE_CALLS = 2


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
        checkpoints: CheckpointStore | None = None,
    ) -> None:
        if models is None:
            if llm is None:
                raise ValueError("AgentRunner requires llm or models")
            models = ModelRouter.from_single(llm)
        if registry is None or executor is None:
            raise ValueError("AgentRunner requires registry and executor")

        self._tracer = tracer if tracer is not None else Tracer(None, enabled=False)
        if models.ledger is None:
            models = models.metered(UsageLedger(), model_name, self._tracer)
        assert models.ledger is not None
        self._ledger = models.ledger
        self._models = models
        self._llm = models.worker
        self._registry = registry
        self._executor = executor
        self._memory = memory if memory is not None else ConversationMemory()
        self._history = history
        self._checkpoints = checkpoints
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
        self._turn_started = 0.0
        self._inflight: list[Message] = []
        self.last_result: AgentResult | None = None
        self._stuck_steps: list = []
        self._stuck_warned_at: int | None = None

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def ledger(self) -> UsageLedger:
        return self._ledger

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
        """Run one turn. The time and tool budgets are per turn; cost spans the session."""
        self._turn_started = time.monotonic()
        self._tool_calls = 0
        self._inflight = []
        self.last_result = None
        cost_before = self._ledger.total_cost
        span = self._tracer.start("turn", mode=self._mode.value, depth=self._depth)
        status = "ok"
        try:
            yield from self._run_turn(user_text)
        except (Exception, KeyboardInterrupt) as exc:
            status = "error"
            reason = (
                StopReason.INTERRUPTED
                if isinstance(exc, KeyboardInterrupt)
                else StopReason.PROVIDER_ERROR
            )
            span.set(stop_reason=reason.value, error_type=type(exc).__name__)
            self._commit_memory(drop_incomplete_tool_group(self._inflight))
            if self._history:
                self._history.stopped(reason)
            self._clear_checkpoint()
            raise
        finally:
            result = self.last_result
            if result is not None:
                span.set(
                    stop_reason=result.stop_reason.value,
                    steps=result.steps,
                    tool_calls=self._tool_calls,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                    cost_usd=round(self._ledger.total_cost - cost_before, 6),
                )
            self._tracer.end(span, status=status)

    def _run_turn(self, user_text: str) -> Iterator[AgentEvent]:
        if self._history:
            self._history.user_turn(user_text)

        # Soft guardrail: compact before building the prompt when over budget.
        if needs_compaction(self._memory, self._policy):
            with self._tracer.span("compaction"):
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
        self._inflight = messages
        tools = self._tool_specs_for_mode()
        total_usage = Usage()
        self._stuck_steps = []
        self._stuck_warned_at = None
        max_steps = self._policy.max_agent_steps
        partial_text: list[str] = []
        nudged_empty = False

        for step in range(1, max_steps + 1):
            stop = self._check_hard_limits()
            if stop is not None:
                yield from self._finish_stopped(messages, total_usage, step - 1, stop)
                return

            verdict = detect_stuck(self._stuck_steps)
            if verdict.stuck and self._stuck_warned_at is None:
                self._stuck_warned_at = len(self._stuck_steps)
                hint = f"[SYSTEM — STUCK DETECTOR]\n{verdict.message}"
                messages.append(Message(role=Role.USER, content=hint))
                self._tracer.emit("stuck", reason=verdict.reason)
                yield AgentEvent(kind="stuck", text=verdict.message, stop_reason=verdict.reason)
            elif (
                verdict.stuck
                and self._stuck_warned_at is not None
                and len(self._stuck_steps) - self._stuck_warned_at >= _STUCK_GRACE_CALLS
            ):
                yield from self._finish_stopped(messages, total_usage, step - 1, StopReason.STUCK)
                return

            # Free, deterministic trim first. The model-written summary runs between turns.
            shrunk, elided = shrink_tool_results(messages, self._policy)
            if elided:
                messages[:] = shrunk
                self._tracer.emit("context_shrink", elided=elided, step=step)

            yield AgentEvent(kind="thinking")
            result: CompletionResult = self._llm.complete(
                messages,
                temperature=self._temperature,
                tools=tools,
            )
            total_usage = total_usage.merged(result.usage)

            stop = self._check_cost()
            if stop is not None:
                yield from self._finish_stopped(
                    messages, total_usage, step, stop, draft=result.content or ""
                )
                return

            if result.truncated:
                # A cut-off tool call has invalid arguments and must not run.
                self._tracer.emit("truncated", step=step, had_tool_calls=result.has_tool_calls)
                if result.content:
                    partial_text.append(result.content)
                messages.append(Message(role=Role.ASSISTANT, content=result.content or "(cut off)"))
                note = (
                    "[SYSTEM — OUTPUT LIMIT] Your reply hit the output limit and no tool ran. "
                    "Make smaller changes with edit_file."
                    if result.has_tool_calls
                    else "[SYSTEM — OUTPUT LIMIT] Continue exactly where you stopped."
                )
                messages.append(Message(role=Role.USER, content=note))
                yield AgentEvent(kind="truncated", text=note)
                continue

            if not result.content and not result.has_tool_calls and not nudged_empty:
                nudged_empty = True
                self._tracer.emit("empty_response", step=step)
                messages.append(
                    Message(
                        role=Role.USER,
                        content="[SYSTEM] Your reply was empty. Answer or call a tool.",
                    )
                )
                continue

            if result.content and result.has_tool_calls:
                yield AgentEvent(kind="assistant", text=result.content)

            if not result.has_tool_calls:
                text = "".join(partial_text) + (result.content or "")
                text = text or "(the model returned no answer)"
                messages.append(Message(role=Role.ASSISTANT, content=text))
                self._commit_memory(messages)
                self._clear_checkpoint()
                self.last_result = AgentResult(
                    final_text=text,
                    usage=total_usage,
                    steps=step,
                    stop_reason=StopReason.COMPLETED,
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
                with self._tracer.span(
                    "tool", name=call.name, args=safe_args(call.arguments)
                ) as ts:
                    tool_result = self._execute_gated(call)
                    ts.set(
                        is_error=tool_result.is_error,
                        result_chars=len(tool_result.content),
                        truncated="...[truncated" in tool_result.content,
                    )
                    if tool_result.is_error:
                        ts.status = "tool_error"
                if tool_result.is_error and tool_result.content.startswith("Permission denied"):
                    yield AgentEvent(
                        kind="permission",
                        text=tool_result.content,
                        tool_name=call.name,
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
            self._save_checkpoint(messages)

        # Max iterations — one last synthesis without tools, then kill reason.
        stop = self._check_hard_limits()
        if stop is not None:
            yield from self._finish_stopped(messages, total_usage, max_steps, stop)
            return
        yield AgentEvent(kind="thinking")
        result = self._llm.complete(messages, temperature=self._temperature, tools=None)
        total_usage = total_usage.merged(result.usage)
        text = result.content or "(max steps reached)"
        messages.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(messages)
        self._clear_checkpoint()
        self.last_result = AgentResult(
            final_text=text,
            usage=total_usage,
            steps=max_steps,
            stop_reason=StopReason.MAX_ITERATIONS,
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
            return [s for s in specs if not requires_approval(s.name)]
        return specs

    def _execute_gated(self, call) -> ToolResult:
        if requires_approval(call.name):
            # Hiding a tool from the model is not enforcement. Refuse the call as well.
            if self._mode in READ_ONLY_MODES:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"Permission denied: {self._mode.value} mode blocks {call.name}",
                    is_error=True,
                )
            if self._permission_gate is not None:
                with self._tracer.span("permission", tool=call.name) as span:
                    decision = self._permission_gate.decide(call)
                    span.set(
                        allowed=decision.allowed,
                        choice=decision.choice.value,
                        reason=decision.reason,
                    )
                if not decision.allowed:
                    return self._permission_gate.deny_result(call, decision)
            if self._history:
                self._history.tool_intent(call.name, args=safe_args(call.arguments))
        return self._executor.execute(call)

    def _check_hard_limits(self) -> StopReason | None:
        elapsed = time.monotonic() - self._turn_started
        return (
            self._policy.check_session_timeout(elapsed)
            or self._policy.check_tool_calls(self._tool_calls)
            or self._check_cost()
        )

    def _check_cost(self) -> StopReason | None:
        # A cap cannot be enforced on spend Cozmo has no price for, so fail closed.
        if self._policy.max_cost_usd is not None and self._ledger.unpriced:
            return StopReason.COST_UNKNOWN
        return self._policy.check_cost(self._ledger.total_cost)

    def _save_checkpoint(self, messages: list[Message]) -> None:
        if self._checkpoints is not None and self._history is not None:
            self._checkpoints.save(self._history.session_id, messages)

    def _clear_checkpoint(self) -> None:
        if self._checkpoints is not None and self._history is not None:
            self._checkpoints.clear(self._history.session_id)

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
        if reason is StopReason.COST_UNKNOWN:
            models = ", ".join(sorted(self._ledger.unpriced))
            text = f"(stopped: no price for {models}. Add it to model_prices or unset max_cost_usd)"
        # Keep the turn so the next message has context, minus any unanswered tool calls.
        kept = drop_incomplete_tool_group(messages)
        kept.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(kept)
        self._clear_checkpoint()
        self.last_result = AgentResult(
            final_text=text,
            usage=total_usage,
            steps=max(steps, 0),
            stop_reason=reason,
        )
        if self._history:
            self._history.stopped(reason, steps=steps)
        yield AgentEvent(kind="stopped", text=text, stop_reason=reason.value)
        yield AgentEvent(kind="done", text=text)

    def _commit_memory(self, messages: list[Message]) -> None:
        self._memory.replace_history([m for m in messages if m.role != Role.SYSTEM])
