"""Agent runner - ReAct loop + conversation memory + optional tracing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolResult
from cozmo.infra.telemetry.tracer import Tracer
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt

@dataclass(frozen=True)
class AgentEvent:
    kind: str  # thinking | assistant | tool_call | tool_result | done
    text: str = ""
    tool_name: str = ""

@dataclass(frozen=True)
class AgentResult:
    final_text: str
    usage: Usage
    steps: int

class AgentRunner:
    """
    ReAct loop: prior memory + user message -> LLM (+ tools) ->
    execute tool_calls -> observe -> repeat until final answer.
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        *,
        memory: ConversationMemory | None = None,
        tracer: Tracer | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_steps: int = 8,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._memory = memory if memory is not None else ConversationMemory()
        self._tracer = tracer
        self._system_prompt = system_prompt or load_system_prompt("agent")
        self._temperature = temperature
        self._max_steps = max_steps
        self.last_result: AgentResult | None = None

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def run(self, user_text: str) -> AgentResult:
        for _ in self.run_events(user_text):
            pass
        assert self.last_result is not None
        return self.last_result

    def run_events(self, user_text: str) -> Iterator[AgentEvent]:
        messages: list[Message] = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        tools = self._registry.specs()
        total_usage = Usage()
        self.last_result = None

        for step in range(1, self._max_steps + 1):
            yield AgentEvent(kind="thinking")
            t0 = time.perf_counter()
            result: CompletionResult = self._llm.complete(
                messages,
                temperature=self._temperature,
                tools=tools,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            total_usage = total_usage.merged(result.usage)
            if self._tracer:
                self._tracer.emit(
                    "llm",
                    step=step,
                    latency_ms=round(latency_ms, 1),
                    tokens=result.usage.total_tokens,
                    tool_calls=[tc.name for tc in result.tool_calls],
                    finish_reason=result.finish_reason,
                )

            if result.content and result.has_tool_calls:
                yield AgentEvent(kind="assistant", text=result.content)

            if not result.has_tool_calls:
                messages.append(
                    Message(role=Role.ASSISTANT, content=result.content or "")
                )
                self._commit_memory(messages)
                self.last_result = AgentResult(
                    final_text=result.content or "",
                    usage=total_usage,
                    steps=step,
                )
                if self._tracer:
                    self._tracer.emit(
                        "agent_done",
                        steps=step,
                        tokens=total_usage.total_tokens,
                    )
                yield AgentEvent(kind="done", text=result.content or "")
                return

            messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=result.content or "",
                    tool_calls=result.tool_calls,
                )
            )

            for call in result.tool_calls:
                yield AgentEvent(
                    kind="tool_call",
                    text=call.arguments,
                    tool_name=call.name,
                )
                t1 = time.perf_counter()
                tool_result: ToolResult = self._executor.execute(call)
                if self._tracer:
                    self._tracer.emit(
                        "tool",
                        name=call.name,
                        is_error=tool_result.is_error,
                        latency_ms=round((time.perf_counter() - t1) * 1000, 1),
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

        yield AgentEvent(kind="thinking")
        result = self._llm.complete(messages, temperature=self._temperature, tools=None)
        total_usage = total_usage.merged(result.usage)
        text = result.content or "(max steps reached)"
        messages.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(messages)
        self.last_result = AgentResult(
            final_text=text, usage=total_usage, steps=self._max_steps
        )
        if self._tracer:
            self._tracer.emit(
                "agent_done",
                steps=self._max_steps,
                tokens=total_usage.total_tokens,
                max_steps=True,
            )
        yield AgentEvent(kind="done", text=text)

    def _commit_memory(self, messages: list[Message]) -> None:
        self._memory.replace_history(
            [m for m in messages if m.role != Role.SYSTEM]
        )
