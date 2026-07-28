"""
Agent runner — ReAct loop + conversation memory.

What: think → tool calls → observe → repeat; persist turns across prompts.
Why: tools alone do nothing; memory makes multi-turn coding sessions work.
Layer: app (ViewModel).
Flutter: Cubit with List<Message> state + maxSteps loop.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolResult
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt


@dataclass(frozen=True)
class AgentEvent:
    """Flutter: state emissions the UI listens to."""

    kind: str  # assistant | tool_call | tool_result | done
    text: str = ""
    tool_name: str = ""


@dataclass(frozen=True)
class AgentResult:
    final_text: str
    usage: Usage
    steps: int


class AgentRunner:
    """
    ReAct-style loop with optional multi-turn memory:

      prior memory + new user message
        → LLM (with tool schemas)
        → if tool_calls: execute → append results → LLM again
        → else: final answer → save history into memory
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        *,
        memory: ConversationMemory | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_steps: int = 8,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._memory = memory if memory is not None else ConversationMemory()
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
        # Flutter: state.messages + new user bubble
        messages: list[Message] = [
            *self._memory.for_prompt(self._system_prompt),
            Message(role=Role.USER, content=user_text),
        ]
        tools = self._registry.specs()
        total_usage = Usage()
        self.last_result = None

        for step in range(1, self._max_steps + 1):
            result: CompletionResult = self._llm.complete(
                messages,
                temperature=self._temperature,
                tools=tools,
            )
            total_usage = total_usage.merged(result.usage)

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
                tool_result: ToolResult = self._executor.execute(call)
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

        result = self._llm.complete(messages, temperature=self._temperature, tools=None)
        total_usage = total_usage.merged(result.usage)
        text = result.content or "(max steps reached)"
        messages.append(Message(role=Role.ASSISTANT, content=text))
        self._commit_memory(messages)
        self.last_result = AgentResult(
            final_text=text, usage=total_usage, steps=self._max_steps
        )
        yield AgentEvent(kind="done", text=text)

    def _commit_memory(self, messages: list[Message]) -> None:
        """Persist non-system messages so the next user turn remembers this one."""
        self._memory.replace_history(
            [m for m in messages if m.role != Role.SYSTEM]
        )
