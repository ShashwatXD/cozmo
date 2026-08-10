"""Stub LLM - fake provider for wiring + tests."""

from __future__ import annotations

from collections.abc import Iterator

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall, ToolSpec

class StubLLMClient:
    """

    Scripted tool calls apply to each new user turn:
    - first complete after a USER → return tool_calls (if scripted)
    - after TOOL results for that turn → return final_text
    - otherwise echo
    """

    def __init__(
        self,
        *,
        scripted_tool_calls: tuple[ToolCall, ...] | None = None,
        final_text: str = "done",
    ) -> None:
        self._scripted = scripted_tool_calls
        self._final_text = final_text

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        after_user = _messages_since_last_user(messages)

        # Tool results already in for this turn → finish
        if any(m.role == Role.TOOL for m in after_user):
            return CompletionResult(
                content=self._final_text,
                usage=Usage(),
                finish_reason="stop",
            )

        # Fresh user turn + scripted tools → ask for tools
        if self._scripted and not any(
            m.role == Role.ASSISTANT and m.tool_calls for m in after_user
        ):
            return CompletionResult(
                content="",
                usage=Usage(),
                finish_reason="tool_calls",
                tool_calls=self._scripted,
            )

        last = messages[-1].content if messages else ""
        if json_mode:
            content = '{"ok": true, "echo": ' + repr(last[:80]) + "}"
        else:
            content = (
                f"[cozmo stub] Got your message ({len(last)} chars). "
                "Set COZMO_PROVIDER=openai or ollama for a real model."
            )
        return CompletionResult(
            content=content,
            usage=Usage(),
            finish_reason="stop",
        )

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        result = self.complete(messages, temperature=temperature)
        for word in (result.content or "ok").split(" "):
            yield word + " "

def _messages_since_last_user(messages: list[Message]) -> list[Message]:
    last_user = -1
    for i, m in enumerate(messages):
        if m.role == Role.USER:
            last_user = i
    if last_user < 0:
        return []
    return messages[last_user + 1 :]
