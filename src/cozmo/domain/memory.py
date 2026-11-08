"""Conversation memory - sliding window + optional rolling summary."""

from __future__ import annotations

from cozmo.domain.messages import Message, Role


def drop_incomplete_tool_group(messages: list[Message]) -> list[Message]:
    """
    Remove a trailing assistant tool-call group that is missing results.

    Providers reject a history where a tool call has no matching result, so a run
    that stops mid-batch must not leave one behind.
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == Role.ASSISTANT and messages[i].tool_calls:
            wanted = {tc.id for tc in messages[i].tool_calls}
            answered = {m.tool_call_id for m in messages[i + 1 :] if m.role == Role.TOOL}
            return messages[:i] if wanted - answered else list(messages)
    return list(messages)


class ConversationMemory:
    """
    Stores non-system messages; for_prompt() returns system + summary + trimmed history.
    Avoids starting the window on an orphan tool message.
    """

    def __init__(self, *, max_messages: int = 40) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be >= 2")
        self._max = max_messages
        self._history: list[Message] = []
        self._summary: str | None = None

    def __len__(self) -> int:
        return len(self._history)

    @property
    def summary(self) -> str | None:
        return self._summary

    @summary.setter
    def summary(self, value: str | None) -> None:
        self._summary = value.strip() if value and value.strip() else None

    def clear(self) -> None:
        self._history.clear()
        self._summary = None

    def add(self, message: Message) -> None:
        if message.role == Role.SYSTEM:
            raise ValueError("System prompts are not stored - pass via for_prompt()")
        self._history.append(message)

    def extend(self, messages: list[Message]) -> None:
        for m in messages:
            self.add(m)

    def replace_history(self, messages: list[Message]) -> None:
        self._history = [m for m in messages if m.role != Role.SYSTEM]

    def snapshot(self) -> list[Message]:
        return list(self._history)

    def for_prompt(self, system_prompt: str) -> list[Message]:
        out = [Message(role=Role.SYSTEM, content=system_prompt)]
        if self._summary:
            out.append(
                Message(
                    role=Role.SYSTEM,
                    content=f"Session summary so far:\n{self._summary}",
                )
            )
        out.extend(self._window(self._history))
        return out

    def _window(self, history: list[Message]) -> list[Message]:
        if len(history) <= self._max:
            return list(history)
        start = len(history) - self._max
        while start > 0 and history[start].role == Role.TOOL:
            start -= 1
        window = history[start:]
        while len(window) > self._max and window:
            window = window[1:]
            while window and window[0].role == Role.TOOL:
                window = window[1:]
        return window
