"""Conversation memory - sliding window + optional rolling summary."""

from __future__ import annotations

from cozmo.domain.messages import Message, Role


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
