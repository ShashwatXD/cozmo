"""Conversation memory - sliding window over chat turns."""

from __future__ import annotations

from cozmo.domain.messages import Message, Role


class ConversationMemory:
    """
    Stores non-system messages; for_prompt() returns system + trimmed history.
    Avoids starting the window on an orphan tool message.
    """

    def __init__(self, *, max_messages: int = 40) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be >= 2")
        self._max = max_messages
        self._history: list[Message] = []

    def __len__(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()

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
        return [
            Message(role=Role.SYSTEM, content=system_prompt),
            *self._window(self._history),
        ]

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
