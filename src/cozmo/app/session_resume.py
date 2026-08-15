"""Resume ConversationMemory from JSONL session audit events."""

from __future__ import annotations

from cozmo.domain.events import EventKind, SessionEvent
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role


def hydrate_memory_from_events(
    events: list[SessionEvent],
    memory: ConversationMemory | None = None,
) -> ConversationMemory:
    """
    Rebuild a lossy transcript for continue.

    Uses the latest compact summary (if any), then replays user/assistant
    turns after that compact. Tool events are audit-only and skipped.
    """
    mem = memory if memory is not None else ConversationMemory()
    mem.clear()

    last_compact = -1
    last_summary = ""
    for i, event in enumerate(events):
        if event.kind == EventKind.COMPACT:
            last_compact = i
            last_summary = str(event.data.get("summary") or "").strip()

    if last_summary:
        mem.summary = last_summary

    start = last_compact + 1 if last_compact >= 0 else 0
    for event in events[start:]:
        if event.kind == EventKind.USER_TURN:
            text = str(event.data.get("text") or "")
            if text:
                mem.add(Message(role=Role.USER, content=text))
        elif event.kind == EventKind.ASSISTANT_TURN:
            text = str(event.data.get("text") or "")
            if text:
                mem.add(Message(role=Role.ASSISTANT, content=text))
    return mem


def session_preview(events: list[SessionEvent], *, max_len: int = 72) -> str:
    """First user turn snippet for /sessions listing."""
    for event in events:
        if event.kind == EventKind.USER_TURN:
            text = str(event.data.get("text") or "").strip().replace("\n", " ")
            if len(text) > max_len:
                return text[: max_len - 1] + "…"
            return text
    return ""
