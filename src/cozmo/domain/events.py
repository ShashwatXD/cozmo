"""Session event records — pure data, no I/O."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_TURN = "user_turn"
    ASSISTANT_TURN = "assistant_turn"
    TOOL = "tool"
    COMPACT = "compact"
    STOPPED = "stopped"
    SUBAGENT = "subagent"


@dataclass(frozen=True)
class SessionEvent:
    kind: EventKind
    session_id: str
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "session_id": self.session_id,
            "ts": self.ts,
            "data": self.data,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SessionEvent:
        return SessionEvent(
            kind=EventKind(raw["kind"]),
            session_id=str(raw["session_id"]),
            ts=float(raw.get("ts", time.time())),
            data=dict(raw.get("data") or {}),
        )


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
