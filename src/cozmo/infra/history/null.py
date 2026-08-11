"""Null / no-op event store for tests and disabled history."""

from __future__ import annotations

from typing import Any

from cozmo.domain.events import SessionEvent


class NullEventStore:
    def append(self, event: SessionEvent) -> None:
        return

    def list_events(self, session_id: str) -> list[SessionEvent]:
        return []

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return []
