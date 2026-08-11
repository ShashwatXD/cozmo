"""Event / session history port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cozmo.domain.events import SessionEvent


@runtime_checkable
class EventStore(Protocol):
    """Append-only session event history (JSONL on disk in infra)."""

    def append(self, event: SessionEvent) -> None:
        ...

    def list_events(self, session_id: str) -> list[SessionEvent]:
        ...

    def list_sessions(self, *, limit: int = 20) -> list[dict]:
        """Newest-first session index rows: id, started_at, workdir, …"""
        ...
