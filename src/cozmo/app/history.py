"""Record session events through the EventStore port."""

from __future__ import annotations

from typing import Any

from cozmo.domain.events import EventKind, SessionEvent, new_session_id
from cozmo.domain.guardrails import StopReason
from cozmo.domain.ports_history import EventStore


class SessionHistory:
    def __init__(self, store: EventStore, *, session_id: str | None = None) -> None:
        self._store = store
        self.session_id = session_id or new_session_id()

    def emit(self, kind: EventKind, **data: Any) -> SessionEvent:
        event = SessionEvent(kind=kind, session_id=self.session_id, data=dict(data))
        self._store.append(event)
        return event

    def session_start(self, **data: Any) -> SessionEvent:
        return self.emit(EventKind.SESSION_START, **data)

    def session_end(self, **data: Any) -> SessionEvent:
        return self.emit(EventKind.SESSION_END, **data)

    def user_turn(self, text: str) -> SessionEvent:
        return self.emit(EventKind.USER_TURN, text=text[:4000])

    def assistant_turn(self, text: str, **data: Any) -> SessionEvent:
        return self.emit(EventKind.ASSISTANT_TURN, text=text[:4000], **data)

    def tool(self, name: str, *, is_error: bool = False, preview: str = "") -> SessionEvent:
        return self.emit(
            EventKind.TOOL,
            name=name,
            is_error=is_error,
            preview=preview[:500],
        )

    def compact(self, summary: str) -> SessionEvent:
        return self.emit(EventKind.COMPACT, summary=summary[:2000])

    def stopped(self, reason: StopReason | str, **data: Any) -> SessionEvent:
        value = reason.value if isinstance(reason, StopReason) else str(reason)
        return self.emit(EventKind.STOPPED, reason=value, **data)

    def subagent(self, goal: str, **data: Any) -> SessionEvent:
        return self.emit(EventKind.SUBAGENT, goal=goal[:1000], **data)

    def list_recent_sessions(self, *, limit: int = 10) -> list[dict]:
        from cozmo.app.session_resume import session_preview

        rows = self._store.list_sessions(limit=limit)
        out: list[dict] = []
        for row in rows:
            enriched = dict(row)
            if not enriched.get("preview"):
                events = self._store.list_events(str(row.get("id") or ""))
                enriched["preview"] = session_preview(events)
            out.append(enriched)
        return out

    def list_events(self, session_id: str | None = None) -> list[SessionEvent]:
        sid = session_id or self.session_id
        return self._store.list_events(sid)

    def attach(self, session_id: str) -> None:
        """Continue writing events into an existing session file."""
        self.session_id = session_id

    def most_recent_session_id(self) -> str | None:
        fn = getattr(self._store, "most_recent_session_id", None)
        if callable(fn):
            return fn()
        rows = self._store.list_sessions(limit=1)
        if not rows:
            return None
        return str(rows[0].get("id") or "") or None
