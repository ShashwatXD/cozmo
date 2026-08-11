"""JSONL event history store."""

from pathlib import Path

from cozmo.app.history import SessionHistory
from cozmo.domain.events import EventKind
from cozmo.infra.history.store import JsonlEventStore


def test_jsonl_event_store_append_and_list(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True, max_sessions=10)
    hist = SessionHistory(store, session_id="abc123")
    hist.session_start(model="stub")
    hist.user_turn("hello")
    hist.assistant_turn("world")
    hist.stopped("completed")

    events = store.list_events("abc123")
    kinds = [e.kind for e in events]
    assert EventKind.SESSION_START in kinds
    assert EventKind.USER_TURN in kinds
    assert EventKind.ASSISTANT_TURN in kinds

    sessions = store.list_sessions(limit=5)
    assert any(s.get("id") == "abc123" for s in sessions)
    assert (tmp_path / ".cozmo" / "history" / "abc123.jsonl").is_file()
