"""Session event shapes."""

from cozmo.domain.events import EventKind, SessionEvent, new_session_id


def test_session_event_roundtrip() -> None:
    sid = new_session_id()
    ev = SessionEvent(kind=EventKind.USER_TURN, session_id=sid, data={"text": "hi"})
    raw = ev.to_dict()
    back = SessionEvent.from_dict(raw)
    assert back.kind == EventKind.USER_TURN
    assert back.session_id == sid
    assert back.data["text"] == "hi"
