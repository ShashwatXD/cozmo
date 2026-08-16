"""History RAG: index session turns and search_history tool."""

from pathlib import Path

from cozmo.app.history import SessionHistory
from cozmo.domain.events import EventKind, SessionEvent
from cozmo.infra.history.rag import HistoryRagIndex
from cozmo.infra.history.store import JsonlEventStore
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard


def test_history_rag_indexes_turns_incrementally(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    rag = HistoryRagIndex(tmp_path, StubEmbedder(dim=32), enabled=True)
    history = SessionHistory(store, rag_index=rag)

    history.session_start(workdir=str(tmp_path))
    history.user_turn("Why does shell need approval?")
    assert len(rag) >= 1
    first = len(rag)

    history.assistant_turn("PermissionGate asks before run_shell.")
    assert len(rag) > first

    # Unchanged re-sync adds nothing
    n = rag.sync_from_store(store, history.session_id)
    assert n == 0
    assert len(rag) > first


def test_history_rag_search_finds_prior_turn(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    rag = HistoryRagIndex(tmp_path, StubEmbedder(dim=32), enabled=True)
    history = SessionHistory(store, rag_index=rag)
    history.session_start()
    history.user_turn("off-by-one in math_utils add")
    history.assistant_turn("Confirmed off-by-one returns a - b")

    out = rag.search("math_utils off-by-one", top_k=3)
    assert "No history" not in out
    assert "math_utils" in out.lower() or "off-by-one" in out.lower()


def test_search_history_tool_registered(tmp_path: Path) -> None:
    rag = HistoryRagIndex(tmp_path, StubEmbedder(dim=32), enabled=True)
    rag.sync_session(
        "abc123",
        [
            SessionEvent(
                kind=EventKind.USER_TURN,
                session_id="abc123",
                data={"text": "shell permission always ask"},
            )
        ],
    )
    guard = WorkspaceGuard(tmp_path, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard, history_rag=rag)
    from cozmo.domain.tools import ToolCall
    from cozmo.infra.tools.registry import ToolExecutor

    ex = ToolExecutor(reg)
    tr = ex.execute(
        ToolCall(id="1", name="search_history", arguments='{"query":"permission"}')
    )
    assert not tr.is_error
    assert "permission" in tr.content.lower() or "shell" in tr.content.lower()


def test_catch_up_from_existing_jsonl(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    sid = "deadbeefcafe"
    # Write without going through SessionHistory rag
    store.append(
        SessionEvent(
            kind=EventKind.SESSION_START, session_id=sid, data={"workdir": str(tmp_path)}
        )
    )
    store.append(
        SessionEvent(
            kind=EventKind.USER_TURN,
            session_id=sid,
            data={"text": "hybrid RRF retrieval"},
        )
    )
    store.append(
        SessionEvent(
            kind=EventKind.ASSISTANT_TURN,
            session_id=sid,
            data={"text": "BM25 plus vectors fused with RRF"},
        )
    )

    rag = HistoryRagIndex(tmp_path, StubEmbedder(dim=32), enabled=True)
    n = rag.catch_up(store, limit=10)
    assert n >= 2
    assert len(rag) >= 2
