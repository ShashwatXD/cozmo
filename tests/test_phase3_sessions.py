"""Ask mode + session resume/export (Phase 3)."""

import json
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.app.history import SessionHistory
from cozmo.app.permissions import PermissionChoice, PermissionGate
from cozmo.app.session_export import export_session_json, export_session_markdown
from cozmo.app.session_resume import hydrate_memory_from_events, session_preview
from cozmo.domain.events import EventKind
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.mode import AgentMode
from cozmo.domain.tools import ToolCall
from cozmo.infra.history.store import JsonlEventStore
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permission_store import PermissionStore
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.prompts.loader import load_system_prompt


def test_ask_mode_hides_mutating_tools(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path, allow_write=True, allow_shell=True)
    reg = build_default_registry(guard)
    gate = PermissionGate(
        workdir=tmp_path,
        store=PermissionStore.load(tmp_path),
        mode=AgentMode.ASK,
        ask=lambda _c, _p: PermissionChoice.ALLOW_ONCE,
    )
    runner = AgentRunner(
        StubLLMClient(final_text="explained"),
        reg,
        ToolExecutor(reg),
        permission_gate=gate,
        mode=AgentMode.ASK,
        max_steps=2,
    )
    names = {s.name for s in runner._tool_specs_for_mode()}
    assert "write_file" not in names
    assert "run_shell" not in names
    assert "read_file" in names
    assert "ask" in runner.system_prompt.lower() or "ASK" in load_system_prompt("ask")


def test_ask_mode_gate_denies_write(tmp_path: Path) -> None:
    gate = PermissionGate(
        workdir=tmp_path,
        store=PermissionStore.load(tmp_path),
        mode=AgentMode.ASK,
        ask=lambda _c, _p: PermissionChoice.ALLOW_ONCE,
    )
    decision = gate.decide(
        ToolCall(
            id="1",
            name="write_file",
            arguments=json.dumps({"path": "x.py", "content": "x"}),
        )
    )
    assert not decision.allowed
    assert "ask" in decision.reason.lower()


def test_hydrate_memory_skips_tools_uses_compact(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    hist = SessionHistory(store, session_id="sess01")
    hist.session_start(model="stub")
    hist.user_turn("old question")
    hist.assistant_turn("old answer")
    hist.tool("search_repo", preview="hit")
    hist.compact("User asked about auth; found login.py")
    hist.user_turn("follow up")
    hist.assistant_turn("more detail")

    events = store.list_events("sess01")
    memory = hydrate_memory_from_events(events)
    assert memory.summary and "auth" in memory.summary
    texts = [m.content for m in memory.snapshot()]
    assert "old question" not in texts
    assert "follow up" in texts
    assert "more detail" in texts
    assert all("hit" not in t for t in texts)


def test_continue_appends_same_session(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    hist = SessionHistory(store, session_id="abc")
    hist.session_start(model="stub")
    hist.user_turn("hi")
    hist.assistant_turn("hello")

    memory = ConversationMemory()
    events = hist.list_events("abc")
    hydrate_memory_from_events(events, memory)
    hist.attach("abc")
    hist.user_turn("again")
    assert hist.session_id == "abc"
    kinds = [e.kind for e in store.list_events("abc")]
    assert kinds.count(EventKind.USER_TURN) == 2


def test_sessions_preview_and_most_recent(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    a = SessionHistory(store, session_id="aaaa")
    a.session_start(model="m1")
    a.user_turn("first session about widgets")
    b = SessionHistory(store, session_id="bbbb")
    b.session_start(model="m2")
    b.user_turn("second session")

    hist = SessionHistory(store)
    rows = hist.list_recent_sessions(limit=5)
    assert rows[0]["id"] == "bbbb"
    assert "second" in (rows[0].get("preview") or "")
    assert hist.most_recent_session_id() == "bbbb"
    assert "widgets" in session_preview(store.list_events("aaaa"))


def test_export_md_and_json(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path, enabled=True)
    hist = SessionHistory(store, session_id="exp1")
    hist.session_start(model="stub", provider="stub", workdir=str(tmp_path))
    hist.user_turn("how does x work?")
    hist.tool("read_file", preview="def x():")
    hist.assistant_turn("x does y")
    events = store.list_events("exp1")
    index = {"id": "exp1", "model": "stub", "provider": "stub", "workdir": str(tmp_path)}
    md = export_session_markdown("exp1", events, index=index)
    assert "# Cozmo session" in md
    assert "how does x work?" in md
    assert "### Tool" in md
    js = json.loads(export_session_json("exp1", events, index=index))
    assert js["session_id"] == "exp1"
    assert len(js["events"]) >= 3
