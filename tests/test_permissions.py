"""Permission gate + store (no network)."""

import json
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.app.permissions import PermissionChoice, PermissionGate, build_preview
from cozmo.domain.mode import AgentMode
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permission_store import PermissionRule, PermissionStore
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor


def test_permission_store_roundtrip(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    store.add_allow(PermissionRule(tool="run_shell", pattern="pytest *"))
    store.add_deny(PermissionRule(tool="run_shell", pattern="rm *"))
    reloaded = PermissionStore.load(tmp_path)
    assert any(r.pattern == "pytest *" for r in reloaded.allow)
    assert any(r.pattern == "rm *" for r in reloaded.deny)
    path = tmp_path / ".cozmo" / "permissions.json"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1


def test_gate_deny_by_default(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    gate = PermissionGate(
        workdir=tmp_path,
        store=store,
        default_choice=PermissionChoice.DENY,
    )
    call = ToolCall(
        id="1",
        name="write_file",
        arguments=json.dumps({"path": "a.txt", "content": "hi"}),
    )
    decision = gate.decide(call)
    assert not decision.allowed
    assert (tmp_path / "a.txt").exists() is False


def test_gate_allow_once_executes(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    gate = PermissionGate(
        workdir=tmp_path,
        store=store,
        ask=lambda _c, _p: PermissionChoice.ALLOW_ONCE,
    )
    guard = WorkspaceGuard(tmp_path, allow_write=True)
    reg = build_default_registry(guard)
    runner = AgentRunner(
        StubLLMClient(
            scripted_tool_calls=(
                ToolCall(
                    id="1",
                    name="write_file",
                    arguments=json.dumps({"path": "a.txt", "content": "hello"}),
                ),
            ),
            final_text="wrote",
        ),
        reg,
        ToolExecutor(reg),
        permission_gate=gate,
        max_steps=5,
    )
    result = runner.run("write a file")
    assert "wrote" in result.final_text
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"


def test_gate_always_allow_persists(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    choices = iter([PermissionChoice.ALWAYS_ALLOW])
    gate = PermissionGate(
        workdir=tmp_path,
        store=store,
        ask=lambda _c, _p: next(choices),
    )
    call = ToolCall(
        id="1",
        name="run_shell",
        arguments=json.dumps({"command": "echo hi"}),
    )
    d1 = gate.decide(call)
    assert d1.allowed
    # Second call: no ask needed (persisted)
    gate.ask = lambda _c, _p: (_ for _ in ()).throw(AssertionError("should not ask"))
    d2 = gate.decide(call)
    assert d2.allowed
    assert d2.choice == PermissionChoice.ALWAYS_ALLOW


def test_gate_plan_mode_blocks_write(tmp_path: Path) -> None:
    store = PermissionStore.load(tmp_path)
    gate = PermissionGate(
        workdir=tmp_path,
        store=store,
        mode=AgentMode.PLAN,
        ask=lambda _c, _p: PermissionChoice.ALLOW_ONCE,
    )
    call = ToolCall(
        id="1",
        name="write_file",
        arguments=json.dumps({"path": "x.py", "content": "x"}),
    )
    decision = gate.decide(call)
    assert not decision.allowed
    assert "plan mode" in decision.reason.lower()


def test_write_preview_diff(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\n", encoding="utf-8")
    call = ToolCall(
        id="1",
        name="write_file",
        arguments=json.dumps({"path": "f.txt", "content": "one\ntwo\n"}),
    )
    preview = build_preview(tmp_path, call)
    assert "write_file" in preview or "f.txt" in preview
    assert "+two" in preview or "two" in preview


def test_agent_plan_mode_hides_mutating_tools(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path, allow_write=True, allow_shell=True)
    reg = build_default_registry(guard)
    store = PermissionStore.load(tmp_path)
    gate = PermissionGate(workdir=tmp_path, store=store, mode=AgentMode.PLAN)
    runner = AgentRunner(
        StubLLMClient(final_text="plan ready"),
        reg,
        ToolExecutor(reg),
        permission_gate=gate,
        mode=AgentMode.PLAN,
        max_steps=3,
    )
    names = {s.name for s in runner._tool_specs_for_mode()}
    assert "write_file" not in names
    assert "run_shell" not in names
    assert "read_file" in names
    runner.set_mode(AgentMode.AGENT)
    names2 = {s.name for s in runner._tool_specs_for_mode()}
    assert "write_file" in names2
