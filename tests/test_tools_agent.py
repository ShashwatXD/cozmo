"""Tool permissions + registry + agent loop (no network)."""

import json
from pathlib import Path

import pytest

from cozmo.app.agent import AgentRunner
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import PermissionError_, WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


def test_resolve_blocks_escape(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(PermissionError_):
        guard.resolve("../outside.txt")


def test_read_write_search(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path, allow_write=True)
    reg = build_default_registry(guard)
    ex = ToolExecutor(reg)

    write = ToolCall(
        id="1",
        name="write_file",
        arguments=json.dumps({"path": "a.txt", "content": "hello cozmo"}),
    )
    assert not ex.execute(write).is_error

    read = ToolCall(id="2", name="read_file", arguments=json.dumps({"path": "a.txt"}))
    assert ex.execute(read).content == "hello cozmo"

    search = ToolCall(
        id="3", name="search_repo", arguments=json.dumps({"query": "cozmo"})
    )
    assert "a.txt" in ex.execute(search).content


def test_shell_disabled(tmp_path: Path) -> None:
    guard = WorkspaceGuard(tmp_path, allow_shell=False)
    reg = build_default_registry(guard)
    ex = ToolExecutor(reg)
    result = ex.execute(
        ToolCall(id="1", name="run_shell", arguments=json.dumps({"command": "echo hi"}))
    )
    assert result.is_error
    assert "disabled" in result.content.lower()


def test_agent_loop_with_scripted_tools() -> None:
    # Flutter: Cubit with FakeRepo that returns a "use tool" then "done"
    call = ToolCall(
        id="call_1",
        name="read_file",
        arguments=json.dumps({"path": "math_utils.py"}),
    )
    llm = StubLLMClient(
        scripted_tool_calls=(call,),
        final_text="Found add() with off-by-one.",
    )
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    runner = AgentRunner(llm, reg, ToolExecutor(reg), max_steps=5)
    result = runner.run("What is wrong in math_utils?")
    assert "off-by-one" in result.final_text
    assert result.steps >= 2
    assert len(runner.memory) >= 2


def test_agent_memory_across_turns() -> None:
    call = ToolCall(
        id="call_1",
        name="read_file",
        arguments=json.dumps({"path": "math_utils.py"}),
    )
    llm = StubLLMClient(scripted_tool_calls=(call,), final_text="bug noted")
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    runner = AgentRunner(llm, reg, ToolExecutor(reg), max_steps=5)
    runner.run("Read the file")
    n_after_first = len(runner.memory)
    runner.run("Remind me what you found")
    assert len(runner.memory) > n_after_first
