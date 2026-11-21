"""The dashboard groups traces into turns and serves them on localhost only."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.cli.dashboard import start_server
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.telemetry import trace_reader
from cozmo.infra.telemetry.tracer import Tracer
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor


def _traced_turn(root: Path) -> Path:
    (root / "math_utils.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    trace = root / ".cozmo" / "traces.jsonl"
    reg = build_default_registry(WorkspaceGuard(root))
    llm = StubLLMClient(
        scripted_tool_calls=(
            ToolCall(id="1", name="read_file", arguments='{"path": "math_utils.py"}'),
        ),
        final_text="add() subtracts: math_utils.py:2 returns a - b",
    )
    AgentRunner(llm, reg, ToolExecutor(reg), tracer=Tracer(trace, session_id="s1")).run(
        "why is add wrong?"
    )
    return trace


def test_a_turn_is_traced_step_by_step(tmp_path: Path) -> None:
    trace = _traced_turn(tmp_path)
    [turn] = trace_reader.list_turns(trace)
    assert turn["question"] == "why is add wrong?"
    assert (turn["llm_calls"], turn["tool_calls"], turn["stop_reason"]) == (2, 1, "completed")
    assert turn["groundedness"] is not None
    events = trace_reader.turn_detail(trace, turn["id"])["events"]
    tool = next(e for e in events if e["kind"] == "tool")
    assert "return a - b" in tool["data"]["result"]
    llm = next(e for e in events if e["kind"] == "llm")
    assert llm["data"]["messages"][-1]["content"] == "why is add wrong?"


def test_server_answers_localhost_and_refuses_other_hosts(tmp_path: Path) -> None:
    _traced_turn(tmp_path)
    server, url = start_server(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        turns = json.load(urllib.request.urlopen(f"{url}/api/turns"))["turns"]
        assert len(turns) == 1
        page = urllib.request.urlopen(url).read().decode()
        assert "<title>Cozmo Traces</title>" in page
        evil = urllib.request.Request(f"{url}/api/turns", headers={"Host": "evil.example"})
        try:
            urllib.request.urlopen(evil)
            raise AssertionError("expected 403")
        except urllib.error.HTTPError as err:
            assert err.code == 403
    finally:
        server.shutdown()
        server.server_close()
