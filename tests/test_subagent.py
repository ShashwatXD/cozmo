"""Model router + subagent depth."""

import json
from pathlib import Path

from cozmo.app.model_router import ModelRouter
from cozmo.app.subagent import SubAgentService, register_subagent_tool
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.roles import ModelRole
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


def test_model_router_fallback() -> None:
    a = StubLLMClient(final_text="orch")
    b = StubLLMClient(final_text="work")
    router = ModelRouter({ModelRole.ORCHESTRATOR: a, ModelRole.WORKER: b})
    assert router.orchestrator is a
    assert router.worker is b
    worker_only = ModelRouter({ModelRole.WORKER: b})
    assert worker_only.orchestrator is b
    assert worker_only.worker is b


def test_subagent_depth_kill() -> None:
    llm = StubLLMClient(final_text="done")
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    policy = AgentPolicy(max_subagent_depth=0, max_subagent_steps=2)
    svc = SubAgentService(
        models=ModelRouter.from_single(llm),
        parent_registry=reg,
        policy=policy,
        depth=0,
    )
    out = json.loads(svc.run("explore math_utils"))
    assert out["ok"] is False
    assert out["stop_reason"] == "subagent_depth"


def test_register_subagent_tool() -> None:
    llm = StubLLMClient(final_text="found add()")
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    svc = SubAgentService(
        models=ModelRouter.from_single(llm),
        parent_registry=reg,
        policy=AgentPolicy(max_subagent_depth=1, max_subagent_steps=3),
    )
    register_subagent_tool(reg, svc)
    assert any(s.name == "run_subtask" for s in reg.specs())
    ex = ToolExecutor(reg)
    from cozmo.domain.tools import ToolCall

    result = ex.execute(
        ToolCall(
            id="1",
            name="run_subtask",
            arguments=json.dumps({"goal": "What is in math_utils?"}),
        )
    )
    assert not result.is_error
    payload = json.loads(result.content)
    assert "summary" in payload
    assert "paths" in payload
    assert "claims" in payload
