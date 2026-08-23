"""Multi-agent team orchestration."""

import json
from pathlib import Path

from cozmo.app.model_router import ModelRouter
from cozmo.app.team import (
    TeamService,
    plan_team_orders,
    register_team_tool,
)
from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.messages import Message, Role
from cozmo.domain.roles import ModelRole
from cozmo.domain.team import list_default_workers, resolve_agent
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"

PLAN_JSON = json.dumps(
    {
        "plan": "Research the codebase then review findings",
        "orders": [
            {"agentId": "researcher", "goal": "Find add() in math_utils"},
            {"agentId": "reviewer", "goal": "Check the researcher's findings"},
        ],
    }
)


class OrchestratorStub:
    """Returns director JSON on json_mode; synthesis text otherwise."""

    def __init__(self, *, plan_json: str = PLAN_JSON, synthesis: str = "Team synthesis") -> None:
        self._plan_json = plan_json
        self._synthesis = synthesis

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools=None,
    ) -> CompletionResult:
        if json_mode:
            return CompletionResult(content=self._plan_json, usage=Usage(), finish_reason="stop")
        return CompletionResult(content=self._synthesis, usage=Usage(), finish_reason="stop")

    def stream(self, messages, *, temperature: float = 0.2):
        yield self.complete(messages, temperature=temperature).content or ""


def test_resolve_default_agents() -> None:
    workers = list_default_workers()
    names = {w.name for w in workers}
    assert names == {"researcher", "coder", "reviewer"}
    assert resolve_agent("researcher") is not None
    assert resolve_agent("director") is not None
    assert resolve_agent("unknown") is None


def test_plan_team_orders_parses_json() -> None:
    orch = OrchestratorStub()
    models = ModelRouter({ModelRole.ORCHESTRATOR: orch, ModelRole.WORKER: StubLLMClient()})
    workers = list_default_workers()[:2]
    plan, orders = plan_team_orders(
        models, task="Explore math_utils", workers=workers, temperature=0.0
    )
    assert "Research" in plan
    assert len(orders) == 2
    assert orders[0].agent.name == "researcher"
    assert "add()" in orders[0].goal


def test_plan_team_orders_fallback_on_bad_json() -> None:
    orch = OrchestratorStub(plan_json="not json at all")
    models = ModelRouter({ModelRole.ORCHESTRATOR: orch, ModelRole.WORKER: StubLLMClient()})
    workers = list_default_workers()[:1]
    plan, orders = plan_team_orders(models, task="Do the thing", workers=workers)
    assert "Fallback" in plan
    assert len(orders) == 1
    assert orders[0].goal == "Do the thing"


def test_team_run_end_to_end() -> None:
    worker = StubLLMClient(
        final_text=json.dumps(
            {
                "paths": ["math_utils.py"],
                "claims": ["add() at math_utils.py:1"],
                "open_questions": [],
                "summary": "Found add()",
            }
        )
    )
    orch = OrchestratorStub(synthesis="Combined: add() lives in math_utils.py")
    models = ModelRouter({ModelRole.ORCHESTRATOR: orch, ModelRole.WORKER: worker})

    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    svc = TeamService(
        models=models,
        parent_registry=reg,
        policy=AgentPolicy(max_subagent_depth=1, max_subagent_steps=3),
    )
    out = json.loads(
        svc.run(
            "Where is add() defined?",
            worker_names=["researcher", "reviewer"],
            parallel=False,
        )
    )
    assert out["ok"] is True
    assert out["team_id"]
    assert out["plan"]
    assert len(out["workers"]) == 2
    assert "Combined" in out["synthesis"]


def test_team_depth_blocked() -> None:
    orch = OrchestratorStub()
    models = ModelRouter({ModelRole.ORCHESTRATOR: orch, ModelRole.WORKER: StubLLMClient()})
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    svc = TeamService(
        models=models,
        parent_registry=reg,
        policy=AgentPolicy(max_subagent_depth=0),
        depth=0,
    )
    out = json.loads(svc.run("task"))
    assert out["ok"] is False
    assert out["stop_reason"] == "subagent_depth"


def test_register_team_tool() -> None:
    orch = OrchestratorStub()
    worker = StubLLMClient(final_text='{"paths":[],"claims":[],"open_questions":[],"summary":"ok"}')
    models = ModelRouter({ModelRole.ORCHESTRATOR: orch, ModelRole.WORKER: worker})
    guard = WorkspaceGuard(FIXTURE, allow_write=False, allow_shell=False)
    reg = build_default_registry(guard)
    svc = TeamService(
        models=models,
        parent_registry=reg,
        policy=AgentPolicy(max_subagent_depth=1, max_subagent_steps=2),
    )
    register_team_tool(reg, svc)
    assert any(s.name == "run_team" for s in reg.specs())
    ex = ToolExecutor(reg)
    from cozmo.domain.tools import ToolCall

    result = ex.execute(
        ToolCall(
            id="1",
            name="run_team",
            arguments=json.dumps({"task": "Analyze math_utils", "workers": ["researcher"]}),
        )
    )
    assert not result.is_error
    payload = json.loads(result.content)
    assert "synthesis" in payload
    assert "workers" in payload
