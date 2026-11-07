"""Hierarchical multi-agent teams: director → workers → synthesizer."""

from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.app.subagent import SubAgentService
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.messages import Message, Role
from cozmo.domain.team import (
    AgentDef,
    allowed_tools_for,
    list_default_workers,
    resolve_agent,
)
from cozmo.domain.tools import ToolSpec
from cozmo.prompts.loader import load_system_prompt

RUN_TEAM_SPEC = ToolSpec(
    name="run_team",
    description=(
        "Run a hierarchical multi-agent team for complex tasks. "
        "Director plans → workers (researcher, coder, reviewer) execute "
        "(parallel by default) → synthesizer merges into one answer. "
        "Use for multi-step investigation, cross-cutting analysis, or tasks "
        "needing distinct roles. Workers are read-only; parent agent applies edits."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The team task / user goal to accomplish",
            },
            "workers": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional worker agent names or ids (default: researcher, coder, reviewer)"
                ),
            },
            "parallel": {
                "type": "boolean",
                "description": "Run worker goals in parallel (default true)",
                "default": True,
            },
            "max_loops": {
                "type": "integer",
                "description": "Director refinement loops, 1–3 (default 1)",
                "default": 1,
            },
        },
        "required": ["task"],
    },
)

_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


@dataclass(frozen=True)
class WorkerOrder:
    agent: AgentDef
    goal: str


@dataclass(frozen=True)
class WorkerResult:
    agent_id: str
    agent_name: str
    goal: str
    summary: str
    paths: list[str]
    claims: list[str]
    open_questions: list[str]
    ok: bool
    steps: int
    stop_reason: str
    error: str = ""


def plan_team_orders(
    models: ModelRouter,
    *,
    task: str,
    workers: list[AgentDef],
    temperature: float = 0.2,
) -> tuple[str, list[WorkerOrder]]:
    """Director LLM assigns one goal per worker. Falls back if JSON parse fails."""
    if not workers:
        return "No workers available", []

    worker_list = "\n".join(f"- {w.name} ({w.id}): {w.description}" for w in workers)
    prompt = f"Task:\n{task}\n\nWorkers:\n{worker_list}\n\nRespond with JSON only."
    director_prompt = load_system_prompt("director")
    result = models.orchestrator.complete(
        [
            Message(role=Role.SYSTEM, content=director_prompt),
            Message(role=Role.USER, content=prompt),
        ],
        temperature=temperature,
        json_mode=True,
    )
    text = (result.content or "").strip()
    parsed = _parse_director_plan(text, workers, task)
    return parsed


def _parse_director_plan(
    text: str,
    workers: list[AgentDef],
    task: str,
) -> tuple[str, list[WorkerOrder]]:
    fallback = (
        "Fallback: each worker receives the full task",
        [WorkerOrder(agent=w, goal=task) for w in workers],
    )
    match = _JSON_OBJ.search(text)
    if not match:
        return fallback

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return fallback

    if not isinstance(data, dict):
        return fallback

    plan = str(data.get("plan") or "Director plan").strip()
    raw_orders = data.get("orders")
    if not isinstance(raw_orders, list):
        return plan or fallback[0], fallback[1]

    orders: list[WorkerOrder] = []
    for item in raw_orders:
        if not isinstance(item, dict):
            continue
        agent_key = str(item.get("agentId") or item.get("agentName") or "").strip()
        goal = str(item.get("goal") or "").strip()
        if not agent_key or not goal:
            continue
        agent = resolve_agent(agent_key)
        if agent is None or agent.name in {"director", "synthesizer"}:
            continue
        orders.append(WorkerOrder(agent=agent, goal=goal))

    if not orders:
        return plan or fallback[0], fallback[1]
    return plan, orders


def synthesize_team_results(
    models: ModelRouter,
    *,
    task: str,
    plan: str,
    worker_results: list[WorkerResult],
    temperature: float = 0.2,
) -> str:
    synth_prompt = load_system_prompt("synthesizer")
    payload = {
        "task": task,
        "plan": plan,
        "workers": [
            {
                "agent": r.agent_name,
                "goal": r.goal,
                "ok": r.ok,
                "summary": r.summary,
                "paths": r.paths,
                "claims": r.claims,
                "open_questions": r.open_questions,
                "error": r.error or None,
            }
            for r in worker_results
        ],
    }
    result = models.orchestrator.complete(
        [
            Message(role=Role.SYSTEM, content=synth_prompt),
            Message(
                role=Role.USER,
                content=(
                    "Synthesize these multi-agent results into one answer.\n\n"
                    + json.dumps(payload, indent=2)
                ),
            ),
        ],
        temperature=temperature,
    )
    return (result.content or "").strip() or "(empty synthesis)"


class TeamService:
    """
    Hierarchical team orchestration wired as run_team from the app/CLI layer.
    Director + synthesizer use orchestrator; workers use scoped SubAgentService.
    """

    def __init__(
        self,
        *,
        models: ModelRouter,
        parent_registry: Any,
        policy: AgentPolicy,
        history: SessionHistory | None = None,
        temperature: float = 0.2,
        model_name: str = "stub-model",
        depth: int = 0,
        max_tool_result_chars: int = 24_000,
    ) -> None:
        self._models = models
        self._parent_registry = parent_registry
        self._policy = policy
        self._history = history
        self._temperature = temperature
        self._model_name = model_name
        self._depth = depth
        self._max_tool_result_chars = max_tool_result_chars

    def _worker_service(self, agent: AgentDef) -> SubAgentService:
        prompt = load_system_prompt(agent.prompt_name)
        return SubAgentService(
            models=self._models,
            parent_registry=self._parent_registry,
            policy=self._policy,
            history=None,
            temperature=self._temperature,
            model_name=self._model_name,
            depth=self._depth,
            allowed_tools=allowed_tools_for(agent),
            system_prompt=prompt,
            max_tool_result_chars=self._max_tool_result_chars,
        )

    def _run_worker(self, order: WorkerOrder, *, context: str) -> WorkerResult:
        goal = order.goal
        if context:
            goal = f"{context}\n\nGoal:\n{order.goal}"
        try:
            raw = self._worker_service(order.agent).run(goal, max_steps=order.agent.max_steps)
            pack = json.loads(raw)
            return WorkerResult(
                agent_id=order.agent.id,
                agent_name=order.agent.name,
                goal=order.goal,
                summary=str(pack.get("summary") or ""),
                paths=[str(p) for p in pack.get("paths") or []][:20],
                claims=[str(c) for c in pack.get("claims") or []][:30],
                open_questions=[str(q) for q in pack.get("open_questions") or []][:15],
                ok=bool(pack.get("ok")),
                steps=int(pack.get("steps") or 0),
                stop_reason=str(pack.get("stop_reason") or ""),
            )
        except Exception as exc:  # noqa: BLE001 — worker failure must not kill team
            return WorkerResult(
                agent_id=order.agent.id,
                agent_name=order.agent.name,
                goal=order.goal,
                summary="",
                paths=[],
                claims=[],
                open_questions=[],
                ok=False,
                steps=0,
                stop_reason="error",
                error=str(exc),
            )

    def run(
        self,
        task: str,
        *,
        worker_names: list[str] | None = None,
        parallel: bool = True,
        max_loops: int = 1,
    ) -> str:
        depth_stop = self._policy.check_subagent_depth(self._depth + 1)
        if depth_stop is not None:
            return json.dumps(
                {
                    "ok": False,
                    "team_id": None,
                    "plan": "",
                    "workers": [],
                    "synthesis": (
                        f"team blocked: subagent depth exceeded ({self._policy.max_subagent_depth})"
                    ),
                    "loops": 0,
                    "stop_reason": depth_stop.value,
                }
            )

        task = task.strip()
        if not task:
            raise ValueError("task is required")

        workers = self._resolve_workers(worker_names)
        loops = max(1, min(max_loops, 3))
        team_id = uuid.uuid4().hex[:12]

        if self._history:
            self._history.team(task, team_id=team_id, workers=[w.name for w in workers])

        all_results: list[WorkerResult] = []
        last_plan = ""

        for loop in range(loops):
            loop_task = task
            if loop > 0 and all_results:
                loop_task = (
                    f"{task}\n\nPrevious worker results:\n"
                    f"{json.dumps([r.__dict__ for r in all_results], indent=2)}\n\n"
                    "Refine remaining work."
                )

            plan, orders = plan_team_orders(
                self._models,
                task=loop_task,
                workers=workers,
                temperature=self._temperature,
            )
            last_plan = plan
            if not orders:
                break

            context = f"Team task: {task}\nDirector plan: {plan}"
            loop_results = self._run_orders(orders, context=context, parallel=parallel)
            all_results.extend(loop_results)

        synthesis = synthesize_team_results(
            self._models,
            task=task,
            plan=last_plan,
            worker_results=all_results,
            temperature=self._temperature,
        )

        ok = any(r.ok for r in all_results) and bool(synthesis)
        return json.dumps(
            {
                "ok": ok,
                "team_id": team_id,
                "plan": last_plan,
                "workers": [r.__dict__ for r in all_results],
                "synthesis": synthesis,
                "loops": loops,
            }
        )

    def _resolve_workers(self, worker_names: list[str] | None) -> list[AgentDef]:
        if not worker_names:
            return list_default_workers()
        out: list[AgentDef] = []
        for name in worker_names:
            agent = resolve_agent(str(name))
            if agent is None or agent.name in {"director", "synthesizer"}:
                continue
            if agent not in out:
                out.append(agent)
        return out or list_default_workers()

    def _run_orders(
        self,
        orders: list[WorkerOrder],
        *,
        context: str,
        parallel: bool,
    ) -> list[WorkerResult]:
        if not orders:
            return []
        if not parallel or len(orders) == 1:
            return [self._run_worker(o, context=context) for o in orders]

        results: list[WorkerResult | None] = [None] * len(orders)
        with ThreadPoolExecutor(max_workers=len(orders)) as pool:
            future_map = {
                pool.submit(self._run_worker, order, context=context): i
                for i, order in enumerate(orders)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                results[idx] = future.result()
        return [r for r in results if r is not None]

    def handler(self, args: dict[str, Any]) -> str:
        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("task is required")

        raw_workers = args.get("workers")
        worker_names: list[str] | None = None
        if isinstance(raw_workers, list):
            worker_names = [str(w) for w in raw_workers if str(w).strip()]

        parallel = args.get("parallel", True)
        if isinstance(parallel, str):
            parallel = parallel.lower() not in {"0", "false", "no"}

        max_loops = args.get("max_loops", 1)
        loops = 1
        if isinstance(max_loops, int):
            loops = max_loops
        elif isinstance(max_loops, str) and max_loops.isdigit():
            loops = int(max_loops)

        return self.run(
            task,
            worker_names=worker_names,
            parallel=bool(parallel),
            max_loops=loops,
        )


def register_team_tool(registry: Any, service: TeamService) -> None:
    """Wire run_team from app/CLI; keeps infra free of app imports."""
    registry.register(RUN_TEAM_SPEC, service.handler)
