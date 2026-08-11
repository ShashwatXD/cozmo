"""Nested subagent runner — scoped tools + tighter AgentPolicy."""

from __future__ import annotations

import json
from typing import Any

from cozmo.app.agent import AgentRunner
from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.domain.guardrails import AgentPolicy, StopReason
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.tools import ToolSpec
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.prompts.loader import load_system_prompt

SUBTASK_SPEC = ToolSpec(
    name="run_subtask",
    description=(
        "Spawn a scoped subagent (read/search tools only) to explore or "
        "answer a sub-goal. Returns JSON with summary — use for investigation "
        "without polluting the main thread."
    ),
    parameters={
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Sub-task goal"},
            "max_steps": {
                "type": "integer",
                "description": "Optional step cap for the child agent",
            },
        },
        "required": ["goal"],
    },
)


class SubAgentService:
    """
    Spawns a nested AgentRunner with worker model + child policy.
    Registered as a tool from the app/CLI layer (not from infra → app).
    """

    def __init__(
        self,
        *,
        models: ModelRouter,
        parent_registry: ToolRegistry,
        policy: AgentPolicy,
        history: SessionHistory | None = None,
        temperature: float = 0.2,
        model_name: str = "stub-model",
        depth: int = 0,
        allowed_tools: frozenset[str] | None = None,
    ) -> None:
        self._models = models
        self._parent_registry = parent_registry
        self._policy = policy
        self._history = history
        self._temperature = temperature
        self._model_name = model_name
        self._depth = depth
        self._allowed = allowed_tools or frozenset(
            {
                "read_file",
                "search_repo",
                "semantic_search",
                "symbol_search",
                "find_references",
                "get_codebase_graph",
                "git_status",
                "git_diff",
            }
        )

    def run(self, goal: str, *, max_steps: int | None = None) -> str:
        depth_stop = self._policy.check_subagent_depth(self._depth + 1)
        if depth_stop is not None:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"subagent depth exceeded "
                        f"({self._policy.max_subagent_depth})"
                    ),
                    "stop_reason": depth_stop.value,
                }
            )

        child_policy = self._policy.for_subagent()
        if max_steps is not None:
            from dataclasses import replace

            child_policy = replace(child_policy, max_agent_steps=max_steps)

        child_registry = ToolRegistry()
        for spec in self._parent_registry.specs():
            if spec.name not in self._allowed:
                continue
            parent_tool = self._parent_registry.get(spec.name)
            if parent_tool is None:
                continue
            child_registry.register(parent_tool.spec, parent_tool.handler)

        runner = AgentRunner(
            models=self._models,
            registry=child_registry,
            executor=ToolExecutor(child_registry),
            memory=ConversationMemory(max_messages=child_policy.memory_max_messages),
            history=None,
            policy=child_policy,
            system_prompt=load_system_prompt("subagent"),
            temperature=self._temperature,
            model_name=self._model_name,
            depth=self._depth + 1,
        )
        if self._history:
            self._history.subagent(goal, depth=self._depth + 1)

        result = runner.run(goal)
        payload: dict[str, Any] = {
            "ok": result.stop_reason == StopReason.COMPLETED,
            "summary": result.final_text[:4000],
            "steps": result.steps,
            "stop_reason": result.stop_reason.value,
            "tokens": result.usage.total_tokens,
        }
        return json.dumps(payload)

    def handler(self, args: dict[str, Any]) -> str:
        goal = str(args.get("goal") or "").strip()
        if not goal:
            raise ValueError("goal is required")
        max_steps = args.get("max_steps")
        steps: int | None
        if isinstance(max_steps, int):
            steps = max_steps
        elif isinstance(max_steps, str) and max_steps.isdigit():
            steps = int(max_steps)
        else:
            steps = None
        return self.run(goal, max_steps=steps)


def register_subagent_tool(registry: ToolRegistry, service: SubAgentService) -> None:
    """Wire run_subtask from app/CLI — keeps infra free of app imports."""
    registry.register(SUBTASK_SPEC, service.handler)
