"""Nested subagent runner: scoped tools + compact evidence packs."""

from __future__ import annotations

import json
import re
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
        "Spawn a scoped subagent (read/search only) for a sub-goal. "
        "Returns a compact JSON evidence pack: paths, claims, open_questions, summary."
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

_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _evidence_pack(final_text: str, *, steps: int, stop_reason: str, tokens: int) -> dict[str, Any]:
    """Parse subagent output into a bounded evidence pack."""
    text = (final_text or "").strip()
    data: dict[str, Any] | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ.search(text)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        data = {
            "paths": [],
            "claims": [],
            "open_questions": [],
            "summary": text[:1500] or "(empty)",
        }

    def _str_list(key: str, *, limit: int) -> list[str]:
        raw = data.get(key) if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for item in raw:
            s = str(item).strip()
            if s:
                out.append(s[:400])
            if len(out) >= limit:
                break
        return out

    summary = str(data.get("summary") or "").strip()[:1500]
    if not summary and isinstance(data.get("claims"), list) and data["claims"]:
        summary = str(data["claims"][0])[:1500]

    return {
        "ok": stop_reason == StopReason.COMPLETED.value,
        "paths": _str_list("paths", limit=20),
        "claims": _str_list("claims", limit=30),
        "open_questions": _str_list("open_questions", limit=15),
        "summary": summary or "(no summary)",
        "steps": steps,
        "stop_reason": stop_reason,
        "tokens": tokens,
    }


class SubAgentService:
    """
    Spawns a nested AgentRunner with worker model + child policy.
    Registered as a tool from the app/CLI layer (not from infra -> app).
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
        self._allowed = allowed_tools or frozenset(
            {
                "read_file",
                "search_repo",
                "semantic_search",
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
                    "paths": [],
                    "claims": [],
                    "open_questions": [],
                    "summary": (
                        f"subagent depth exceeded "
                        f"({self._policy.max_subagent_depth})"
                    ),
                    "stop_reason": depth_stop.value,
                    "steps": 0,
                    "tokens": 0,
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
            executor=ToolExecutor(
                child_registry, max_chars=self._max_tool_result_chars
            ),
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
        pack = _evidence_pack(
            result.final_text,
            steps=result.steps,
            stop_reason=result.stop_reason.value,
            tokens=result.usage.total_tokens,
        )
        return json.dumps(pack)

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
    """Wire run_subtask from app/CLI; keeps infra free of app imports."""
    registry.register(SUBTASK_SPEC, service.handler)
