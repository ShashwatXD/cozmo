"""Execution engine - runs planned steps via the tool registry."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from cozmo.domain.tools import ToolCall
from cozmo.infra.tools.registry import ToolExecutor, ToolRegistry
from cozmo.runtime.planner import Planner, StepPlan

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

@dataclass
class StepResult:
    """Outcome of a single execution step."""

    step_id: str
    status: str = "pending"  # pending | running | done | failed
    output: str = ""
    error: str | None = None

class ExecutionEngine:
    """
    Takes a Planner + ToolRegistry + ToolExecutor.
    Produces a plan for a task, then executes each step sequentially.
    """

    def __init__(
        self,
        planner: Planner,
        registry: ToolRegistry,
        executor: ToolExecutor,
    ) -> None:
        self._planner = planner
        self._registry = registry
        self._executor = executor

    def execute(self, task: str) -> list[StepResult]:
        """Plan and execute all steps for *task*. Returns ordered StepResults."""
        available = [s.name for s in self._registry.specs()]
        steps = self._planner.plan(task, available)

        if not steps:
            return [
                StepResult(
                    step_id="step_0",
                    status="done",
                    output="No actionable steps identified for this task.",
                )
            ]

        results: list[StepResult] = []
        for step in steps:
            result = self._execute_step(step)
            results.append(result)
        return results

    def _execute_step(self, step: StepPlan) -> StepResult:
        """Execute a single step with up to _MAX_RETRIES on failure."""
        result = StepResult(step_id=step.step_id, status="running")

        if step.tool_name is None:
            result.status = "done"
            result.output = f"Action noted: {step.action}"
            return result

        for attempt in range(_MAX_RETRIES + 1):
            call = ToolCall(
                id=f"{step.step_id}_attempt_{attempt}",
                name=step.tool_name,
                arguments=json.dumps(step.args),
            )
            tool_result = self._executor.execute(call)

            if not tool_result.is_error:
                result.status = "done"
                result.output = tool_result.content
                result.error = None
                return result

            result.error = tool_result.content
            logger.warning(
                "Step %s attempt %d failed: %s",
                step.step_id,
                attempt + 1,
                tool_result.content,
            )

        result.status = "failed"
        result.output = ""
        return result
