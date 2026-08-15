"""Simple heuristic planner for retrieval / file tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepPlan:
    """A single planned step in a task execution."""

    step_id: str
    action: str
    tool_name: str | None = None
    args: dict[str, object] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


# Keyword patterns → (tool_name, action_description)
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"(?:search|find|look)\s+(?:for\s+)?(?:code|function|class)", re.I),
        "semantic_search",
        "Semantic search for code",
    ),
    (
        re.compile(r"(?:find|where|locate|look\s?up)", re.I),
        "semantic_search",
        "Semantic search for relevant code",
    ),
    (
        re.compile(r"(?:read|show|open|cat|view)\s+(?:file|content)", re.I),
        "read_file",
        "Read file contents",
    ),
]


def _extract_subject(task: str) -> str:
    """Try to pull the main subject (symbol / path) from the task string."""
    m = re.search(r"['\"]([^'\"]+)['\"]", task)
    if m:
        return m.group(1)
    m = re.search(r"`([^`]+)`", task)
    if m:
        return m.group(1)
    return ""


class Planner:
    """Keyword-based planner that suggests retrieval tools for a task."""

    def plan(self, task: str, available_tools: list[str]) -> list[StepPlan]:
        """Produce an ordered list of steps for *task* using *available_tools*."""
        steps: list[StepPlan] = []
        seen_tools: set[str] = set()
        subject = _extract_subject(task)

        for pattern, tool_name, action in _PATTERNS:
            if tool_name in seen_tools:
                continue
            if not pattern.search(task):
                continue
            if tool_name not in available_tools:
                continue

            args: dict[str, object] = {}
            if tool_name == "semantic_search":
                args["query"] = subject or task[:120]
            elif tool_name == "read_file" and subject:
                args["path"] = subject

            step_id = f"step_{len(steps) + 1}"
            depends = [steps[-1].step_id] if steps else []
            steps.append(
                StepPlan(
                    step_id=step_id,
                    action=action,
                    tool_name=tool_name,
                    args=args,
                    depends_on=depends,
                )
            )
            seen_tools.add(tool_name)

        if not steps and "semantic_search" in available_tools:
            steps.append(
                StepPlan(
                    step_id="step_1",
                    action="Semantic search for task context",
                    tool_name="semantic_search",
                    args={"query": subject or task[:120]},
                )
            )

        return steps
