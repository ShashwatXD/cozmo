"""
Simple heuristic planner for code intelligence tasks.

What: Maps a natural-language task to a sequence of StepPlans using keyword matching.
Why: Gives the execution engine a structured plan without requiring an LLM call.
Layer: runtime (orchestration helper, no vendor SDKs).
How to test: Unit-test plan() with various task strings; assert correct tool suggestions.
"""

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


# Keyword patterns → (tool_name, action_description, arg_extractor)
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # symbol / definition lookup
    (
        re.compile(r"(?:find|where|locate|look\s?up).*(?:defin|declar|symbol)", re.I),
        "symbol_search",
        "Search for symbol definition",
    ),
    (
        re.compile(r"(?:defin|declar).*(?:of|for)\s+(\w+)", re.I),
        "symbol_search",
        "Search for symbol definition",
    ),
    # references / callers
    (
        re.compile(r"(?:who|what|where).*(?:call|use|refer|import)", re.I),
        "find_references",
        "Find references to symbol",
    ),
    (
        re.compile(r"(?:find|show|list).*(?:refer|call|usage)", re.I),
        "find_references",
        "Find references to symbol",
    ),
    # graph / dependency queries
    (
        re.compile(r"(?:depend|import|call)\s*(?:graph|tree|chain)", re.I),
        "get_codebase_graph",
        "Retrieve codebase graph",
    ),
    # semantic search fallback
    (
        re.compile(r"(?:search|find|look)\s+(?:for\s+)?(?:code|function|class)", re.I),
        "semantic_search",
        "Semantic search for code",
    ),
    # file read
    (
        re.compile(r"(?:read|show|open|cat|view)\s+(?:file|content)", re.I),
        "read_file",
        "Read file contents",
    ),
]


def _extract_subject(task: str) -> str:
    """Try to pull the main subject (symbol / path) from the task string."""
    # Look for quoted strings first
    m = re.search(r"['\"]([^'\"]+)['\"]", task)
    if m:
        return m.group(1)
    # Look for backtick-wrapped identifiers
    m = re.search(r"`([^`]+)`", task)
    if m:
        return m.group(1)
    return ""


class Planner:
    """Keyword-based planner that suggests code intelligence tools for a task."""

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
            if tool_name == "symbol_search" and subject:
                args["query"] = subject
            elif tool_name == "find_references" and subject:
                args["symbol_name"] = subject
            elif tool_name == "get_codebase_graph":
                # infer graph type from task
                if re.search(r"import", task, re.I):
                    args["graph_type"] = "import"
                elif re.search(r"call", task, re.I):
                    args["graph_type"] = "call"
                else:
                    args["graph_type"] = "dependency"

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

        # Fallback: if no pattern matched, suggest semantic_search if available
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
