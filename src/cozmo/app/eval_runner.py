"""
Golden-task evaluation harness.

Runs scripted agent scenarios against a fixture repo with StubLLM
or a live provider; asserts expected substrings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.factory import build_llm
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor
from cozmo.settings import Settings


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expect_substring: str | None = None
    scripted_tool: str | None = None
    scripted_args: dict | None = None
    final_text: str = "eval ok"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    detail: str


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        id="read_math_utils",
        prompt="Read math_utils.py and explain any bug.",
        scripted_tool="read_file",
        scripted_args={"path": "math_utils.py"},
        final_text="off-by-one in add()",
        expect_substring="off-by-one",
    ),
    EvalCase(
        id="search_pattern",
        prompt="Search the repo for off-by-one",
        scripted_tool="search_repo",
        scripted_args={"query": "off-by-one"},
        final_text="found off-by-one comment",
        expect_substring="off-by-one",
    ),
]


def run_eval(
    workdir: Path,
    *,
    settings: Settings | None = None,
    live: bool = False,
) -> list[EvalResult]:
    settings = settings or Settings(provider="stub")
    results: list[EvalResult] = []
    for case in DEFAULT_CASES:
        try:
            if live:
                llm = build_llm(settings)
            else:
                calls = ()
                if case.scripted_tool:
                    calls = (
                        ToolCall(
                            id=f"eval_{case.id}",
                            name=case.scripted_tool,
                            arguments=json.dumps(case.scripted_args or {}),
                        ),
                    )
                llm = StubLLMClient(
                    scripted_tool_calls=calls, final_text=case.final_text
                )
            guard = WorkspaceGuard(workdir, allow_write=False, allow_shell=False)
            reg = build_default_registry(guard)
            runner = AgentRunner(
                llm,
                reg,
                ToolExecutor(reg),
                memory=ConversationMemory(max_messages=20),
                max_steps=6,
            )
            out = runner.run(case.prompt)
            ok = True
            detail = (out.final_text or "")[:200]
            if case.expect_substring:
                if case.expect_substring.lower() not in detail.lower():
                    ok = False
                    detail = f"missing {case.expect_substring!r} in {out.final_text!r}"
            results.append(EvalResult(case.id, ok, detail))
        except Exception as exc:  # noqa: BLE001
            results.append(EvalResult(case.id, False, str(exc)))
    return results
