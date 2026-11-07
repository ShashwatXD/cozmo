"""
Golden-task evaluation harness.

Runs scripted agent scenarios against a fixture repo with StubLLM
or a live provider; asserts expected substrings.

Also runs deterministic groundedness + citation eval cases (no LLM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.app.groundedness import (
    GroundingContext,
    score_groundedness,
)
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Role
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
    min_groundedness: int | None = None


@dataclass(frozen=True)
class GroundednessCase:
    id: str
    answer: str
    contexts: tuple[GroundingContext, ...]
    min_score: int = 60
    max_unsupported: int = 0
    min_invalid_citations: int = 0
    min_unsupported: int = 0


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    detail: str
    suite: str = "golden"


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        id="read_math_utils",
        prompt="Read math_utils.py and explain any bug.",
        scripted_tool="read_file",
        scripted_args={"path": "math_utils.py"},
        final_text="off-by-one in add() at math_utils.py:3",
        expect_substring="off-by-one",
    ),
    EvalCase(
        id="search_pattern",
        prompt="Search the repo for off-by-one",
        scripted_tool="search_repo",
        scripted_args={"query": "off-by-one"},
        final_text="found off-by-one comment in math_utils.py",
        expect_substring="off-by-one",
    ),
]

GROUNDEDNESS_CASES: list[GroundednessCase] = [
    GroundednessCase(
        id="supported_claim",
        answer="Chunks are inserted into rag_chunks with an embedding_json column.",
        contexts=(
            GroundingContext(
                path="server/src/rag/indexer.ts",
                start_line=200,
                end_line=240,
                content=(
                    'const insert = sqlite.prepare("INSERT INTO rag_chunks '
                    "(id, path, start_line, end_line, content, embedding_json, indexed_at) "
                    'VALUES (?, ?, ?, ?, ?, ?, ?)");'
                ),
            ),
            GroundingContext(
                path="server/src/rag/retrieve.ts",
                start_line=20,
                end_line=55,
                content=(
                    "const hits = withVectors.map((c) => ({ path: c.path, "
                    "score: cosineSimilarity(qVec, c.embedding) }))"
                ),
            ),
        ),
        min_score=60,
    ),
    GroundednessCase(
        id="unsupported_claim",
        answer=(
            "Vectors are sharded across a Redis cluster using HNSW graphs for approximate search."
        ),
        contexts=(
            GroundingContext(
                path="server/src/rag/indexer.ts",
                start_line=200,
                end_line=240,
                content="INSERT INTO rag_chunks (id, path, embedding_json)",
            ),
        ),
        min_score=0,
        max_unsupported=99,
        min_unsupported=1,
    ),
    GroundednessCase(
        id="valid_citation",
        answer=(
            "Each result score uses cosineSimilarity on c.embedding. "
            "See server/src/rag/retrieve.ts:20-55."
        ),
        contexts=(
            GroundingContext(
                path="server/src/rag/retrieve.ts",
                start_line=20,
                end_line=55,
                content="score: cosineSimilarity(qVec, c.embedding)",
            ),
        ),
        min_score=60,
        max_unsupported=1,
    ),
    GroundednessCase(
        id="fabricated_citation",
        answer=(
            "Each result score uses cosineSimilarity on c.embedding. "
            "See server/src/rag/pinecone.ts:20-55."
        ),
        contexts=(
            GroundingContext(
                path="server/src/rag/retrieve.ts",
                start_line=20,
                end_line=55,
                content="score: cosineSimilarity(qVec, c.embedding)",
            ),
        ),
        min_score=0,
        max_unsupported=99,
        min_invalid_citations=1,
    ),
]


def _collect_contexts_from_memory(memory: ConversationMemory) -> list[GroundingContext]:
    """Extract grounding contexts from TOOL messages in conversation memory."""
    from cozmo.app.groundedness import contexts_from_tool_results

    contexts: list[GroundingContext] = []
    pending_calls: dict[str, tuple[str, str]] = {}

    for msg in memory.snapshot():
        if msg.role == Role.ASSISTANT and msg.tool_calls:
            for tc in msg.tool_calls:
                pending_calls[tc.id] = (tc.name, tc.arguments)
        elif msg.role == Role.TOOL and msg.tool_call_id:
            name_args = pending_calls.get(msg.tool_call_id)
            if name_args:
                name, args = name_args
                contexts.extend(contexts_from_tool_results(name, args, msg.content))
    return contexts


def run_golden_eval(
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
                calls: tuple[ToolCall, ...] = ()
                if case.scripted_tool:
                    calls = (
                        ToolCall(
                            id=f"eval_{case.id}",
                            name=case.scripted_tool,
                            arguments=json.dumps(case.scripted_args or {}),
                        ),
                    )
                llm = StubLLMClient(scripted_tool_calls=calls, final_text=case.final_text)
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
            detail = (out.final_text or "")[:200]
            ok = True

            if (
                case.expect_substring
                and case.expect_substring.lower() not in (out.final_text or "").lower()
            ):
                ok = False
                detail = f"missing {case.expect_substring!r} in {out.final_text!r}"

            if case.min_groundedness is not None:
                contexts = _collect_contexts_from_memory(runner.memory)
                if contexts:
                    report = score_groundedness(
                        answer=out.final_text or "",
                        contexts=contexts,
                    )
                    detail = f"{report.summary} | {detail}"
                    if report.score < case.min_groundedness:
                        ok = False
                else:
                    ok = False
                    detail = f"no tool context to score | {detail}"

            results.append(EvalResult(case.id, ok, detail, suite="golden"))
        except Exception as exc:  # noqa: BLE001
            results.append(EvalResult(case.id, False, str(exc), suite="golden"))
    return results


def run_groundedness_eval() -> list[EvalResult]:
    """Deterministic faithfulness cases — no agent, no network."""
    results: list[EvalResult] = []
    for case in GROUNDEDNESS_CASES:
        report = score_groundedness(
            answer=case.answer,
            contexts=list(case.contexts),
        )
        invalid = report.citations.get("invalid", [])
        invalid_n = len(invalid) if isinstance(invalid, list) else 0

        ok = report.score >= case.min_score
        if len(report.unsupported) > case.max_unsupported:
            ok = False
        if len(report.unsupported) < case.min_unsupported:
            ok = False
        if invalid_n < case.min_invalid_citations:
            ok = False

        results.append(
            EvalResult(
                case.id,
                ok,
                report.summary,
                suite="groundedness",
            )
        )
    return results


def run_eval(
    workdir: Path,
    *,
    settings: Settings | None = None,
    live: bool = False,
    include_groundedness: bool = True,
) -> list[EvalResult]:
    """Run golden agent cases and optional groundedness/citation evals."""
    results = run_golden_eval(workdir, settings=settings, live=live)
    if include_groundedness:
        results.extend(run_groundedness_eval())
    return results
