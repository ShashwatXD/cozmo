"""
Golden-task evaluation harness.

Runs scripted agent scenarios against a fixture repo with StubLLM
or a live provider; asserts expected substrings.

Also runs deterministic groundedness + citation eval cases (no LLM).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from cozmo.app.agent import AgentRunner
from cozmo.app.groundedness import (
    GroundingContext,
    score_groundedness,
)
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.tools import ToolCall
from cozmo.infra.llm.factory import build_llm
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.telemetry.tracer import Tracer
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
    # A real model words things its own way: with --live, any one of these is enough.
    expect_any: tuple[str, ...] = ()


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
    duration_s: float = 0.0
    tokens: int = 0
    steps: int = 0
    turn_id: str | None = None


DEFAULT_CASES: list[EvalCase] = [
    EvalCase(
        id="read_math_utils",
        prompt="Read math_utils.py and explain any bug.",
        scripted_tool="read_file",
        scripted_args={"path": "math_utils.py"},
        final_text="off-by-one in add() at math_utils.py:3",
        expect_substring="off-by-one",
        expect_any=("a - b", "subtract", "minus", "off-by-one", "instead of add"),
    ),
    EvalCase(
        id="search_pattern",
        prompt="Search the repo for off-by-one",
        scripted_tool="search_repo",
        scripted_args={"query": "off-by-one"},
        final_text="found off-by-one comment in math_utils.py",
        expect_substring="off-by-one",
        expect_any=("math_utils.py",),
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
    from cozmo.app.groundedness import contexts_from_messages

    return contexts_from_messages(memory.snapshot())


def run_golden_eval(
    workdir: Path,
    *,
    settings: Settings | None = None,
    live: bool = False,
    tracer: Tracer | None = None,
) -> list[EvalResult]:
    settings = settings or Settings(provider="stub")
    results: list[EvalResult] = []
    for case in DEFAULT_CASES:
        started = time.monotonic()
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
                tracer=tracer,
                model_name=settings.model if live else "stub-model",
            )
            out = runner.run(case.prompt)
            detail = (out.final_text or "")[:200]
            ok = True
            answer = (out.final_text or "").lower()

            expected = case.expect_any if (live and case.expect_any) else ()
            if expected:
                if not any(e.lower() in answer for e in expected):
                    ok = False
                    detail = f"none of {list(expected)!r} in {out.final_text!r}"
            elif case.expect_substring and case.expect_substring.lower() not in answer:
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

            results.append(
                EvalResult(
                    case.id,
                    ok,
                    detail,
                    suite="golden",
                    duration_s=round(time.monotonic() - started, 2),
                    tokens=out.usage.total_tokens,
                    steps=out.steps,
                    turn_id=tracer.turn_id if tracer else None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                EvalResult(
                    case.id,
                    False,
                    str(exc),
                    suite="golden",
                    duration_s=round(time.monotonic() - started, 2),
                    turn_id=tracer.turn_id if tracer else None,
                )
            )
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
    tracer: Tracer | None = None,
) -> list[EvalResult]:
    """Run golden agent cases and optional groundedness/citation evals."""
    results = run_golden_eval(workdir, settings=settings, live=live, tracer=tracer)
    if include_groundedness:
        results.extend(run_groundedness_eval())
    return results


def save_eval_run(
    results: list[EvalResult],
    out_dir: Path,
    *,
    live: bool,
    provider: str,
    model: str,
    session_id: str | None = None,
) -> Path:
    """Write one eval run to *out_dir* as JSON, for `cozmo trace` to show and compare."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{started}.json"
    payload = {
        "id": started,
        "created_at": time.time(),
        "live": live,
        "provider": provider if live else "stub",
        "model": model if live else "stub-model",
        "session_id": session_id,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
