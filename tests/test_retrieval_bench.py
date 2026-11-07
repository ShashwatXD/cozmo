"""Retrieval quality gate: fails CI if ranking quality regresses on the labeled set."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from cozmo.app.retrieval_bench import BenchQuery, load_queries, run_retrieval_bench
from cozmo.infra.rag.embedder import StubEmbedder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from run_retrieval import CORPUS_REF  # noqa: E402


def _pinned_corpus(tmp_path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git not available")
    tar = tmp_path / "corpus.tar"
    with tar.open("wb") as fh:
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "archive", CORPUS_REF, "src/cozmo"],
            stdout=fh,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if proc.returncode != 0:
        pytest.skip(f"corpus ref {CORPUS_REF} not in this checkout (shallow clone?)")
    with tarfile.open(tar) as t:
        t.extractall(tmp_path, filter="data")
    return tmp_path / "src" / "cozmo"


def test_metrics_on_toy_corpus(tmp_path: Path) -> None:
    (tmp_path / "guard.py").write_text("class WorkspaceGuard:\n    def resolve(self): ...\n")
    (tmp_path / "retry.py").write_text("def retry_with_backoff():\n    pass\n")
    queries = [
        BenchQuery("q1", "WorkspaceGuard resolve", ("guard.py",)),
        BenchQuery("q2", "retry backoff", ("retry.py",)),
    ]
    report = run_retrieval_bench(tmp_path, queries, StubEmbedder(), embedder_name="stub", repeats=1)
    scores = {s.strategy: s for s in report.scores}
    assert scores["pipeline"].recall[1] == 1.0
    assert scores["bm25"].mrr == 1.0
    assert "pipeline" in report.to_markdown()
    assert not (tmp_path / ".cozmo").exists()  # benchmark leaves the corpus clean


def test_retrieval_quality_gate(tmp_path: Path) -> None:
    """
    Offline gate with the deterministic stub embedder (no network).

    Real-embedding numbers live in benchmarks/results/; these floors only catch
    regressions in the lexical half and in fusion / rerank plumbing.
    """
    corpus = _pinned_corpus(tmp_path)
    queries = load_queries(ROOT / "benchmarks" / "queries.jsonl")
    report = run_retrieval_bench(corpus, queries, StubEmbedder(), embedder_name="stub", repeats=1)
    s = {x.strategy: x for x in report.scores}
    assert s["bm25"].recall[5] >= 0.95
    assert s["bm25"].mrr >= 0.65
    assert s["pipeline"].recall[5] >= 0.85
    assert s["pipeline"].mrr >= 0.60
