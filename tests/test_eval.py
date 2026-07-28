"""Eval harness tests (stub provider, no network)."""

from pathlib import Path

from cozmo.app.eval_runner import run_eval
from cozmo.settings import Settings

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


def test_eval_suite_passes_on_fixture() -> None:
    results = run_eval(FIXTURE, settings=Settings(provider="stub"), live=False)
    assert results
    assert all(r.passed for r in results), results
