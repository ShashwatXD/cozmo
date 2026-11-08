"""Keep tests away from the developer's real trust file."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_trust_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COZMO_TRUST_FILE", str(tmp_path / "trust.json"))
