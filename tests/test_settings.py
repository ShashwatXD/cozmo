"""Phase 0: package imports and settings load."""

from cozmo import __version__
from cozmo.settings import Settings, load_settings


def test_version_defined() -> None:
    assert __version__


def test_settings_defaults() -> None:
    # Explicit api_key=None so a developer .env does not fail CI/local asserts.
    s = Settings(provider="stub", model="stub-model", api_key=None)
    assert s.provider == "stub"
    assert s.api_key is None
    assert s.vector_backend == "json"
    assert s.history_enabled is True


def test_load_settings_does_not_crash() -> None:
    # Uses env / optional .env; should always construct
    s = load_settings()
    assert s.provider
