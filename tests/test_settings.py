"""Phase 0: package imports and settings load."""

from cozmo import __version__
from cozmo.settings import Settings, load_settings


def test_version_defined() -> None:
    assert __version__


def test_settings_defaults() -> None:
    s = Settings(provider="stub", model="stub-model")
    assert s.provider == "stub"
    assert s.openai_api_key is None


def test_load_settings_does_not_crash() -> None:
    # Uses env / optional .env; should always construct
    s = load_settings()
    assert s.provider
