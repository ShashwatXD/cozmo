"""Model catalog unit tests (mocked HTTP)."""

from __future__ import annotations

import httpx

from cozmo.infra.llm.model_catalog import (
    OPENAI_FAVORITES,
    validate_and_list_openai,
    validate_and_list_anthropic,
    _merge_favorites,
)


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_merge_favorites_puts_stars_first() -> None:
    merged = _merge_favorites(["z-model", "gpt-4o-mini"], ("gpt-4o", "gpt-4o-mini"))
    assert merged[0] == "gpt-4o"
    assert "gpt-4o-mini" in merged
    assert "z-model" in merged


def test_openai_invalid_key(monkeypatch) -> None:
    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp(401, text="unauthorized")

    monkeypatch.setattr(httpx, "Client", _Client)
    result = validate_and_list_openai("sk-bad")
    assert result.ok is False
    assert "Invalid" in result.error


def test_openai_lists_models(monkeypatch) -> None:
    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp(
                200,
                {
                    "data": [
                        {"id": "gpt-4o-mini"},
                        {"id": "text-embedding-3-small"},
                        {"id": "gpt-4o"},
                    ]
                },
            )

    monkeypatch.setattr(httpx, "Client", _Client)
    result = validate_and_list_openai("sk-good")
    assert result.ok is True
    assert "gpt-4o" in result.models
    assert result.models[0] in OPENAI_FAVORITES


def test_anthropic_fallback_favorites_on_404(monkeypatch) -> None:
    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, *a, **k):
            return _Resp(404, text="not found")

    monkeypatch.setattr(httpx, "Client", _Client)
    result = validate_and_list_anthropic("sk-ant")
    assert result.ok is True
    assert len(result.models) > 0
