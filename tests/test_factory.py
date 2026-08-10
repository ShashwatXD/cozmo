"""Factory returns retry-wrapped clients."""

import pytest

from cozmo.infra.llm.factory import build_llm
from cozmo.infra.llm.retrying import RetryingLLMClient
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.settings import Settings


def test_build_stub_wrapped() -> None:
    llm = build_llm(Settings(provider="stub", max_retries=3))
    assert isinstance(llm, RetryingLLMClient)


def test_build_stub_no_retry() -> None:
    llm = build_llm(Settings(provider="stub", max_retries=1))
    assert isinstance(llm, StubLLMClient)


def test_build_openai_requires_key() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        build_llm(Settings(provider="openai", api_key=None))


def test_build_ollama_wrapped() -> None:
    llm = build_llm(Settings(provider="ollama", model="qwen2.5:3b", max_retries=3))
    assert isinstance(llm, RetryingLLMClient)


def test_build_openrouter_uses_compat(monkeypatch) -> None:
    from cozmo.infra.llm.openai_compatible import OpenAICompatibleClient

    llm = build_llm(
        Settings(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            api_key="sk-or-test",
            max_retries=1,
        )
    )
    assert isinstance(llm, OpenAICompatibleClient)
    assert llm._client.base_url is not None  # type: ignore[attr-defined]


def test_build_anthropic_requires_key() -> None:
    with pytest.raises(ValueError, match="API_KEY"):
        build_llm(Settings(provider="anthropic", api_key=None))


def test_build_anthropic_client() -> None:
    from cozmo.infra.llm.anthropic_client import AnthropicClient

    llm = build_llm(
        Settings(provider="anthropic", model="claude-3-5-haiku-20241022", api_key="sk-ant", max_retries=1)
    )
    assert isinstance(llm, AnthropicClient)
