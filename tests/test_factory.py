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
        build_llm(Settings(provider="openai", openai_api_key=None))


def test_build_ollama_wrapped() -> None:
    llm = build_llm(Settings(provider="ollama", model="qwen2.5:3b", max_retries=3))
    assert isinstance(llm, RetryingLLMClient)


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        build_llm(Settings(provider="nope"))
