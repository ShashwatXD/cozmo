"""Auto embedder selection."""

import pytest

from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.factory import build_embedder, resolve_embedder
from cozmo.infra.rag.openai_embedder import OpenAICompatibleEmbedder
from cozmo.settings import Settings


def test_auto_openai_uses_semantic() -> None:
    backend, model = resolve_embedder(
        Settings(provider="openai", api_key="sk-test", embedder="auto")
    )
    assert backend == "openai"
    assert model == "text-embedding-3-small"


def test_auto_ollama_uses_local_semantic() -> None:
    backend, model = resolve_embedder(Settings(provider="ollama", embedder="auto"))
    assert backend == "ollama"
    assert model == "nomic-embed-text"


def test_auto_stub_uses_stub_embedder() -> None:
    backend, model = resolve_embedder(Settings(provider="stub", embedder="auto"))
    assert backend == "stub"
    assert isinstance(build_embedder(Settings(provider="stub")), StubEmbedder)


def test_explicit_stub_honored() -> None:
    backend, _ = resolve_embedder(
        Settings(provider="openai", api_key="sk", embedder="stub")
    )
    assert backend == "stub"


def test_openai_without_key_raises() -> None:
    with pytest.raises(ValueError, match="COZMO_API_KEY"):
        build_embedder(Settings(provider="openai", api_key=None, embedder="openai"))


def test_build_openai_embedder() -> None:
    emb = build_embedder(
        Settings(provider="openai", api_key="sk-test", embedder="openai")
    )
    assert isinstance(emb, OpenAICompatibleEmbedder)
