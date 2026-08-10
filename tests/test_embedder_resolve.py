"""Auto embedder selection."""

from cozmo.infra.rag.embedder import HashingEmbedder
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


def test_auto_stub_uses_hash() -> None:
    backend, model = resolve_embedder(Settings(provider="stub", embedder="auto"))
    assert backend == "hash"
    assert isinstance(build_embedder(Settings(provider="stub")), HashingEmbedder)


def test_explicit_hash_honored() -> None:
    backend, _ = resolve_embedder(
        Settings(provider="openai", api_key="sk", embedder="hash")
    )
    assert backend == "hash"


def test_openai_without_key_falls_back_to_hash_instance() -> None:
    emb = build_embedder(Settings(provider="openai", api_key=None, embedder="openai"))
    assert isinstance(emb, HashingEmbedder)
