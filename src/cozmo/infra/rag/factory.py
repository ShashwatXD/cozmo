"""Build Embedder from Settings — auto-picks best backend for the provider."""

from __future__ import annotations

from pathlib import Path

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.embedder import HashingEmbedder
from cozmo.infra.rag.openai_embedder import OpenAICompatibleEmbedder
from cozmo.settings import Settings

_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_OLLAMA_EMBED_MODEL = "nomic-embed-text"


def resolve_embedder(settings: Settings) -> tuple[str, str]:
    """
    (backend, embedding_model).

    Default embedder=auto:
      openai/openrouter + api_key → openai semantic embeddings
      ollama                      → local nomic-embed-text
      else                        → hash (offline fallback)

    Explicit hash|openai|ollama in config still works for tests/power users.
    """
    explicit = (settings.embedder or "auto").lower().strip()
    if explicit == "hash":
        return "hash", "hash"
    if explicit == "openai":
        return "openai", settings.embedding_model or _OPENAI_EMBED_MODEL
    if explicit == "ollama":
        return "ollama", settings.embedding_model or _OLLAMA_EMBED_MODEL

    if settings.provider in {"openai", "openrouter"} and settings.api_key:
        return "openai", _OPENAI_EMBED_MODEL
    if settings.provider == "ollama":
        return "ollama", _OLLAMA_EMBED_MODEL
    return "hash", "hash"


def build_embedder(settings: Settings) -> Embedder:
    backend, model = resolve_embedder(settings)

    if backend == "hash":
        return HashingEmbedder()

    if backend == "openai":
        if not settings.api_key:
            return HashingEmbedder()
        return OpenAICompatibleEmbedder(
            model=model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout_s=settings.timeout_s,
        )

    if backend == "ollama":
        return OpenAICompatibleEmbedder(
            model=model,
            api_key=settings.api_key or "ollama",
            base_url=settings.base_url or _OLLAMA_BASE,
            timeout_s=settings.timeout_s,
        )

    return HashingEmbedder()


def build_vector_store(settings: Settings, workdir: Path):
    """Pick JSON or Chroma vector backend from Settings.vector_backend."""
    from pathlib import Path as PathType

    from cozmo.domain.ports_rag import VectorStore
    from cozmo.infra.rag.paths import chroma_dir, index_path
    from cozmo.infra.rag.store import JsonVectorStore

    root = workdir if isinstance(workdir, PathType) else PathType(workdir)
    backend = (getattr(settings, "vector_backend", None) or "json").lower().strip()
    if backend == "chroma":
        from cozmo.infra.rag.chroma_store import ChromaVectorStore

        store: VectorStore = ChromaVectorStore.load(
            index_path(root), persist_dir=chroma_dir(root)
        )
        return store
    return JsonVectorStore.load(index_path(root))
