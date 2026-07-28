"""
Build Embedder from Settings - hash | openai | ollama.
"""

from __future__ import annotations

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.embedder import HashingEmbedder
from cozmo.infra.rag.openai_embedder import OpenAICompatibleEmbedder
from cozmo.settings import Settings

_OLLAMA_BASE = "http://127.0.0.1:11434/v1"


def build_embedder(settings: Settings) -> Embedder:
    backend = (settings.embedder or "hash").lower()
    if backend == "hash":
        return HashingEmbedder()

    if backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("COZMO_OPENAI_API_KEY required for embedder=openai")
        return OpenAICompatibleEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_s=settings.timeout_s,
        )

    if backend == "ollama":
        return OpenAICompatibleEmbedder(
            model=settings.embedding_model,
            api_key=settings.openai_api_key or "ollama",
            base_url=settings.openai_base_url or _OLLAMA_BASE,
            timeout_s=settings.timeout_s,
        )

    raise ValueError(
        f"Unknown embedder '{settings.embedder}'. Supported: hash, openai, ollama"
    )
