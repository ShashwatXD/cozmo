"""
Ports for RAG.

What: Embedder Protocol.
Why: swap hash/Ollama/OpenAI embeddings without touching the agent.
Layer: domain.
Flutter: abstract EmbeddingRepository.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Flutter: abstract class EmbeddingRepo { Future<List<double>> embed(String text); }"""

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...
