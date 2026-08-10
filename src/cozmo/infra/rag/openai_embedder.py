"""
OpenAI-compatible embeddings client.

Works for OpenAI and Ollama (base_url switch), same pattern as chat.
"""

from __future__ import annotations

from openai import OpenAI

class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return list(resp.data[0].embedding)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Batch when possible; fall back per-item if provider rejects batches
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
            ordered = sorted(resp.data, key=lambda d: d.index)
            return [list(d.embedding) for d in ordered]
        except Exception:
            return [self.embed(t) for t in texts]
