"""
OpenAI-compatible embeddings client.

Works for OpenAI and Ollama (base_url switch), same pattern as chat.
"""

from __future__ import annotations

import logging

from openai import APIStatusError, OpenAI

logger = logging.getLogger(__name__)

# OpenAI accepts up to 2048 inputs per request; staying well below keeps each
# request under the per-request token cap for ~600-char code chunks.
MAX_INPUTS_PER_REQUEST = 128


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_batch: int = MAX_INPUTS_PER_REQUEST,
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
        self._max_batch = max(1, max_batch)

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return list(resp.data[0].embedding)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._max_batch):
            out.extend(self._embed_batch(texts[i : i + self._max_batch]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._client.embeddings.create(model=self._model, input=texts)
        except APIStatusError as exc:
            # Some OpenAI-compatible servers reject list input (4xx). Auth and
            # rate-limit errors are not batching problems, so let them surface.
            if exc.status_code in {401, 403, 429} or exc.status_code >= 500:
                raise
            logger.info("batch embedding rejected (%s); falling back per item", exc.status_code)
            return [self.embed(t) for t in texts]
        ordered = sorted(resp.data, key=lambda d: d.index)
        return [list(d.embedding) for d in ordered]
