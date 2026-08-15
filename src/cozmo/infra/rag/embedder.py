"""Local stub embedder for tests / provider=stub only."""

from __future__ import annotations

import math
import re

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+")


class StubEmbedder:
    """
    Deterministic bag-of-tokens vectors for offline tests.

    Production indexing uses OpenAICompatibleEmbedder (OpenAI / Ollama).
    """

    def __init__(self, *, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            idx = sum(ord(c) for c in tok) % self.dim
            sign = 1.0 if (len(tok) % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
