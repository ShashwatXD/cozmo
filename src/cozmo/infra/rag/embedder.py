"""
Embedders.

What: HashingEmbedder (local, no network) for Core RAG demos + tests.
Why: teach the RAG pipeline without pulling a second Ollama model.
Layer: infra.

Note: hashing is lexical overlap, not true semantics. Same pipeline
swaps to a real model later (interview talking point).
"""

from __future__ import annotations

import hashlib
import math
import re


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+")


class HashingEmbedder:
    """
    Bag-of-tokens hashed into a fixed vector (dim=256).

    when queries share words with chunks (great for code identifiers).
    """

    def __init__(self, *, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
