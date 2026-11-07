"""Lexical reranker: no model deps; refines hybrid recall with exact-match evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence

from cozmo.domain.rag import RetrievalCandidate

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+|[^\s\w]", re.UNICODE)

# Words that carry no signal for locating code; matching them should not
# outrank what the hybrid recall stage already found.
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "by",
        "is",
        "are",
        "be",
        "it",
        "its",
        "this",
        "that",
        "when",
        "what",
        "how",
        "does",
        "do",
        "which",
        "into",
        "over",
        "before",
        "after",
        "as",
        "at",
        "not",
        "only",
        "same",
    ]
)


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s) if t.strip()}


def _content_tokens(s: str) -> set[str]:
    return {t for t in _tokens(s) if len(t) > 1 and t not in _STOP and t.isidentifier()}


class LexicalReranker:
    """
    Rerank hybrid candidates with lexical evidence while keeping recall order as a prior.

    Score = normalized recall score (dominant prior)
          + exact whitespace-insensitive query match (identifiers, snippets)
          + share of content words present in the chunk
          + small bonus for query words in the file path.

    The previous version scaled the recall score by 0.15. With RRF scores around
    0.03 that erased the hybrid ranking, so the final order was purely lexical.
    """

    def __init__(
        self,
        *,
        prior_weight: float = 1.0,
        exact_weight: float = 0.6,
        overlap_weight: float = 0.3,
        path_weight: float = 0.1,
    ) -> None:
        self._w_prior = prior_weight
        self._w_exact = exact_weight
        self._w_overlap = overlap_weight
        self._w_path = path_weight

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        top_k: int = 10,
    ) -> list[RetrievalCandidate]:
        if not candidates:
            return []
        q_compact = _compact(query)
        q_tokens = _content_tokens(query) or _tokens(query)
        top = max((c.score for c in candidates), default=0.0)
        scale = top if top > 0 else 1.0

        scored: list[tuple[float, int, RetrievalCandidate]] = []
        for order, c in enumerate(candidates):
            text = c.text or ""
            score = self._w_prior * (max(c.score, 0.0) / scale)
            if q_compact and len(q_compact) > 2 and q_compact in _compact(text):
                score += self._w_exact
            if q_tokens:
                score += self._w_overlap * len(q_tokens & _tokens(text)) / len(q_tokens)
                path_l = c.path.lower()
                hits = sum(1 for tok in q_tokens if len(tok) > 2 and tok in path_l)
                score += self._w_path * min(hits, 3)
            scored.append((score, order, c))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            RetrievalCandidate(
                path=c.path,
                start_line=c.start_line,
                text=c.text,
                score=score,
                chunk_id=c.chunk_id,
            )
            for score, _, c in scored[:top_k]
        ]
