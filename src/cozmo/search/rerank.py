"""Lexical reranker — no model deps; boosts token overlap + whitespace-insensitive hits."""

from __future__ import annotations

import re
from typing import Sequence

from cozmo.domain.rag import RetrievalCandidate

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]+|[0-9]+|[^\s\w]", re.UNICODE)


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s) if t.strip()}


class LexicalReranker:
    """Rerank with token overlap and whitespace-insensitive substring match."""

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
        q_tokens = _tokens(query)
        scored: list[RetrievalCandidate] = []
        for c in candidates:
            text = c.text or ""
            t_compact = _compact(text)
            t_tokens = _tokens(text)
            score = 0.0
            score += 0.15 * float(c.score)
            if q_compact and q_compact in t_compact:
                score += 3.0
                score += 1.0 / (1.0 + abs(len(t_compact) - len(q_compact)) / 40.0)
            if q_tokens and t_tokens:
                overlap = len(q_tokens & t_tokens) / len(q_tokens)
                score += 2.0 * overlap
            path_l = c.path.lower()
            for tok in q_tokens:
                if len(tok) > 2 and tok in path_l:
                    score += 0.4
            scored.append(
                RetrievalCandidate(
                    path=c.path,
                    start_line=c.start_line,
                    text=c.text,
                    score=score,
                    chunk_id=c.chunk_id,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]
