"""
Lexical groundedness + citation validation for agent answers.

Deterministic faithfulness check: how much of an answer is recoverable from
the context the agent was actually given. No LLM judge required — CI-safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"([A-Za-z0-9_@./-]+\.[A-Za-z]{1,6})(?::(\d+)(?:\s*[-–]\s*(\d+))?)?")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "will",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class GroundingContext:
    path: str
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True)
class Citation:
    raw: str
    path: str
    start_line: int | None
    end_line: int | None


@dataclass(frozen=True)
class ClaimScore:
    claim: str
    supported: bool
    overlap: float


@dataclass(frozen=True)
class GroundednessReport:
    score: int
    support_score: float
    citation_score: float | None
    claims: list[ClaimScore] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    citations: dict[str, int | list[Citation]] = field(default_factory=dict)
    summary: str = ""


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _expand_identifier(raw: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    parts = [p.lower() for p in re.split(r"[^A-Za-z0-9]+", spaced) if p]
    expanded = set(parts)
    if len(parts) > 1:
        expanded.add("".join(parts))
    return list(expanded)


def content_words(text: str) -> list[str]:
    words: list[str] = []
    for raw in re.split(r"[^A-Za-z0-9_]+", text):
        if not raw:
            continue
        for part in _expand_identifier(raw):
            if len(part) > 1 and part not in STOPWORDS:
                words.append(_stem(part))
    return words


def split_claims(answer: str) -> list[str]:
    """Split prose into claim-sized units: sentences, list items, and lines."""
    claims: list[str] = []
    for line in answer.split("\n"):
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        for piece in re.split(r"(?<=[.!?])\s+", cleaned):
            s = piece.strip()
            if len(content_words(s)) >= 3:
                claims.append(s)
    return claims


_CODE_EXT = re.compile(
    r"\.(py|ts|tsx|js|jsx|go|rs|md|json|yaml|yml|toml|txt)$",
    re.IGNORECASE,
)


def _looks_like_path(path: str) -> bool:
    return "/" in path or bool(_CODE_EXT.search(path))


def extract_citations(answer: str) -> list[Citation]:
    found: list[Citation] = []
    seen: set[str] = set()
    for match in CITATION_RE.finditer(answer):
        raw = match.group(0)
        path = match.group(1) or ""
        if not _looks_like_path(path):
            continue
        if raw in seen:
            continue
        seen.add(raw)
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else None
        found.append(
            Citation(
                raw=raw,
                path=path,
                start_line=start,
                end_line=end,
            )
        )
    return found


def _matches_context_path(citation_path: str, context_path: str) -> bool:
    if citation_path == context_path:
        return True
    return context_path.endswith(f"/{citation_path}") or citation_path.endswith(f"/{context_path}")


def validate_citation(citation: Citation, contexts: list[GroundingContext]) -> bool:
    for_path = [c for c in contexts if _matches_context_path(citation.path, c.path)]
    if not for_path:
        return False
    if citation.start_line is None:
        return True
    start = citation.start_line
    end = citation.end_line if citation.end_line is not None else citation.start_line
    return any(start >= c.start_line and end <= c.end_line for c in for_path)


def score_groundedness(
    *,
    answer: str,
    contexts: list[GroundingContext],
    threshold: float = 0.6,
) -> GroundednessReport:
    """
    Heuristic faithfulness check: how much of an answer is recoverable from
    the context the agent was actually given.
    """
    claims = split_claims(answer)
    context_vocab = set(content_words("\n".join(c.content for c in contexts)))

    scored: list[ClaimScore] = []
    for claim in claims:
        words = content_words(claim)
        hits = sum(1 for w in words if w in context_vocab)
        overlap = hits / len(words) if words else 0.0
        scored.append(
            ClaimScore(
                claim=claim,
                supported=overlap >= threshold,
                overlap=round(overlap, 3),
            )
        )

    citations = extract_citations(answer)
    invalid = [c for c in citations if not validate_citation(c, contexts)]

    support_score = sum(1 for c in scored if c.supported) / len(scored) if scored else 0.0
    citation_score: float | None
    if citations:
        citation_score = (len(citations) - len(invalid)) / len(citations)
    else:
        citation_score = None

    if citation_score is None:
        combined = support_score
    else:
        combined = support_score * 0.7 + citation_score * 0.3

    score = round(combined * 100)
    unsupported = [c.claim for c in scored if not c.supported]

    valid_citations = len(citations) - len(invalid)
    citation_note = (
        "no citations found"
        if citation_score is None
        else f"{valid_citations}/{len(citations)} citations valid"
    )

    supported_n = len(scored) - len(unsupported)
    summary = (
        f"Groundedness {score}/100 — {supported_n}/{len(scored)} claims supported, {citation_note}."
    )

    return GroundednessReport(
        score=score,
        support_score=round(support_score, 3),
        citation_score=(round(citation_score, 3) if citation_score is not None else None),
        claims=scored,
        unsupported=unsupported,
        citations={
            "total": len(citations),
            "valid": valid_citations,
            "invalid": invalid,
        },
        summary=summary,
    )


def contexts_from_tool_results(
    tool_name: str,
    args: str,
    content: str,
    *,
    default_path: str = "",
) -> list[GroundingContext]:
    """
    Build grounding contexts from a tool result (read_file / search / semantic).
    Best-effort path and line extraction for post-turn eval.
    """
    path = default_path
    start_line = 1
    end_line = 1

    if tool_name == "read_file":
        try:
            import json

            parsed = json.loads(args) if args.strip().startswith("{") else {}
            path = str(parsed.get("path") or path)
            if parsed.get("start_line"):
                start_line = int(parsed["start_line"])
            if parsed.get("end_line"):
                end_line = int(parsed["end_line"])
            elif parsed.get("around_line"):
                center = int(parsed["around_line"])
                window = int(parsed.get("window") or 80)
                start_line = max(1, center - window)
                end_line = center + window
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        lines = content.count("\n") + 1 if content else 1
        if end_line <= start_line:
            end_line = start_line + lines - 1
        return [
            GroundingContext(
                path=path,
                start_line=start_line,
                end_line=end_line,
                content=content[:8000],
            )
        ]

    if tool_name in {"search_repo", "semantic_search"}:
        chunks: list[GroundingContext] = []
        for line in content.splitlines()[:40]:
            m = re.match(r"([^:]+):(\d+)", line)
            if m:
                chunks.append(
                    GroundingContext(
                        path=m.group(1),
                        start_line=int(m.group(2)),
                        end_line=int(m.group(2)),
                        content=line,
                    )
                )
        if chunks:
            return chunks

    if path or content.strip():
        return [
            GroundingContext(
                path=path or "(tool)",
                start_line=1,
                end_line=max(1, content.count("\n") + 1),
                content=content[:8000],
            )
        ]
    return []
