"""Expand retrieval hits with surrounding source lines."""

from __future__ import annotations

from cozmo.domain.rag import ExpandedHit, RetrievalCandidate


def expand_hits(
    hits: list[RetrievalCandidate],
    sources: dict[str, str],
    *,
    before: int = 12,
    after: int = 12,
) -> list[ExpandedHit]:
    """
    Grow each hit to nearby lines from *sources* (path → file text).

    If the file is missing, returns the original chunk text unchanged.
    """
    expanded: list[ExpandedHit] = []
    for hit in hits:
        src = sources.get(hit.path)
        if not src:
            end = hit.start_line
            expanded.append(
                ExpandedHit(
                    path=hit.path,
                    start_line=hit.start_line,
                    end_line=end,
                    text=hit.text,
                    score=hit.score,
                    original_text=hit.text,
                )
            )
            continue
        lines = src.splitlines()
        if not lines:
            expanded.append(
                ExpandedHit(
                    path=hit.path,
                    start_line=hit.start_line,
                    end_line=hit.start_line,
                    text=hit.text,
                    score=hit.score,
                    original_text=hit.text,
                )
            )
            continue
        # start_line is 1-based
        center = max(1, hit.start_line)
        start = max(1, center - before)
        end = min(len(lines), center + after)
        # If chunk is multi-line, try to cover more of it
        chunk_lines = hit.text.count("\n") + 1
        end = min(len(lines), max(end, center + chunk_lines + after // 2))
        body = "\n".join(lines[start - 1 : end])
        expanded.append(
            ExpandedHit(
                path=hit.path,
                start_line=start,
                end_line=end,
                text=body,
                score=hit.score,
                original_text=hit.text,
            )
        )
    return expanded
