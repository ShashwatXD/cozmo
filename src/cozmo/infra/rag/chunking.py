"""
Text chunking for RAG.

What: split file text into overlapping chunks with line numbers.
Why: embeddings work on passages, not whole repos.
Layer: infra/rag (pure-ish helper).
"""

from __future__ import annotations

from cozmo.domain.rag import Chunk


def chunk_text(
    path: str,
    text: str,
    *,
    chunk_size: int = 600,
    overlap: int = 100,
) -> list[Chunk]:
    """
    Chunk by characters on line boundaries when possible.

    """
    if not text.strip():
        return []

    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    start_line = 1
    line_no = 1
    idx = 0

    def flush() -> None:
        nonlocal idx, buf, buf_len, start_line
        body = "".join(buf).strip()
        if not body:
            buf, buf_len = [], 0
            return
        chunks.append(
            Chunk(
                id=f"{path}::{idx}",
                path=path,
                start_line=start_line,
                text=body,
            )
        )
        idx += 1
        # overlap: keep tail of buf
        if overlap <= 0 or not buf:
            buf, buf_len = [], 0
            start_line = line_no
            return
        # rebuild overlap from end of joined text
        joined = "".join(buf)
        tail = joined[-overlap:]
        # approximate start_line for overlap region
        kept_lines = tail.count("\n")
        start_line = max(1, line_no - kept_lines)
        buf = [tail]
        buf_len = len(tail)

    for line in lines:
        if buf_len + len(line) > chunk_size and buf:
            flush()
        if not buf:
            start_line = line_no
        buf.append(line)
        buf_len += len(line)
        line_no += 1

    flush()
    return chunks
