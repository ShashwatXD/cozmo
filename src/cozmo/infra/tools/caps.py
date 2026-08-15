"""Tool output size limits for the context economy."""

from __future__ import annotations

# Default cap on tool result text re-injected into the LLM message list.
DEFAULT_MAX_TOOL_CHARS = 24_000

# Per-tool softer defaults (executor still enforces DEFAULT_MAX_TOOL_CHARS).
TOOL_SOFT_CAPS: dict[str, int] = {
    "read_file": 32_000,
    "search_repo": 16_000,
    "semantic_search": 20_000,
    "run_shell": 20_000,
    "git_diff": 24_000,
    "git_status": 8_000,
    "run_subtask": 8_000,
}


def shape_tool_content(name: str, content: str, *, max_chars: int | None = None) -> str:
    """Truncate tool output with a clear marker. Never return unbounded blobs."""
    if not content:
        return content
    soft = TOOL_SOFT_CAPS.get(name, DEFAULT_MAX_TOOL_CHARS)
    hard = DEFAULT_MAX_TOOL_CHARS if max_chars is None else max_chars
    limit = min(soft, hard)
    if max_chars is None:
        limit = max(256, limit)
    else:
        limit = max(1, limit)
    if len(content) <= limit:
        return content
    omitted = len(content) - limit
    return (
        content[:limit]
        + f"\n...[truncated {omitted} chars; use a narrower query or ranged read_file]"
    )
