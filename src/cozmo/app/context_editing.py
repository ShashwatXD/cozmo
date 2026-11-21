"""Clear old tool results so the model fetches again what it still needs.

A file read three turns ago is resent on every request until compaction, even though the
model rarely needs it again, and the file may have changed since. Before each new turn the
oldest tool results are replaced by one line saying what ran. The model can call the tool
again for anything it still needs, and gets the current version when it does.

This costs no model call, so it runs before compaction. Compaction (an LLM summary) runs
only if the context is still over budget afterwards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall

CLEARED_PREFIX = "[cleared to save context]"
# Short results (a status line, "Wrote 120 chars") cost little and often still matter.
MIN_CHARS_TO_CLEAR = 400


@dataclass(frozen=True)
class ClearReport:
    cleared: int
    chars_freed: int


def _describe(call: ToolCall | None, name: str | None) -> str:
    if call is None:
        return name or "tool"
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if isinstance(args, dict):
        # Big payloads (a file being written) are not worth repeating in the note.
        args = {k: v for k, v in args.items() if not (isinstance(v, str) and len(v) > 120)}
        shown = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
    else:
        shown = ""
    return f"{call.name}({shown})"


def clear_old_tool_results(
    history: list[Message],
    *,
    keep_recent: int,
    min_chars: int = MIN_CHARS_TO_CLEAR,
) -> tuple[list[Message], ClearReport]:
    """Return *history* with all but the newest *keep_recent* tool results replaced by a note."""
    tool_positions = [i for i, m in enumerate(history) if m.role == Role.TOOL]
    if keep_recent < 0 or len(tool_positions) <= keep_recent:
        return history, ClearReport(0, 0)
    old = set(tool_positions[: len(tool_positions) - keep_recent])

    calls = {tc.id: tc for m in history if m.tool_calls for tc in m.tool_calls}
    out: list[Message] = []
    cleared = freed = 0
    for i, m in enumerate(history):
        content = m.content or ""
        if i in old and len(content) >= min_chars and not content.startswith(CLEARED_PREFIX):
            what = _describe(calls.get(m.tool_call_id or ""), m.name)
            note = (
                f"{CLEARED_PREFIX} {what} returned {len(content):,} chars here earlier. "
                "Call it again if you still need the result."
            )
            out.append(replace(m, content=note))
            cleared += 1
            freed += len(content) - len(note)
        else:
            out.append(m)
    return out, ClearReport(cleared, freed)
