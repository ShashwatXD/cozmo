"""
Structured run traces for observability.

Every event is one JSON line in `.cozmo/traces.jsonl`, tagged with the session and the
turn it belongs to, so `cozmo trace` can show a turn step by step:

  turn_start   the user's message, mode and model
  llm          one model request: the messages sent, the reply, tool calls, tokens, cost, time
  tool         one tool call: its arguments, result, error flag and time
  context_cleared / compact / stuck   what the harness did to the context
  agent_done   how the turn ended: stop reason, totals, groundedness of the answer
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Long strings are cut so one turn cannot grow the trace file without bound.
MAX_TRACE_CHARS = 8_000


def clip(text: str | None, limit: int = MAX_TRACE_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n…[{len(text) - limit:,} more chars not kept in the trace]"


def snapshot_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """The request as the model received it, as plain JSON with long text clipped."""
    out: list[dict[str, Any]] = []
    for m in messages:
        item: dict[str, Any] = {
            "role": getattr(m.role, "value", str(m.role)),
            "content": clip(m.content),
        }
        if getattr(m, "tool_calls", None):
            item["tool_calls"] = [
                {"name": tc.name, "arguments": clip(tc.arguments, 2_000)} for tc in m.tool_calls
            ]
        if getattr(m, "tool_call_id", None):
            item["tool_call_id"] = m.tool_call_id
        if getattr(m, "name", None):
            item["name"] = m.name
        out.append(item)
    return out


@dataclass
class TraceEvent:
    ts: float
    run_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    turn_id: str | None = None


class Tracer:
    def __init__(
        self,
        path: Path | None = None,
        *,
        enabled: bool = True,
        session_id: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.path = path
        self.run_id = uuid.uuid4().hex[:12]
        self.session_id = session_id or self.run_id
        self.turn_id: str | None = None
        self.events: list[TraceEvent] = []
        if path and enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def start_turn(self, text: str, **data: Any) -> str:
        """Begin a new turn; every later event carries its id until the next turn starts."""
        self.turn_id = uuid.uuid4().hex[:12]
        self.emit("turn_start", text=clip(text, 4_000), **data)
        return self.turn_id

    def emit(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        ev = TraceEvent(
            ts=time.time(),
            run_id=self.run_id,
            kind=kind,
            data=data,
            session_id=self.session_id,
            turn_id=self.turn_id,
        )
        self.events.append(ev)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), default=str) + "\n")
