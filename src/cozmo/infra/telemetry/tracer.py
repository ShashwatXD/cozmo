"""
Structured run traces for observability.

Emits JSONL events: llm, tool, agent_done - tokens, latency, cost.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    ts: float
    run_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, path: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self.run_id = uuid.uuid4().hex[:12]
        self.events: list[TraceEvent] = []
        if path and enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, **data: Any) -> None:
        if not self.enabled:
            return
        ev = TraceEvent(ts=time.time(), run_id=self.run_id, kind=kind, data=data)
        self.events.append(ev)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), default=str) + "\n")
