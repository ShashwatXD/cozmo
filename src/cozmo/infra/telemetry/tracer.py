"""
Structured run traces for observability.

Each unit of work is a span written to JSONL when it ends: turn -> llm / tool /
permission, with subagent turns nested under the tool that started them. A turn's
spans share a turn_id, so one run can be rebuilt as a tree (`cozmo trace last`).
"""

from __future__ import annotations

import contextvars
import json
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MAX_TRACE_BYTES = 10 * 1024 * 1024

# Argument fields worth seeing in a trace. File contents are never written.
_SAFE_ARG_FIELDS = (
    "path",
    "query",
    "command",
    "goal",
    "task",
    "start_line",
    "end_line",
    "around_line",
)
_SECRET = re.compile(
    r"sk-[A-Za-z0-9_-]{10,}|gh[pousr]_[A-Za-z0-9]{10,}|xox[abpr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{12,}"
    r"|(?i:(?:api[_-]?key|token|password|secret)\s*[=:]\s*)\S+"
)


def redact(text: str) -> str:
    return _SECRET.sub("[redacted]", text)


def safe_args(raw: str) -> dict[str, Any]:
    """Keep a short, redacted view of tool arguments."""
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"invalid_json": True}
    if not isinstance(data, dict):
        return {}
    return {k: redact(str(data[k]))[:200] for k in _SAFE_ARG_FIELDS if k in data}


@dataclass
class TraceEvent:
    ts: float
    run_id: str
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    span_id: str = ""
    parent_id: str = ""
    turn_id: str = ""
    duration_ms: float | None = None
    status: str = "ok"


@dataclass
class Span:
    kind: str
    span_id: str
    parent_id: str
    turn_id: str
    started: float
    attrs: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    parent: Span | None = None

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)


_current: contextvars.ContextVar[Span | None] = contextvars.ContextVar("cozmo_span", default=None)


class Tracer:
    def __init__(self, path: Path | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self.run_id = uuid.uuid4().hex[:12]
        self.events: list[TraceEvent] = []
        self._lock = threading.Lock()
        if path and enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def start(self, kind: str, **attrs: Any) -> Span:
        parent = _current.get()
        span_id = uuid.uuid4().hex[:12]
        span = Span(
            kind=kind,
            span_id=span_id,
            parent_id=parent.span_id if parent else "",
            turn_id=parent.turn_id if parent else span_id,
            started=time.perf_counter(),
            attrs=dict(attrs),
            parent=parent,
        )
        _current.set(span)
        return span

    def end(self, span: Span, *, status: str | None = None) -> None:
        _current.set(span.parent)
        if status:
            span.status = status
        if not self.enabled:
            return
        self._write(
            TraceEvent(
                ts=time.time(),
                run_id=self.run_id,
                kind=span.kind,
                data=span.attrs,
                span_id=span.span_id,
                parent_id=span.parent_id,
                turn_id=span.turn_id,
                duration_ms=round((time.perf_counter() - span.started) * 1000, 1),
                status=span.status,
            )
        )

    @contextmanager
    def span(self, kind: str, **attrs: Any) -> Iterator[Span]:
        span = self.start(kind, **attrs)
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.set(error_type=type(exc).__name__, error=redact(str(exc))[:300])
            raise
        finally:
            self.end(span)

    def emit(self, kind: str, **data: Any) -> None:
        """A point-in-time event inside the current span."""
        if not self.enabled:
            return
        parent = _current.get()
        self._write(
            TraceEvent(
                ts=time.time(),
                run_id=self.run_id,
                kind=kind,
                data=data,
                span_id=uuid.uuid4().hex[:12],
                parent_id=parent.span_id if parent else "",
                turn_id=parent.turn_id if parent else "",
            )
        )

    def _write(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)
            if self.path is None:
                return
            self._rotate()
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")

    def _rotate(self) -> None:
        assert self.path is not None
        try:
            if self.path.stat().st_size > MAX_TRACE_BYTES:
                self.path.replace(self.path.with_suffix(".jsonl.1"))
        except OSError:
            pass
