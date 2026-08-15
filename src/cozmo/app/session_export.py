"""Export session audit trails to markdown or JSON."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from cozmo.domain.events import EventKind, SessionEvent


def export_session_json(
    session_id: str,
    events: list[SessionEvent],
    *,
    index: dict[str, Any] | None = None,
) -> str:
    payload = {
        "session_id": session_id,
        "index": index or {},
        "events": [e.to_dict() for e in events],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def export_session_markdown(
    session_id: str,
    events: list[SessionEvent],
    *,
    index: dict[str, Any] | None = None,
) -> str:
    meta = index or {}
    started = meta.get("started_at")
    if started is None and events:
        started = events[0].ts
    started_s = _fmt_ts(started) if started is not None else "?"
    model = meta.get("model") or _from_start(events, "model") or "?"
    provider = meta.get("provider") or _from_start(events, "provider") or "?"
    workdir = meta.get("workdir") or _from_start(events, "workdir") or "?"

    lines = [
        f"# Cozmo session `{session_id}`",
        "",
        f"- Started: {started_s}",
        f"- Model: {provider} / {model}",
        f"- Workdir: {workdir}",
        "",
        "## Turns",
        "",
    ]
    for event in events:
        if event.kind == EventKind.SESSION_START:
            continue
        if event.kind == EventKind.SESSION_END:
            lines.append("### Session end")
            lines.append("")
            continue
        if event.kind == EventKind.USER_TURN:
            lines.append("### User")
            lines.append(str(event.data.get("text") or ""))
            lines.append("")
        elif event.kind == EventKind.ASSISTANT_TURN:
            lines.append("### Assistant")
            lines.append(str(event.data.get("text") or ""))
            lines.append("")
        elif event.kind == EventKind.TOOL:
            name = event.data.get("name") or "?"
            err = event.data.get("is_error")
            lines.append(f"### Tool `{name}` (error={err})")
            preview = str(event.data.get("preview") or "").strip()
            if preview:
                for pl in preview.splitlines():
                    lines.append(f"> {pl}")
            lines.append("")
        elif event.kind == EventKind.COMPACT:
            lines.append("### Compact")
            lines.append(str(event.data.get("summary") or ""))
            lines.append("")
        elif event.kind == EventKind.STOPPED:
            reason = event.data.get("reason") or "?"
            steps = event.data.get("steps")
            extra = f" steps={steps}" if steps is not None else ""
            lines.append(f"### Stopped")
            lines.append(f"reason={reason}{extra}")
            lines.append("")
        elif event.kind == EventKind.SUBAGENT:
            lines.append("### Subagent")
            lines.append(str(event.data.get("goal") or ""))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _from_start(events: list[SessionEvent], key: str) -> Any:
    for event in events:
        if event.kind == EventKind.SESSION_START:
            return event.data.get(key)
    return None


def _fmt_ts(ts: float | int | str) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (TypeError, ValueError, OSError):
        return str(ts)
