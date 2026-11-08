"""Turn raw trace spans into answers: what happened, what it cost, what is failing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

Event = dict[str, Any]


def load_trace_events(path: Path) -> list[Event]:
    """Read the rotated file then the current one, skipping damaged lines."""
    events: list[Event] = []
    for file in (path.with_suffix(".jsonl.1"), path):
        if not file.is_file():
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                events.append(row)
    return events


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q / 100 * (len(ordered) - 1)))]


def _root_turns(events: list[Event]) -> list[Event]:
    return [e for e in events if e.get("kind") == "turn" and not e.get("parent_id")]


def summarize(events: list[Event], *, last: int | None = None) -> dict[str, Any]:
    """Aggregate metrics over the most recent *last* top-level turns."""
    turns = _root_turns(events)
    if last:
        turns = turns[-last:]
    keep = {t.get("turn_id") for t in turns}
    scoped = [e for e in events if e.get("turn_id") in keep]

    stops = Counter(str(t.get("data", {}).get("stop_reason", "unknown")) for t in turns)
    llm = [e for e in scoped if e.get("kind") == "llm"]
    tools = [e for e in scoped if e.get("kind") == "tool"]
    perms = [e for e in scoped if e.get("kind") == "permission"]

    by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    )
    for e in llm:
        d = e.get("data", {})
        row = by_model[str(d.get("model", "?"))]
        row["calls"] += 1
        row["input_tokens"] += d.get("input_tokens") or 0
        row["output_tokens"] += d.get("output_tokens") or 0
        row["cost_usd"] += d.get("cost_usd") or 0.0

    by_tool: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "errors": 0})
    for e in tools:
        d = e.get("data", {})
        row = by_tool[str(d.get("name", "?"))]
        row["calls"] += 1
        row["errors"] += 1 if d.get("is_error") else 0

    def durations(rows: list[Event]) -> list[float]:
        return [float(e["duration_ms"]) for e in rows if e.get("duration_ms") is not None]

    def count(kind: str) -> int:
        return sum(1 for e in scoped if e.get("kind") == kind)

    turn_ms, llm_ms = durations(turns), durations(llm)
    return {
        "turns": len(turns),
        "completed_rate": round(stops["completed"] / len(turns), 3) if turns else 0.0,
        "stop_reasons": dict(stops),
        "turn_ms_p50": _pct(turn_ms, 50),
        "turn_ms_p95": _pct(turn_ms, 95),
        "llm_ms_p50": _pct(llm_ms, 50),
        "llm_ms_p95": _pct(llm_ms, 95),
        "llm_errors": sum(1 for e in llm if e.get("status") == "error"),
        "retries": sum(e.get("data", {}).get("retries") or 0 for e in llm),
        "cost_usd": round(sum(r["cost_usd"] for r in by_model.values()), 6),
        "by_model": {k: dict(v) for k, v in by_model.items()},
        "by_tool": {k: dict(v) for k, v in by_tool.items()},
        "approvals_allowed": sum(1 for e in perms if e.get("data", {}).get("allowed")),
        "approvals_denied": sum(1 for e in perms if not e.get("data", {}).get("allowed")),
        "stuck": count("stuck"),
        "context_shrinks": count("context_shrink"),
        "compactions": count("compaction"),
        "truncated_outputs": count("truncated"),
    }


def list_turns(events: list[Event], *, limit: int = 10) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in _root_turns(events)[-limit:]:
        d = t.get("data", {})
        out.append(
            {
                "turn_id": t.get("turn_id"),
                "stop": d.get("stop_reason", "?"),
                "steps": d.get("steps", 0),
                "tools": d.get("tool_calls", 0),
                "tokens": (d.get("input_tokens") or 0) + (d.get("output_tokens") or 0),
                "cost_usd": d.get("cost_usd", 0.0),
                "ms": t.get("duration_ms", 0),
            }
        )
    return out


def _describe(e: Event) -> str:
    d = e.get("data", {})
    kind = e.get("kind", "?")
    ms = e.get("duration_ms")
    took = f" {ms:.0f}ms" if isinstance(ms, (int, float)) else ""
    if kind == "turn":
        return (
            f"turn{took} stop={d.get('stop_reason', '?')} steps={d.get('steps', 0)} "
            f"tools={d.get('tool_calls', 0)} depth={d.get('depth', 0)}"
        )
    if kind == "llm":
        retry = f" retries={d['retries']}" if d.get("retries") else ""
        fail = f" ERROR {d.get('error_type')}" if e.get("status") == "error" else ""
        return (
            f"llm {d.get('role', '')}/{d.get('model', '?')}{took} "
            f"in={d.get('input_tokens', 0)} out={d.get('output_tokens', 0)} "
            f"${d.get('cost_usd', 0):.4f} finish={d.get('finish_reason')}{retry}{fail}"
        )
    if kind == "tool":
        args = " ".join(f"{k}={v}" for k, v in (d.get("args") or {}).items())
        state = "error" if d.get("is_error") else "ok"
        cut = " truncated" if d.get("truncated") else ""
        return (
            f"tool {d.get('name', '?')}{took} {state} {d.get('result_chars', 0)} chars{cut} {args}"
        )
    if kind == "permission":
        return (
            f"permission {d.get('tool')}{took} allowed={d.get('allowed')} ({d.get('reason', '')})"
        )
    extra = " ".join(f"{k}={v}" for k, v in d.items())
    return f"{kind}{took} {extra}".rstrip()


def render_turn(events: list[Event], turn_id: str) -> list[str]:
    """One turn as an indented tree, in start order."""
    if turn_id == "last":
        roots = _root_turns(events)
        if not roots:
            return []
        turn_id = str(roots[-1].get("turn_id"))
    scoped = [e for e in events if str(e.get("turn_id", "")).startswith(turn_id)]
    children: dict[str, list[Event]] = defaultdict(list)
    for e in scoped:
        children[str(e.get("parent_id") or "")].append(e)

    def start(e: Event) -> float:
        return float(e.get("ts", 0)) - float(e.get("duration_ms") or 0) / 1000

    lines: list[str] = []

    def walk(parent: str, depth: int) -> None:
        for e in sorted(children.get(parent, []), key=start):
            lines.append("  " * depth + _describe(e))
            walk(str(e.get("span_id")), depth + 1)

    walk("", 0)
    return lines
