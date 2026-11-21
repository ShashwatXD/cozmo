"""Read `.cozmo/traces.jsonl` and `.cozmo/evals/` back into turns and runs for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_events(path: Path) -> list[dict[str, Any]]:
    """Every event in the trace file. A damaged line is skipped, not fatal."""
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and "kind" in item:
                events.append(item)
    return events


def _turn_key(ev: dict[str, Any]) -> str:
    # Traces written before turns existed only have a run id; each run becomes one entry.
    return ev.get("turn_id") or f"run-{ev.get('run_id', '?')}"


def group_turns(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    turns: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        turns.setdefault(_turn_key(ev), []).append(ev)
    return turns


def summarize_turn(turn_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((e for e in events if e["kind"] == "turn_start"), None)
    done = next((e for e in reversed(events) if e["kind"] == "agent_done"), None)
    llm = [e for e in events if e["kind"] == "llm"]
    tools = [e for e in events if e["kind"] == "tool"]
    data = (done or {}).get("data", {})
    first_ts = min(e["ts"] for e in events)
    session = events[0].get("session_id") or ""
    grounded = data.get("groundedness") or {}
    return {
        "id": turn_id,
        "session_id": session,
        "is_eval": str(session).startswith("eval-"),
        "started_at": first_ts,
        "question": (start or {}).get("data", {}).get("text", "(no question recorded)"),
        "mode": (start or {}).get("data", {}).get("mode", ""),
        "model": (start or {}).get("data", {}).get("model")
        or next((e["data"].get("model") for e in llm if e["data"].get("model")), ""),
        "llm_calls": len(llm),
        "tool_calls": len(tools),
        "tool_errors": sum(1 for e in tools if e["data"].get("is_error")),
        "tokens": data.get("tokens", sum(e["data"].get("tokens", 0) for e in llm)),
        "cost_usd": data.get("cost_usd", sum(e["data"].get("cost_usd", 0) or 0 for e in llm)),
        "duration_s": data.get("duration_s", round(max(e["ts"] for e in events) - first_ts, 2)),
        "stop_reason": data.get("stop_reason", "unfinished" if not done else ""),
        "steps": data.get("steps", len(llm)),
        "groundedness": grounded.get("score"),
        "cut_off": any(e["data"].get("finish_reason") == "length" for e in llm),
    }


def list_turns(path: Path, *, limit: int = 300) -> list[dict[str, Any]]:
    turns = group_turns(load_events(path))
    summaries = [summarize_turn(tid, evs) for tid, evs in turns.items()]
    summaries.sort(key=lambda t: t["started_at"], reverse=True)
    return summaries[:limit]


def turn_detail(path: Path, turn_id: str) -> dict[str, Any] | None:
    events = group_turns(load_events(path)).get(turn_id)
    if not events:
        return None
    return {"summary": summarize_turn(turn_id, events), "events": events}


def list_eval_runs(evals_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for f in sorted(evals_dir.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(
            {
                k: data.get(k)
                for k in ("id", "created_at", "live", "provider", "model", "passed", "total")
            }
        )
    return runs


def eval_run(evals_dir: Path, run_id: str) -> dict[str, Any] | None:
    # The id is a file stem we wrote ourselves; anything else is refused.
    if not run_id or any(c in run_id for c in "/\\.") or len(run_id) > 64:
        return None
    f = evals_dir / f"{run_id}.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
