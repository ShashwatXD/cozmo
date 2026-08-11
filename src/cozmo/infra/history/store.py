"""JSONL session event store under .cozmo/history/ (mirrors config roots)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cozmo.domain.events import SessionEvent
from cozmo.infra.config.paths import (
    session_events_path,
    sessions_index_path,
    workspace_history_dir,
)


class JsonlEventStore:
    def __init__(
        self,
        workdir: Path,
        *,
        enabled: bool = True,
        max_sessions: int = 50,
        max_events_per_session: int = 2000,
    ) -> None:
        self._workdir = workdir.resolve()
        self._enabled = enabled
        self._max_sessions = max_sessions
        self._max_events = max_events_per_session
        if enabled:
            workspace_history_dir(self._workdir).mkdir(parents=True, exist_ok=True)

    def append(self, event: SessionEvent) -> None:
        if not self._enabled:
            return
        path = session_events_path(self._workdir, event.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), default=str) + "\n")
        if event.kind.value == "session_start":
            self._index_session(event)
        self._trim_session_file(path)
        self._prune_old_sessions()

    def list_events(self, session_id: str) -> list[SessionEvent]:
        path = session_events_path(self._workdir, session_id)
        if not path.is_file():
            return []
        out: list[SessionEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(SessionEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return out

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        path = sessions_index_path(self._workdir)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        rows.reverse()
        return rows[:limit]

    def _index_session(self, event: SessionEvent) -> None:
        path = sessions_index_path(self._workdir)
        row = {
            "id": event.session_id,
            "started_at": event.ts,
            "workdir": str(self._workdir),
            **{k: v for k, v in event.data.items() if k in {"provider", "model"}},
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _trim_session_file(self, path: Path) -> None:
        if self._max_events <= 0 or not path.is_file():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_events:
            return
        path.write_text("\n".join(lines[-self._max_events :]) + "\n", encoding="utf-8")

    def _prune_old_sessions(self) -> None:
        if self._max_sessions <= 0:
            return
        hist = workspace_history_dir(self._workdir)
        files = sorted(
            hist.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
        )
        # Keep sessions.jsonl + newest session files.
        session_files = [p for p in files if p.name != "sessions.jsonl"]
        overflow = len(session_files) - self._max_sessions
        if overflow <= 0:
            return
        for path in session_files[:overflow]:
            try:
                path.unlink()
            except OSError:
                pass
