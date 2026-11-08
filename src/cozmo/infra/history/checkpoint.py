"""Per-step checkpoints so a crash mid-turn can resume with its tool evidence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from cozmo.domain.memory import drop_incomplete_tool_group
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall
from cozmo.infra.config.paths import workspace_dir


class CheckpointStore:
    """One JSON file per session, replaced atomically, deleted when a turn ends."""

    def __init__(self, workdir: Path, *, enabled: bool = True) -> None:
        self._dir = workspace_dir(workdir) / "checkpoints"
        self._enabled = enabled

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def save(self, session_id: str, messages: list[Message]) -> None:
        if not self._enabled:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        rows = [{**asdict(m), "role": m.role.value} for m in messages if m.role != Role.SYSTEM]
        tmp = self._path(session_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(self._path(session_id))

    def load(self, session_id: str) -> list[Message] | None:
        """Messages from the unfinished turn, or None if the last turn ended cleanly."""
        try:
            rows = json.loads(self._path(session_id).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        messages = [
            Message(
                role=Role(r["role"]),
                content=r.get("content") or "",
                tool_calls=tuple(ToolCall(**tc) for tc in r.get("tool_calls") or ()),
                tool_call_id=r.get("tool_call_id"),
                name=r.get("name"),
            )
            for r in rows
        ]
        return drop_incomplete_tool_group(messages)

    def clear(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)
