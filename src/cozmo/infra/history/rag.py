"""History RAG: incremental embed of session turns for search_history."""

from __future__ import annotations

import json
import logging
from typing import Any

from cozmo.domain.events import EventKind, SessionEvent
from cozmo.domain.ports_history import EventStore
from cozmo.domain.ports_rag import Embedder
from cozmo.domain.rag import Chunk
from cozmo.infra.config.paths import history_index_path, history_rag_meta_path
from cozmo.infra.rag.store import JsonVectorStore
from cozmo.search.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

INDEXABLE = frozenset(
    {
        EventKind.USER_TURN,
        EventKind.ASSISTANT_TURN,
        EventKind.COMPACT,
    }
)

_BATCH = 32


def _session_path(session_id: str) -> str:
    return f"history/{session_id}"


def _event_text(event: SessionEvent) -> str:
    data = event.data or {}
    if event.kind == EventKind.USER_TURN:
        body = str(data.get("text") or "").strip()
        label = "user"
    elif event.kind == EventKind.ASSISTANT_TURN:
        body = str(data.get("text") or "").strip()
        label = "assistant"
    elif event.kind == EventKind.COMPACT:
        body = str(data.get("summary") or "").strip()
        label = "compact"
    else:
        return ""
    if not body:
        return ""
    return f"[{label}] session={event.session_id}\n{body}"


class HistoryRagIndex:
    """
    Derived vector index over JSONL history.

    Source of truth remains event JSONL. This store is updated incrementally
    after indexable events (user/assistant/compact).
    """

    def __init__(
        self,
        workdir: Any,
        embedder: Embedder,
        *,
        enabled: bool = True,
    ) -> None:
        from pathlib import Path

        self._workdir = Path(workdir).resolve()
        self._embedder = embedder
        self._enabled = enabled
        self._index_path = history_index_path(self._workdir)
        self._meta_path = history_rag_meta_path(self._workdir)
        self._store = (
            JsonVectorStore.load(self._index_path)
            if enabled and self._index_path.is_file()
            else JsonVectorStore()
        )
        self._meta = self._load_meta() if enabled else {"sessions": {}}

    def __len__(self) -> int:
        return len(self._store)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sync_session(self, session_id: str, events: list[SessionEvent]) -> int:
        """Index new indexable events for one session. Returns chunks added."""
        if not self._enabled or not session_id:
            return 0
        sessions = self._meta.setdefault("sessions", {})
        prev = sessions.get(session_id) or {}
        prev_seen = int(prev.get("events_seen") or 0)
        prev_head = float(prev.get("head_ts") or 0.0)
        head_ts = float(events[0].ts) if events else 0.0

        # Trim/rewrite invalidated the cursor — rebuild this session's chunks.
        if events and (len(events) < prev_seen or (prev_seen and prev_head != head_ts)):
            self._store.drop_paths({_session_path(session_id)})
            prev_seen = 0

        if not events:
            sessions[session_id] = {"events_seen": 0, "head_ts": 0.0}
            self._persist()
            return 0

        new_events = events[prev_seen:]
        added = 0
        batch_texts: list[str] = []
        batch_meta: list[tuple[str, int, str]] = []  # path, start_line(=idx), text

        for offset, event in enumerate(new_events, start=prev_seen):
            if event.kind not in INDEXABLE:
                continue
            text = _event_text(event)
            if not text:
                continue
            path = _session_path(session_id)
            batch_texts.append(text)
            batch_meta.append((path, offset + 1, text))
            if len(batch_texts) >= _BATCH:
                added += self._flush_batch(batch_texts, batch_meta)
                batch_texts, batch_meta = [], []

        if batch_texts:
            added += self._flush_batch(batch_texts, batch_meta)

        sessions[session_id] = {
            "events_seen": len(events),
            "head_ts": head_ts,
        }
        self._persist()
        return added

    def sync_from_store(self, store: EventStore, session_id: str) -> int:
        return self.sync_session(session_id, store.list_events(session_id))

    def catch_up(self, store: EventStore, *, limit: int = 50) -> int:
        """Index any sessions not fully reflected in the history vector store."""
        if not self._enabled:
            return 0
        total = 0
        active: set[str] = set()
        try:
            rows = store.list_sessions(limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("history rag catch_up list failed: %s", exc)
            return 0
        for row in rows:
            sid = str(row.get("id") or "")
            if not sid:
                continue
            active.add(sid)
            try:
                total += self.sync_from_store(store, sid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("history rag sync %s failed: %s", sid, exc)
        self.prune_missing(active)
        return total

    def prune_missing(self, active_session_ids: set[str]) -> int:
        """Drop chunks for sessions no longer on disk."""
        if not self._enabled:
            return 0
        sessions = self._meta.setdefault("sessions", {})
        stale = [sid for sid in list(sessions) if sid not in active_session_ids]
        removed = 0
        for sid in stale:
            removed += self._store.drop_paths({_session_path(sid)})
            sessions.pop(sid, None)
        if stale:
            self._persist()
        return removed

    def search(self, query: str, *, top_k: int = 5) -> str:
        if not self._enabled:
            return "History RAG disabled."
        if len(self._store) == 0:
            return (
                "No history index yet. Prior user/assistant turns are indexed "
                "as you chat; try again after a completed turn."
            )
        q = (query or "").strip()
        if not q:
            return "search_history requires a non-empty query."
        top_k = max(1, min(int(top_k), 20))
        candidate_k = min(50, max(top_k, len(self._store)))
        pipeline = RetrievalPipeline(
            self._store,
            self._embedder,
            sources={},
            candidate_k=candidate_k,
            top_k=top_k,
            expand_before=0,
            expand_after=0,
        )
        hits = pipeline.retrieve(q, top_k=top_k)
        if not hits:
            return "No history hits."
        lines: list[str] = []
        for h in hits:
            preview = h.text[:1200]
            lines.append(f"score={h.score:.3f} {h.path}\n{preview}")
        return "\n---\n".join(lines)

    def _flush_batch(
        self,
        texts: list[str],
        meta: list[tuple[str, int, str]],
    ) -> int:
        embeddings = self._embedder.embed_many(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"embed_many returned {len(embeddings)} for {len(texts)} history chunks"
            )
        for (path, start_line, text), emb in zip(meta, embeddings, strict=True):
            if not emb:
                raise RuntimeError("empty history embedding")
            chunk = Chunk(
                id=f"{path}::{start_line}",
                path=path,
                start_line=start_line,
                text=text,
            )
            self._store.add(chunk, emb)
        return len(texts)

    def _persist(self) -> None:
        try:
            self._store.save(self._index_path)
            self._meta_path.parent.mkdir(parents=True, exist_ok=True)
            self._meta_path.write_text(
                json.dumps(self._meta, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("history rag persist failed: %s", exc)

    def _load_meta(self) -> dict[str, Any]:
        if not self._meta_path.is_file():
            return {"version": 1, "sessions": {}}
        try:
            raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "sessions": {}}
        if not isinstance(raw, dict):
            return {"version": 1, "sessions": {}}
        sessions = raw.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        return {"version": 1, "sessions": sessions}
