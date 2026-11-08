"""Record real model replies, then replay them offline against changed harness code."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message
from cozmo.domain.ports import LLMClient
from cozmo.domain.tools import ToolCall, ToolSpec


def _prompt_hash(messages: list[Message]) -> str:
    raw = "\x00".join(f"{m.role.value}:{m.content}" for m in messages)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class RecordingLLM:
    def __init__(self, inner: LLMClient, path: Path) -> None:
        self._inner = inner
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        result = self._inner.complete(
            messages, temperature=temperature, json_mode=json_mode, tools=tools
        )
        row = {
            "prompt_hash": _prompt_hash(messages),
            "content": result.content,
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "tool_calls": [[t.id, t.name, t.arguments] for t in result.tool_calls],
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        return result

    def stream(self, messages: list[Message], *, temperature: float = 0.2) -> Iterator[str]:
        return self._inner.stream(messages, temperature=temperature)


class ReplayLLM:
    """Serves recorded replies in order. `drift` counts prompts that no longer match."""

    def __init__(self, path: Path) -> None:
        lines = path.read_text(encoding="utf-8").splitlines()
        self._rows = [json.loads(line) for line in lines if line.strip()]
        self.drift = 0

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        if not self._rows:
            raise RuntimeError(
                "cassette exhausted: the harness made more model calls than recorded"
            )
        row = self._rows.pop(0)
        if row["prompt_hash"] != _prompt_hash(messages):
            self.drift += 1
        return CompletionResult(
            content=row["content"],
            usage=Usage(row["prompt_tokens"], row["completion_tokens"]),
            finish_reason=row["finish_reason"],
            tool_calls=tuple(ToolCall(*t) for t in row["tool_calls"]),
        )

    def stream(self, messages: list[Message], *, temperature: float = 0.2) -> Iterator[str]:
        yield self.complete(messages).content
