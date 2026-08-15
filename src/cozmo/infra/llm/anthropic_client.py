"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall, ToolSpec
from cozmo.infra.llm.curl_log import log_llm_curl

_API = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"

def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters
            if t.parameters
            else {"type": "object", "properties": {}},
        }
        for t in tools
    ]

def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    system_parts: list[str] = []
    rest: list[Message] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
        else:
            rest.append(m)
    system = "\n\n".join(system_parts) if system_parts else None
    return system, rest

def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            continue
        if m.role == Role.ASSISTANT and m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": args if isinstance(args, dict) else {},
                    }
                )
            out.append({"role": "assistant", "content": content})
        elif m.role == Role.TOOL:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == Role.USER:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.ASSISTANT:
            out.append({"role": "assistant", "content": m.content or ""})
    return out

class AnthropicClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_s: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_s
        self._max_tokens = max_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _VERSION,
            "content-type": "application/json",
        }

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": temperature,
            "messages": _to_anthropic_messages(rest),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = _to_anthropic_tools(tools)
        if json_mode and system:
            body["system"] = system + "\n\nRespond with valid JSON only."

        headers = self._headers()
        log_llm_curl(url=_API, headers=headers, body=body)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(_API, headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic error {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        usage_raw = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_raw.get("input_tokens") or 0),
            completion_tokens=int(usage_raw.get("output_tokens") or 0),
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=json.dumps(block.get("input") or {}),
                    )
                )
        return CompletionResult(
            content="".join(text_parts),
            usage=usage,
            finish_reason=str(data.get("stop_reason") or ""),
            tool_calls=tuple(tool_calls),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        result = self.complete(messages, temperature=temperature)
        if result.content:
            yield result.content
