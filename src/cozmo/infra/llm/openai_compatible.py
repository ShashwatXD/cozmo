"""OpenAI-compatible chat adapter."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from cozmo.domain.completion import CompletionResult, Usage
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall, ToolSpec
from cozmo.infra.llm.curl_log import log_llm_curl

_DEFAULT_BASE = "https://api.openai.com/v1"

def _to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.ASSISTANT and m.tool_calls:
            out.append(
                {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments,
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
            )
        elif m.role == Role.TOOL:
            item: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": m.tool_call_id or "",
                "content": m.content,
            }
            if m.name:
                item["name"] = m.name
            out.append(item)
        else:
            out.append({"role": m.role.value, "content": m.content})
    return out

class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or _DEFAULT_BASE).rstrip("/")
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _chat_completions_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        tools: list[ToolSpec] | None = None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_api_messages(messages),
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = [t.to_openai() for t in tools]
            kwargs["tool_choice"] = "auto"

        log_llm_curl(
            url=self._chat_completions_url(),
            headers=self._request_headers(),
            body=kwargs,
        )
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = Usage()
        if resp.usage is not None:
            usage = Usage(
                prompt_tokens=resp.usage.prompt_tokens or 0,
                completion_tokens=resp.usage.completion_tokens or 0,
            )

        tool_calls: list[ToolCall] = []
        raw_calls = choice.message.tool_calls or []
        for tc in raw_calls:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments or "{}",
                )
            )

        return CompletionResult(
            content=choice.message.content or "",
            usage=usage,
            finish_reason=choice.finish_reason,
            tool_calls=tuple(tool_calls),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": _to_api_messages(messages),
            "temperature": temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        try:
            stream = self._client.chat.completions.create(
                **kwargs,
                stream_options={"include_usage": True},
            )
            body: dict[str, Any] = {
                **kwargs,
                "stream_options": {"include_usage": True},
            }
        except TypeError:
            stream = self._client.chat.completions.create(**kwargs)
            body = kwargs

        log_llm_curl(
            url=self._chat_completions_url(),
            headers=self._request_headers(),
            body=body,
        )

        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta and delta.content:
                yield delta.content
