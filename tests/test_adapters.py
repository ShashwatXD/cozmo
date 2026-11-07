"""Provider adapters, model catalog, curl logging and session export (no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from cozmo.app.session_export import export_session_json, export_session_markdown
from cozmo.domain.events import EventKind, SessionEvent
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall, ToolSpec
from cozmo.infra.llm import curl_log, model_catalog
from cozmo.infra.llm.anthropic_client import _to_anthropic_messages
from cozmo.infra.llm.openai_compatible import OpenAICompatibleClient, _to_api_messages

CALL = ToolCall(id="t1", name="read_file", arguments='{"path": "a.py"}')
CONVO = [
    Message(role=Role.SYSTEM, content="sys"),
    Message(role=Role.USER, content="read a.py"),
    Message(role=Role.ASSISTANT, content="", tool_calls=(CALL,)),
    Message(role=Role.TOOL, content="print(1)", tool_call_id="t1", name="read_file"),
]


# --- OpenAI-compatible -------------------------------------------------------


def test_openai_message_mapping_round_trips_tool_calls() -> None:
    api = _to_api_messages(CONVO)
    assert api[0] == {"role": "system", "content": "sys"}
    assert api[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert api[2]["content"] is None
    assert api[3] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "print(1)",
        "name": "read_file",
    }


def _fake_openai(monkeypatch: pytest.MonkeyPatch, client: OpenAICompatibleClient, response):
    sent: dict = {}

    def create(**kwargs):
        sent.update(kwargs)
        return response(kwargs)

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    monkeypatch.setattr("cozmo.infra.llm.openai_compatible.log_llm_curl", lambda **_: None)
    return sent


def test_openai_complete_parses_tool_calls_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleClient(model="gpt-4o-mini", api_key="k")
    tc = SimpleNamespace(id="x", function=SimpleNamespace(name="search_repo", arguments=None))
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[tc]), finish_reason="tool_calls"
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3),
    )
    sent = _fake_openai(monkeypatch, client, lambda _kw: resp)
    tool = ToolSpec(name="search_repo", description="d", parameters={"type": "object"})
    out = client.complete(CONVO, json_mode=True, tools=[tool])
    assert out.tool_calls == (ToolCall(id="x", name="search_repo", arguments="{}"),)
    assert out.usage.total_tokens == 15
    assert out.content == ""
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["tool_choice"] == "auto"


def test_openai_stream_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenAICompatibleClient(model="m", api_key="k")

    def chunks(_kw):
        for piece in ["he", None, "llo"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])
        yield SimpleNamespace(choices=[])  # final usage-only chunk

    sent = _fake_openai(monkeypatch, client, chunks)
    assert "".join(client.stream(CONVO)) == "hello"
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


# --- Anthropic message shaping -----------------------------------------------


def test_anthropic_groups_tool_results_into_one_user_turn() -> None:
    call2 = ToolCall(id="t2", name="read_file", arguments="not json")
    msgs = [
        Message(role=Role.USER, content="hi"),
        Message(role=Role.ASSISTANT, content="looking", tool_calls=(CALL, call2)),
        Message(role=Role.TOOL, content="a", tool_call_id="t1"),
        Message(role=Role.TOOL, content="b", tool_call_id="t2"),
    ]
    out = _to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"][0] == {"type": "text", "text": "looking"}
    assert out[1]["content"][2]["input"] == {}  # malformed arguments degrade safely
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]


# --- model catalog -----------------------------------------------------------


def _mock_get(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(model_catalog.httpx, "Client", factory)


def test_openai_catalog_filters_non_chat_models(monkeypatch: pytest.MonkeyPatch) -> None:
    data = {"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"}]}
    _mock_get(monkeypatch, lambda r: httpx.Response(200, json=data))
    res = model_catalog.validate_and_list("openai", "k")
    assert res.ok
    assert "gpt-4o" in res.models
    assert "text-embedding-3-small" not in res.models and "whisper-1" not in res.models


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openrouter"])
def test_catalog_reports_bad_key(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    _mock_get(monkeypatch, lambda r: httpx.Response(401))
    res = model_catalog.validate_and_list(provider, "bad")
    assert not res.ok and "unauthorized" in res.error


def test_anthropic_catalog_falls_back_to_favorites_on_404(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_get(monkeypatch, lambda r: httpx.Response(404))
    res = model_catalog.validate_and_list("anthropic", "k")
    assert res.ok and res.models == model_catalog.ANTHROPIC_FAVORITES


def test_ollama_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_get(
        monkeypatch,
        lambda r: httpx.Response(
            200, json={"models": [{"name": "qwen2.5:3b"}, {"model": "llama3"}]}
        ),
    )
    res = model_catalog.validate_and_list("ollama", "", base_url="http://localhost:11434/v1")
    assert res.ok and res.models == ("llama3", "qwen2.5:3b")


def test_catalog_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_r):
        raise httpx.ConnectError("refused")

    _mock_get(monkeypatch, boom)
    assert "Cannot reach Ollama" in model_catalog.list_ollama_models().error
    assert "Network error" in model_catalog.validate_and_list("openrouter", "k").error
    assert not model_catalog.validate_and_list("stub", "k").ok


# --- curl logging ------------------------------------------------------------


def test_curl_redacts_keys_and_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    kw = {
        "url": "https://api.example/v1/chat",
        "headers": {"Authorization": "Bearer sk-secret", "x-api-key": "sk-2", "A": "b"},
        "body": {"q": "it's"},
    }
    curl = curl_log.format_llm_curl(**kw)
    assert "sk-secret" not in curl and "sk-2" not in curl
    assert "$COZMO_API_KEY" in curl

    monkeypatch.delenv("COZMO_DEBUG_CURL", raising=False)
    curl_log.log_llm_curl(**kw)
    assert capsys.readouterr().err == ""
    monkeypatch.setenv("COZMO_DEBUG_CURL", "1")
    curl_log.log_llm_curl(**kw)
    assert "[cozmo llm curl]" in capsys.readouterr().err


# --- session export ----------------------------------------------------------


def _events() -> list[SessionEvent]:
    def ev(kind, /, **data):
        return SessionEvent(kind=kind, session_id="s1", ts=1_700_000_000.0, data=data)

    return [
        ev(EventKind.SESSION_START, model="gpt-4o-mini", provider="openai", workdir="/repo"),
        ev(EventKind.USER_TURN, text="where is add?"),
        ev(EventKind.TOOL, name="read_file", is_error=False, preview="def add\nreturn a+b"),
        ev(EventKind.COMPACT, summary="earlier turns"),
        ev(EventKind.ASSISTANT_TURN, text="math_utils.py:1"),
        ev(EventKind.STOPPED, reason="max_iterations"),
        ev(EventKind.SUBAGENT, goal="scan", kind="team"),
        ev(EventKind.SESSION_END),
    ]


def test_markdown_export_covers_every_event_kind() -> None:
    md = export_session_markdown("s1", _events())
    assert md.startswith("# Cozmo session `s1`")
    assert "openai / gpt-4o-mini" in md
    for needle in ("### User", "where is add?", "read_file", "> def add", "math_utils.py:1"):
        assert needle in md
    assert "### Session end" in md


def test_json_export_round_trips() -> None:
    payload = json.loads(export_session_json("s1", _events(), index={"model": "m"}))
    assert payload["index"] == {"model": "m"}
    restored = [SessionEvent.from_dict(e) for e in payload["events"]]
    assert restored == _events()
