"""Phase 1: ChatUseCase + stub (no network)."""

from cozmo.app.chat import ChatUseCase
from cozmo.infra.llm.stub import StubLLMClient


def test_chat_complete() -> None:
    uc = ChatUseCase(StubLLMClient(), system_prompt="You are test.")
    result = uc.run("hello")
    assert "stub" in result.content.lower()


def test_chat_stream() -> None:
    uc = ChatUseCase(StubLLMClient(), system_prompt="sys")
    chunks = list(uc.stream("hello"))
    assert len(chunks) >= 1
    assert "stub" in "".join(chunks).lower()


def test_json_mode_stub() -> None:
    uc = ChatUseCase(StubLLMClient())
    result = uc.run("ping", json_mode=True)
    assert result.content.strip().startswith("{")
