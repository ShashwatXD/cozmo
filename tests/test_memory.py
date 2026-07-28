"""Conversation memory sliding window."""

from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.tools import ToolCall


def test_for_prompt_includes_system() -> None:
    mem = ConversationMemory(max_messages=10)
    mem.add(Message(role=Role.USER, content="hi"))
    prompt = mem.for_prompt("SYS")
    assert prompt[0].role == Role.SYSTEM
    assert prompt[0].content == "SYS"
    assert prompt[1].content == "hi"


def test_sliding_window_keeps_newest() -> None:
    mem = ConversationMemory(max_messages=4)
    for i in range(10):
        mem.add(Message(role=Role.USER, content=f"u{i}"))
        mem.add(Message(role=Role.ASSISTANT, content=f"a{i}"))
    body = mem.for_prompt("S")[1:]
    assert len(body) <= 4
    assert body[-1].content == "a9"


def test_window_does_not_start_on_orphan_tool() -> None:
    mem = ConversationMemory(max_messages=3)
    mem.add(Message(role=Role.USER, content="q"))
    mem.add(
        Message(
            role=Role.ASSISTANT,
            content="",
            tool_calls=(ToolCall(id="1", name="read_file", arguments="{}"),),
        )
    )
    mem.add(Message(role=Role.TOOL, content="file...", tool_call_id="1", name="read_file"))
    mem.add(Message(role=Role.ASSISTANT, content="done"))
    # Force a tiny window that might cut badly
    mem2 = ConversationMemory(max_messages=2)
    mem2.replace_history(mem.snapshot())
    body = mem2.for_prompt("S")[1:]
    assert body[0].role != Role.TOOL
