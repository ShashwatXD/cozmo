"""Memory compaction with stub LLM."""

from cozmo.app.compaction import compact_memory, needs_compaction
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.infra.llm.stub import StubLLMClient


def test_needs_compaction_by_count() -> None:
    mem = ConversationMemory(max_messages=40)
    for i in range(12):
        mem.add(Message(role=Role.USER, content=f"u{i}"))
        mem.add(Message(role=Role.ASSISTANT, content=f"a{i}"))
    policy = AgentPolicy(max_messages_before_compact=20)
    assert needs_compaction(mem, policy)


def test_compact_memory_summarizes() -> None:
    mem = ConversationMemory(max_messages=40)
    for i in range(20):
        mem.add(Message(role=Role.USER, content=f"user goal {i}"))
        mem.add(Message(role=Role.ASSISTANT, content=f"assistant reply {i}"))
    policy = AgentPolicy(max_messages_before_compact=10, memory_max_messages=40)
    llm = StubLLMClient(final_text="Summary: user was iterating goals.")
    summary = compact_memory(mem, llm, policy, keep_recent=6)
    assert summary is not None
    assert mem.summary is not None
    assert len(mem) <= 6
    prompt = mem.for_prompt("SYS")
    assert any("Session summary" in m.content for m in prompt if m.role == Role.SYSTEM)
