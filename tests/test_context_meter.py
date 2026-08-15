"""Context economy meter (M4)."""

from cozmo.app.context_meter import context_meter
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role


def test_context_meter_pct() -> None:
    memory = ConversationMemory()
    memory.add(Message(role=Role.USER, content="x" * 400))  # ~100 tokens
    policy = AgentPolicy(context_token_budget=1000)
    meter = context_meter(memory, policy, system_prompt="")
    assert meter.est_tokens >= 50
    assert meter.budget == 1000
    assert 0 < meter.pct <= 100
    line = meter.format_line()
    assert "ctx≈" in line
    assert "%" in line


def test_context_meter_near_compact() -> None:
    memory = ConversationMemory()
    # Force message-count compact threshold
    for i in range(30):
        memory.add(Message(role=Role.USER, content=f"msg {i}"))
    policy = AgentPolicy(
        max_messages_before_compact=30,
        context_token_budget=1_000_000,
    )
    meter = context_meter(memory, policy)
    assert meter.near_compact
    assert "near compact" in meter.format_line()
