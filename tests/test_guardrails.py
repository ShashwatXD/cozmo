"""Guardrail policy unit tests."""

from cozmo.domain.guardrails import AgentPolicy, StopReason, estimate_tokens


def test_max_iterations() -> None:
    p = AgentPolicy(max_agent_steps=3)
    assert p.check_steps(4) == StopReason.MAX_ITERATIONS
    assert p.check_steps(3) is None


def test_compact_triggers() -> None:
    p = AgentPolicy(max_messages_before_compact=10, context_token_budget=1000)
    assert p.should_compact_messages(10)
    assert not p.should_compact_messages(9)
    assert p.should_compact_tokens(700)
    assert not p.should_compact_tokens(100)


def test_tool_and_cost_kills() -> None:
    p = AgentPolicy(max_tool_calls_per_session=2, max_cost_usd=0.01)
    assert p.check_tool_calls(3) == StopReason.MAX_TOOL_CALLS
    assert p.check_cost(0.02) == StopReason.MAX_COST


def test_subagent_policy_tighter() -> None:
    p = AgentPolicy(max_agent_steps=12, max_subagent_steps=4, max_subagent_depth=2)
    child = p.for_subagent()
    assert child.max_agent_steps == 4
    assert child.max_subagent_depth == 1


def test_estimate_tokens() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
