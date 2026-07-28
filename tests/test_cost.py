"""Cost estimator (pure)."""

from cozmo.domain.completion import Usage
from cozmo.domain.cost import estimate_cost_usd, format_cost_line


def test_ollama_model_is_free() -> None:
    assert estimate_cost_usd("qwen2.5:3b", Usage(10, 20)) == 0.0


def test_gpt4o_mini_has_positive_cost() -> None:
    cost = estimate_cost_usd(
        "gpt-4o-mini", Usage(prompt_tokens=1_000_000, completion_tokens=0)
    )
    assert cost == 0.15


def test_format_cost_line() -> None:
    line = format_cost_line(
        provider="openai",
        model="gpt-4o-mini",
        usage=Usage(100, 50),
    )
    assert line is not None
    assert "tokens=150" in line
    assert "est_usd=" in line
