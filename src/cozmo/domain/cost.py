"""
Cost estimation from token usage.

What: map (model, Usage) → approximate USD.
Why: every production gateway tracks spend; local/Ollama = $0.
Layer: domain (pure math) - no network.
"""

from __future__ import annotations

from cozmo.domain.completion import Usage

# (input_per_1M_tokens, output_per_1M_tokens) USD - rough public list prices
# Update when you care; interviews care that you *track*, not exact cents.
_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost_usd(model: str, usage: Usage) -> float | None:
    """
    Return estimated USD, 0.0 for local/unknown-free, or None if priced but no tokens.
    """
    key = model.strip().lower()
    # Local / stub - no cloud bill
    if key.startswith("stub") or ":" in key or key.startswith("llama") or key.startswith(
        "qwen"
    ):
        return 0.0

    rates = _PRICE_PER_1M.get(key)
    if rates is None:
        # Unknown cloud model - still return 0 with honesty later via CLI note
        return 0.0

    in_rate, out_rate = rates
    if usage.total_tokens == 0:
        return None
    return (usage.prompt_tokens / 1_000_000) * in_rate + (
        usage.completion_tokens / 1_000_000
    ) * out_rate


def format_cost_line(
    *,
    provider: str,
    model: str,
    usage: Usage,
) -> str | None:
    """One log line for the CLI footer, or None if nothing useful."""
    if usage.total_tokens <= 0 and provider == "stub":
        return None
    cost = estimate_cost_usd(model, usage)
    parts = [f"[{provider}/{model}]", f"tokens={usage.total_tokens}"]
    if cost is not None:
        parts.append(f"est_usd=${cost:.6f}")
    return " ".join(parts)
