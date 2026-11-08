"""Cost estimation from token usage."""

from __future__ import annotations

import threading

from cozmo.domain.completion import Usage

# (input_per_1M_tokens, output_per_1M_tokens) USD - rough public list prices
_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
}


Prices = dict[str, tuple[float, float]]


def _is_local(key: str) -> bool:
    return ":" in key or key.startswith(("stub", "llama", "qwen"))


def price_known(model: str, prices: Prices | None = None) -> bool:
    key = model.strip().lower()
    return _is_local(key) or key in _PRICE_PER_1M or key in (prices or {})


def estimate_cost_usd(model: str, usage: Usage, prices: Prices | None = None) -> float | None:
    """
    Return estimated USD, 0.0 for local/unknown-free, or None if priced but no tokens.
    """
    key = model.strip().lower()
    rates = (prices or {}).get(key)
    # Local / stub - no cloud bill
    if rates is None and _is_local(key):
        return 0.0

    rates = rates or _PRICE_PER_1M.get(key)
    if rates is None:
        return 0.0

    in_rate, out_rate = rates
    if usage.total_tokens == 0:
        return None
    return (usage.prompt_tokens / 1_000_000) * in_rate + (
        usage.completion_tokens / 1_000_000
    ) * out_rate


class UsageLedger:
    """
    One running total for a whole session.

    Every model client records here, so compaction, director, synthesizer and
    subagent spend all count against the same budget. Thread-safe for parallel workers.
    """

    def __init__(self, prices: Prices | None = None) -> None:
        self._prices = {k.lower(): v for k, v in (prices or {}).items()}
        self._lock = threading.Lock()
        self.total_cost = 0.0
        self.calls = 0
        self.retries = 0
        self.by_model: dict[str, Usage] = {}
        self.unpriced: set[str] = set()

    def record(self, model: str, usage: Usage) -> float:
        cost = estimate_cost_usd(model, usage, self._prices) or 0.0
        with self._lock:
            self.calls += 1
            self.total_cost += cost
            self.by_model[model] = self.by_model.get(model, Usage()).merged(usage)
            if not price_known(model, self._prices):
                self.unpriced.add(model)
        return cost

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1


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
