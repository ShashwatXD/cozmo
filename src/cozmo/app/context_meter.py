"""Approximate context economy for REPL meter (M4)."""

from __future__ import annotations

from dataclasses import dataclass

from cozmo.app.compaction import estimate_context_tokens, needs_compaction
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.memory import ConversationMemory


@dataclass(frozen=True)
class ContextMeter:
    est_tokens: int
    budget: int
    pct: int
    near_compact: bool
    has_summary: bool

    def format_line(self) -> str:
        """Short footer fragment, e.g. ctx≈3.2k/24k (13%)."""
        est = _fmt_k(self.est_tokens)
        bud = _fmt_k(self.budget)
        bits = [f"ctx≈{est}/{bud} ({self.pct}%)"]
        if self.has_summary:
            bits.append("compacted")
        elif self.near_compact:
            bits.append("near compact")
        return " · ".join(bits)


def context_meter(
    memory: ConversationMemory,
    policy: AgentPolicy,
    *,
    system_prompt: str = "",
) -> ContextMeter:
    est = estimate_context_tokens(memory, system_prompt=system_prompt)
    budget = max(1, policy.context_token_budget)
    pct = min(100, int(round(100 * est / budget)))
    return ContextMeter(
        est_tokens=est,
        budget=budget,
        pct=pct,
        near_compact=needs_compaction(memory, policy),
        has_summary=bool(memory.summary),
    )


def _fmt_k(n: int) -> str:
    if n >= 1000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return str(n)
