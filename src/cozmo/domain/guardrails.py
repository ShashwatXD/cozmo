"""Agent policy / guardrails — pure rules, no I/O."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class StopReason(str, Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    MAX_COST = "max_cost"
    SESSION_TIMEOUT = "session_timeout"
    MAX_TOOL_CALLS = "max_tool_calls"
    SUBAGENT_DEPTH = "subagent_depth"


@dataclass(frozen=True)
class AgentPolicy:
    """Budgets and soft/hard limits for one agent (or nested subagent)."""

    max_agent_steps: int = 8
    max_messages_before_compact: int = 30
    memory_max_messages: int = 40
    context_token_budget: int = 24_000
    max_tool_calls_per_session: int = 40
    max_cost_usd: float | None = None
    session_timeout_s: float | None = 600.0
    max_subagent_depth: int = 1
    max_subagent_steps: int = 4
    shell_timeout_s: float = 60.0

    def for_subagent(self) -> AgentPolicy:
        """Tighter copy for a nested worker."""
        return replace(
            self,
            max_agent_steps=min(self.max_agent_steps, self.max_subagent_steps),
            max_subagent_depth=max(0, self.max_subagent_depth - 1),
            max_tool_calls_per_session=min(self.max_tool_calls_per_session, 20),
            max_messages_before_compact=min(self.max_messages_before_compact, 16),
            memory_max_messages=min(self.memory_max_messages, 24),
        )

    def should_compact_messages(self, message_count: int) -> bool:
        return message_count >= self.max_messages_before_compact

    def should_compact_tokens(self, est_tokens: int) -> bool:
        return est_tokens >= int(self.context_token_budget * 0.7)

    def check_steps(self, step: int) -> StopReason | None:
        if step > self.max_agent_steps:
            return StopReason.MAX_ITERATIONS
        return None

    def check_tool_calls(self, tool_calls: int) -> StopReason | None:
        if tool_calls > self.max_tool_calls_per_session:
            return StopReason.MAX_TOOL_CALLS
        return None

    def check_cost(self, cost_usd: float) -> StopReason | None:
        if self.max_cost_usd is not None and cost_usd > self.max_cost_usd:
            return StopReason.MAX_COST
        return None

    def check_session_timeout(self, elapsed_s: float) -> StopReason | None:
        if self.session_timeout_s is not None and elapsed_s > self.session_timeout_s:
            return StopReason.SESSION_TIMEOUT
        return None

    def check_subagent_depth(self, depth: int) -> StopReason | None:
        if depth > self.max_subagent_depth:
            return StopReason.SUBAGENT_DEPTH
        return None

    @staticmethod
    def from_mapping(data: dict[str, Any]) -> AgentPolicy:
        """Build from Settings-like dict without importing pydantic."""
        known = {f.name for f in AgentPolicy.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and v is not None}
        return AgentPolicy(**kwargs)

    @classmethod
    def from_settings(cls, settings: Any) -> AgentPolicy:
        """Accept a Settings object or mapping."""
        if isinstance(settings, dict):
            return cls.from_mapping(settings)
        data = {
            "max_agent_steps": getattr(settings, "max_agent_steps", 8),
            "max_messages_before_compact": getattr(
                settings, "max_messages_before_compact", 30
            ),
            "memory_max_messages": getattr(settings, "memory_max_messages", 40),
            "context_token_budget": getattr(settings, "context_token_budget", 24_000),
            "max_tool_calls_per_session": getattr(
                settings, "max_tool_calls_per_session", 40
            ),
            "max_cost_usd": getattr(settings, "max_cost_usd", None),
            "session_timeout_s": getattr(settings, "session_timeout_s", 600.0),
            "max_subagent_depth": getattr(settings, "max_subagent_depth", 1),
            "max_subagent_steps": getattr(settings, "max_subagent_steps", 4),
            "shell_timeout_s": getattr(settings, "shell_timeout_s", 60.0),
        }
        return cls.from_mapping(data)


def estimate_tokens(text: str) -> int:
    """Rough char/4 estimate — good enough for budgets without a tokenizer."""
    if not text:
        return 0
    return max(1, len(text) // 4)
