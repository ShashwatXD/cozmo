"""Session agent modes."""

from __future__ import annotations

from enum import Enum


class AgentMode(str, Enum):
    """REPL / CLI session mode."""

    AGENT = "agent"  # full tools subject to permission prompts
    PLAN = "plan"  # explore then propose; mutating tools blocked until /agent
    ASK = "ask"


MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "run_shell"})

MCP_PREFIX = "mcp__"

READ_ONLY_MODES: frozenset[AgentMode] = frozenset({AgentMode.ASK, AgentMode.PLAN})


def is_mcp_tool(name: str) -> bool:
    return name.startswith(MCP_PREFIX)


def requires_approval(name: str) -> bool:
    """MCP tools are third-party code, so they count as mutating until allowed."""
    return name in MUTATING_TOOLS or is_mcp_tool(name)


def prompt_name_for_mode(mode: AgentMode) -> str:
    if mode == AgentMode.ASK:
        return "ask"
    if mode == AgentMode.PLAN:
        return "plan"
    return "agent"
