"""Multi-agent team definitions — pure data, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgentToolAccess(str, Enum):
    """Which tools a team agent may use."""

    NONE = "none"
    READ = "read"
    WORKSPACE = "workspace"  # read + write/shell (reserved; workers stay read-only for now)


@dataclass(frozen=True)
class AgentDef:
    id: str
    name: str
    description: str
    prompt_name: str
    tools: AgentToolAccess
    max_steps: int = 6


READ_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "search_repo",
        "semantic_search",
        "search_history",
        "git_status",
        "git_diff",
    }
)

WORKSPACE_TOOLS: frozenset[str] = READ_TOOLS | frozenset({"write_file", "run_shell"})

_ORCHESTRATOR_NAMES = frozenset({"director", "synthesizer"})


DEFAULT_AGENTS: dict[str, AgentDef] = {
    "director": AgentDef(
        id="director",
        name="director",
        description="Plans work and delegates to specialist workers",
        prompt_name="director",
        tools=AgentToolAccess.NONE,
        max_steps=4,
    ),
    "researcher": AgentDef(
        id="researcher",
        name="researcher",
        description="Explores the codebase and gathers evidence",
        prompt_name="researcher",
        tools=AgentToolAccess.READ,
        max_steps=8,
    ),
    "coder": AgentDef(
        id="coder",
        name="coder",
        description="Analyzes code and proposes focused implementation steps",
        prompt_name="coder",
        tools=AgentToolAccess.READ,
        max_steps=8,
    ),
    "reviewer": AgentDef(
        id="reviewer",
        name="reviewer",
        description="Reviews findings for bugs, gaps, and risks",
        prompt_name="reviewer",
        tools=AgentToolAccess.READ,
        max_steps=6,
    ),
    "synthesizer": AgentDef(
        id="synthesizer",
        name="synthesizer",
        description="Merges parallel worker outputs into one answer",
        prompt_name="synthesizer",
        tools=AgentToolAccess.NONE,
        max_steps=4,
    ),
}


def list_default_workers() -> list[AgentDef]:
    """Workers available for director assignment (excludes director/synthesizer)."""
    return [a for a in DEFAULT_AGENTS.values() if a.name not in _ORCHESTRATOR_NAMES]


def resolve_agent(name_or_id: str) -> AgentDef | None:
    key = name_or_id.strip().lower()
    if key in DEFAULT_AGENTS:
        return DEFAULT_AGENTS[key]
    for agent in DEFAULT_AGENTS.values():
        if agent.id.lower() == key or agent.name.lower() == key:
            return agent
    return None


def allowed_tools_for(agent: AgentDef) -> frozenset[str]:
    if agent.tools == AgentToolAccess.NONE:
        return frozenset()
    if agent.tools == AgentToolAccess.WORKSPACE:
        return WORKSPACE_TOOLS
    return READ_TOOLS
