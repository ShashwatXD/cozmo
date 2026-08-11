"""Memory compaction — summarize older turns via injected LLMClient."""

from __future__ import annotations

from cozmo.domain.guardrails import AgentPolicy, estimate_tokens
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.ports import LLMClient
from cozmo.prompts.loader import load_system_prompt


def _history_est_tokens(memory: ConversationMemory) -> int:
    parts = [m.content for m in memory.snapshot()]
    if memory.summary:
        parts.append(memory.summary)
    return estimate_tokens("\n".join(parts))


def needs_compaction(memory: ConversationMemory, policy: AgentPolicy) -> bool:
    return policy.should_compact_messages(len(memory)) or policy.should_compact_tokens(
        _history_est_tokens(memory)
    )


def compact_memory(
    memory: ConversationMemory,
    llm: LLMClient,
    policy: AgentPolicy,
    *,
    keep_recent: int = 12,
    temperature: float = 0.2,
) -> str | None:
    """
    Summarize older messages into memory.summary; keep last keep_recent raw.
    Returns the new summary text, or None if nothing compacted.
    """
    if not needs_compaction(memory, policy):
        return None

    history = memory.snapshot()
    if len(history) <= keep_recent:
        return None

    older = history[:-keep_recent]
    recent = history[-keep_recent:]
    # Don't start recent on orphan tool messages.
    while recent and recent[0].role == Role.TOOL:
        older.append(recent.pop(0))
        if not recent:
            break

    transcript_lines: list[str] = []
    if memory.summary:
        transcript_lines.append(f"Prior summary:\n{memory.summary}")
    for m in older:
        role = m.role.value
        content = (m.content or "")[:2000]
        if m.tool_calls:
            names = ", ".join(tc.name for tc in m.tool_calls)
            content = f"{content} [tools: {names}]".strip()
        transcript_lines.append(f"{role}: {content}")

    system = load_system_prompt("compact")
    prompt = (
        "Compress the following conversation into a short session summary. "
        "Keep goals, decisions, open tasks, and important file paths. "
        "Omit raw tool JSON dumps.\n\n" + "\n".join(transcript_lines)
    )
    result = llm.complete(
        [
            Message(role=Role.SYSTEM, content=system),
            Message(role=Role.USER, content=prompt),
        ],
        temperature=temperature,
        tools=None,
    )
    summary = (result.content or "").strip()
    if not summary:
        return None

    memory.summary = summary
    memory.replace_history(recent)
    # Hard ceiling after compact.
    if len(memory) > policy.memory_max_messages:
        trimmed = memory.snapshot()[-policy.memory_max_messages :]
        while trimmed and trimmed[0].role == Role.TOOL:
            trimmed = trimmed[1:]
        memory.replace_history(trimmed)
    return summary
