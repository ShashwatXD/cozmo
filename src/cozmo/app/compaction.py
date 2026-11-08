"""Memory compaction — summarize older turns via injected LLMClient."""

from __future__ import annotations

from dataclasses import replace

from cozmo.domain.guardrails import AgentPolicy, estimate_tokens
from cozmo.domain.memory import ConversationMemory
from cozmo.domain.messages import Message, Role
from cozmo.domain.ports import LLMClient
from cozmo.prompts.loader import load_system_prompt

_ELIDED = "[older tool output removed to save context:"
_ELIDE_MIN_CHARS = 400
_ELIDE_KEEP_CHARS = 200


def estimate_context_tokens(
    memory: ConversationMemory,
    *,
    system_prompt: str = "",
) -> int:
    """Rough context size for budgets and the REPL meter (char/4)."""
    parts = [m.content for m in memory.snapshot()]
    if memory.summary:
        parts.append(memory.summary)
    if system_prompt:
        parts.append(system_prompt)
    return estimate_tokens("\n".join(parts))


def estimate_messages_tokens(messages: list[Message]) -> int:
    """Size of the exact prompt about to be sent, including tool-call arguments."""
    parts: list[str] = []
    for m in messages:
        parts.append(m.content)
        parts.extend(tc.arguments for tc in m.tool_calls)
    return estimate_tokens("\n".join(parts))


def shrink_tool_results(
    messages: list[Message], policy: AgentPolicy, *, keep_recent: int = 4
) -> tuple[list[Message], int]:
    """
    Stub out old tool results when a turn nears its budget. Free and deterministic.

    Tool output is the bulk of a coding agent's context and is the easiest thing to
    get back, because the model can run the same read again.
    """
    if not policy.should_compact_tokens(estimate_messages_tokens(messages)):
        return messages, 0

    out = list(messages)
    elided = 0
    for keep in (keep_recent, 1):  # second pass only if still over the whole budget
        tool_idx = [i for i, m in enumerate(out) if m.role == Role.TOOL]
        for i in tool_idx[:-keep]:
            msg = out[i]
            if len(msg.content) > _ELIDE_MIN_CHARS and _ELIDED not in msg.content:
                head = msg.content[:_ELIDE_KEEP_CHARS]
                note = f"\n{_ELIDED} {len(msg.content)} chars; run {msg.name} again if needed]"
                out[i] = replace(msg, content=head + note)
                elided += 1
        if estimate_messages_tokens(out) < policy.context_token_budget:
            break
    return out, elided


def needs_compaction(memory: ConversationMemory, policy: AgentPolicy) -> bool:
    return policy.should_compact_messages(len(memory)) or policy.should_compact_tokens(
        estimate_context_tokens(memory)
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
