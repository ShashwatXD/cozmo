"""Mutating-tool permission gate: preview + allow once / always / deny."""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatch
from pathlib import Path

from cozmo.domain.mode import MUTATING_TOOLS, READ_ONLY_MODES, AgentMode
from cozmo.domain.tools import ToolCall, ToolResult
from cozmo.infra.tools.permission_store import PermissionRule, PermissionStore


class PermissionChoice(str, Enum):
    ALLOW_ONCE = "allow_once"
    ALWAYS_ALLOW = "always_allow"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    choice: PermissionChoice
    reason: str = ""


AskFn = Callable[[ToolCall, str], PermissionChoice]


@dataclass
class PermissionGate:
    """
    Gate write_file / run_shell before ToolExecutor.execute.

    Precedence: plan mode → deny rules → allow rules → ask (or default deny).
    WorkspaceGuard remains the hard sandbox underneath.
    """

    workdir: Path
    store: PermissionStore
    mode: AgentMode = AgentMode.AGENT
    ask: AskFn | None = None
    default_choice: PermissionChoice = PermissionChoice.DENY

    def set_mode(self, mode: AgentMode) -> None:
        self.mode = mode

    def decide(self, call: ToolCall) -> PermissionDecision:
        if call.name not in MUTATING_TOOLS:
            return PermissionDecision(
                allowed=True, choice=PermissionChoice.ALLOW_ONCE, reason="read-only"
            )

        if self.mode in READ_ONLY_MODES:
            label = self.mode.value
            return PermissionDecision(
                allowed=False,
                choice=PermissionChoice.DENY,
                reason=f"{label} mode: mutating tools blocked (switch with /agent)",
            )

        args = _parse_args(call.arguments)
        if _matches_rules(call.name, args, self.store.deny):
            return PermissionDecision(
                allowed=False,
                choice=PermissionChoice.DENY,
                reason="denied by .cozmo/permissions.json",
            )
        if _matches_rules(call.name, args, self.store.allow):
            return PermissionDecision(
                allowed=True,
                choice=PermissionChoice.ALWAYS_ALLOW,
                reason="allowed by .cozmo/permissions.json",
            )

        preview = build_preview(self.workdir, call, args)
        choice = self.ask(call, preview) if self.ask else self.default_choice

        if choice == PermissionChoice.ALWAYS_ALLOW:
            self.store.add_allow(_rule_for_call(call.name, args))
            return PermissionDecision(
                allowed=True, choice=choice, reason="always allow (saved)"
            )
        if choice == PermissionChoice.ALLOW_ONCE:
            return PermissionDecision(
                allowed=True, choice=choice, reason="allow once"
            )
        return PermissionDecision(
            allowed=False, choice=PermissionChoice.DENY, reason="denied by user"
        )

    def deny_result(self, call: ToolCall, decision: PermissionDecision) -> ToolResult:
        msg = decision.reason or "Permission denied"
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=f"Permission denied: {msg}",
            is_error=True,
        )


def build_preview(workdir: Path, call: ToolCall, args: dict | None = None) -> str:
    """Human-readable preview for the permission prompt."""
    parsed = args if args is not None else _parse_args(call.arguments)
    if call.name == "write_file":
        return _preview_write(workdir, parsed)
    if call.name == "run_shell":
        cmd = str(parsed.get("command") or "")
        return f"cwd: {workdir}\n$ {cmd}"
    return f"{call.name}\n{call.arguments[:500]}"


def _preview_write(workdir: Path, args: dict) -> str:
    rel = str(args.get("path") or "")
    new_text = str(args.get("content") or "")
    path = (workdir / rel).resolve() if rel else None
    old_text = ""
    label = rel or "(no path)"
    if path is not None and path.is_file():
        try:
            old_text = path.read_text(encoding="utf-8")
        except OSError:
            old_text = ""
    if not old_text and not new_text:
        return f"write_file {label}\n(empty)"
    if not old_text:
        lines = new_text.splitlines()
        head = "\n".join(f"+ {ln}" for ln in lines[:80])
        more = f"\n… ({len(lines) - 80} more lines)" if len(lines) > 80 else ""
        return f"create {label} ({len(new_text)} chars)\n{head}{more}"

    diff = difflib.unified_diff(
        old_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        lineterm="",
    )
    lines = list(diff)
    if not lines:
        return f"write_file {label}\n(no textual diff)"
    body = "\n".join(lines[:120])
    more = f"\n… ({len(lines) - 120} more diff lines)" if len(lines) > 120 else ""
    return f"write_file {label}\n{body}{more}"


def _parse_args(raw: str) -> dict:
    try:
        data = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _rule_for_call(tool: str, args: dict) -> PermissionRule:
    if tool == "write_file":
        path = str(args.get("path") or "").strip() or "*"
        return PermissionRule(tool=tool, path=path)
    if tool == "run_shell":
        pattern = str(args.get("command") or "").strip() or "*"
        return PermissionRule(tool=tool, pattern=pattern)
    return PermissionRule(tool=tool)


def _matches_rules(tool: str, args: dict, rules: list[PermissionRule]) -> bool:
    for rule in rules:
        if rule.tool != tool:
            continue
        if tool == "write_file":
            path = str(args.get("path") or "")
            pattern = rule.path or "*"
            if fnmatch(path, pattern):
                return True
        elif tool == "run_shell":
            command = str(args.get("command") or "")
            pattern = rule.pattern or "*"
            if fnmatch(command, pattern):
                return True
        else:
            return True
    return False
