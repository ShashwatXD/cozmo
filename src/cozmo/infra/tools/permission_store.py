"""Persist project-level mutating-tool allow/deny rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozmo.infra.config.paths import permissions_path

SCHEMA_VERSION = 1


@dataclass
class PermissionRule:
    tool: str
    path: str = ""  # write_file: fnmatch against relative path
    pattern: str = ""  # run_shell: fnmatch against command


@dataclass
class PermissionStore:
    """In-memory + JSON-backed allow/deny lists under .cozmo/permissions.json."""

    allow: list[PermissionRule] = field(default_factory=list)
    deny: list[PermissionRule] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def load(cls, workdir: Path) -> PermissionStore:
        path = permissions_path(workdir)
        store = cls(path=path)
        if not path.is_file():
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return store
        if not isinstance(data, dict):
            return store
        store.allow = _parse_rules(data.get("allow"))
        store.deny = _parse_rules(data.get("deny"))
        return store

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "allow": [_rule_to_dict(r) for r in self.allow],
            "deny": [_rule_to_dict(r) for r in self.deny],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def add_allow(self, rule: PermissionRule) -> None:
        if not _has_rule(self.allow, rule):
            self.allow.append(rule)
            self.save()

    def add_deny(self, rule: PermissionRule) -> None:
        if not _has_rule(self.deny, rule):
            self.deny.append(rule)
            self.save()


def _parse_rules(raw: Any) -> list[PermissionRule]:
    if not isinstance(raw, list):
        return []
    out: list[PermissionRule] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        if not tool:
            continue
        out.append(
            PermissionRule(
                tool=tool,
                path=str(item.get("path") or "").strip(),
                pattern=str(item.get("pattern") or "").strip(),
            )
        )
    return out


def _rule_to_dict(rule: PermissionRule) -> dict[str, str]:
    d: dict[str, str] = {"tool": rule.tool}
    if rule.path:
        d["path"] = rule.path
    if rule.pattern:
        d["pattern"] = rule.pattern
    return d


def _has_rule(rules: list[PermissionRule], rule: PermissionRule) -> bool:
    return any(
        r.tool == rule.tool and r.path == rule.path and r.pattern == rule.pattern
        for r in rules
    )
