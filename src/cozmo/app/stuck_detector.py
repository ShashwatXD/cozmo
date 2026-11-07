"""Detect repeating tool loops in the ReAct agent (OpenHands-style heuristics)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class StuckStep:
    tool_name: str
    args_key: str
    result_key: str
    is_error: bool


StuckReason = Literal[
    "repeating_action_observation",
    "repeating_action_error",
    "alternating_pattern",
]


@dataclass(frozen=True)
class StuckVerdict:
    stuck: bool
    reason: StuckReason = "repeating_action_observation"
    message: str = ""


# OpenHands-style defaults: 4 / 4 / 6
THRESHOLDS = {
    "action_observation": 4,
    "action_error": 4,
    "alternating": 6,
}


def _fingerprint(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)[:800]
    except (TypeError, ValueError):
        return str(value)[:800]


def step_from_tool_call(
    *,
    tool_name: str,
    args: Any,
    result: Any,
    is_error: bool = False,
) -> StuckStep:
    return StuckStep(
        tool_name=tool_name,
        args_key=_fingerprint(args),
        result_key=_fingerprint(result),
        is_error=is_error,
    )


def detect_stuck(steps: list[StuckStep]) -> StuckVerdict:
    """Return stuck=True when recent tool history matches a known loop pattern."""
    min_needed = min(THRESHOLDS["action_observation"], THRESHOLDS["action_error"])
    if len(steps) < min_needed:
        return StuckVerdict(stuck=False)

    recent = steps[-max(THRESHOLDS["alternating"], THRESHOLDS["action_observation"]) :]

    # Same tool + args with errors N times (before identical observation — errors match too)
    if len(recent) >= THRESHOLDS["action_error"]:
        window = recent[-THRESHOLDS["action_error"] :]
        first = window[0]
        if all(
            s.tool_name == first.tool_name and s.args_key == first.args_key and s.is_error
            for s in window
        ):
            return StuckVerdict(
                stuck=True,
                reason="repeating_action_error",
                message=(
                    "Stuck detected: the same failing tool call repeated. "
                    "Do not retry identically — fix inputs, inspect errors, or switch strategy."
                ),
            )

    # Same tool + args + result repeated N times
    if len(recent) >= THRESHOLDS["action_observation"]:
        window = recent[-THRESHOLDS["action_observation"] :]
        first = window[0]
        if all(
            s.tool_name == first.tool_name
            and s.args_key == first.args_key
            and s.result_key == first.result_key
            for s in window
        ):
            return StuckVerdict(
                stuck=True,
                reason="repeating_action_observation",
                message=(
                    "Stuck detected: the same tool call and result repeated. "
                    "Stop looping. Change approach, read different files, or ask the user."
                ),
            )

    # Alternating A/B pattern (A,B,A,B,A,B)
    if len(recent) >= THRESHOLDS["alternating"]:
        window = recent[-THRESHOLDS["alternating"] :]
        alt = all(
            s.tool_name == window[i - 2].tool_name
            and s.args_key == window[i - 2].args_key
            and s.result_key == window[i - 2].result_key
            for i, s in enumerate(window)
            if i >= 2
        ) and not (
            window[0].tool_name == window[1].tool_name and window[0].args_key == window[1].args_key
        )
        if alt:
            return StuckVerdict(
                stuck=True,
                reason="alternating_pattern",
                message=(
                    "Stuck detected: alternating tool loop. "
                    "Break the cycle — synthesize findings and take a new action or finish."
                ),
            )

    return StuckVerdict(stuck=False)
