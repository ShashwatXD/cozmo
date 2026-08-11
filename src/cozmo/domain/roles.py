"""Model roles for multi-model routing."""

from __future__ import annotations

from enum import Enum


class ModelRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    # Optional later — not required for v1 multi-model.
    VERIFIER = "verifier"
