"""
Load versioned system prompts from prompts/.

What: read text files that become the system message.
Why: prompts are not hardcoded inside the agent loop (DRY + versionable).
Layer: small helper used by app (files are content, not infra SDKs).
Flutter: like loading copy from ARB / asset strings.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_system_prompt(name: str = "default") -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        return "You are Cozmo, a helpful coding assistant in the terminal."
    return path.read_text(encoding="utf-8").strip()
