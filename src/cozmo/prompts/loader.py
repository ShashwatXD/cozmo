"""Load versioned system prompts from prompts/."""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent

def load_system_prompt(name: str = "default") -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        return "You are Cozmo, a helpful coding assistant in the terminal."
    return path.read_text(encoding="utf-8").strip()
