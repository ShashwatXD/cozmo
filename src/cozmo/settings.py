"""
Typed app config loaded once at startup.

What: single Settings object (provider, model, workdir, keys, sampling).
Why: no scattered os.getenv in business code.
Layer: bootstraps all layers; not domain logic.
Flutter: like AppConfig / flutter_dotenv into a typed class.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (…/cozmo/) so .env works even if you run from another cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Flutter: AppConfig — one place for knobs."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        env_prefix="COZMO_",
        extra="ignore",
    )

    # stub | openai | ollama
    provider: str = "stub"
    # e.g. gpt-4o-mini | qwen2.5:3b | stub-model
    model: str = "stub-model"
    workdir: Path = Field(default=Path("."))
    openai_api_key: str | None = None
    # Override for Azure/proxy/Ollama; None = OpenAI default
    openai_base_url: str | None = None
    # Sampling: lower = more deterministic (good for coding)
    temperature: float = 0.2
    # Local models can be slow on first load
    timeout_s: float = 120.0
    # How many times to try transient LLM failures (1 = no retry)
    max_retries: int = 3
    # Tool / agent sandbox
    allow_write: bool = True
    allow_shell: bool = False
    max_agent_steps: int = 8
    # Conversation memory sliding window (non-system messages kept)
    memory_max_messages: int = 40
    log_level: str = "INFO"


def load_settings() -> Settings:
    """Load from env / .env. Call from CLI at startup."""
    return Settings()
