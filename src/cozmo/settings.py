"""
Typed app config loaded once at startup.

Single Settings object (provider, model, workdir, keys, sampling).
No scattered os.getenv in business code.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Application configuration from env / .env (prefix COZMO_)."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        env_prefix="COZMO_",
        extra="ignore",
    )

    # stub | openai | ollama
    provider: str = "stub"
    model: str = "stub-model"
    workdir: Path = Field(default=Path("."))
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    temperature: float = 0.2
    timeout_s: float = 120.0
    max_retries: int = 3
    allow_write: bool = True
    allow_shell: bool = False
    max_agent_steps: int = 8
    memory_max_messages: int = 40
    # hash | openai | ollama
    embedder: str = "hash"
    embedding_model: str = "text-embedding-3-small"
    # observability
    trace_enabled: bool = True
    log_level: str = "INFO"


def load_settings() -> Settings:
    return Settings()
