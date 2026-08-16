"""Typed Settings from config files, env, and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from pydantic import AliasChoices, Field
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from cozmo.infra.config.store import load_merged_file_config
from cozmo.infra.mcp.types import McpServerConfig

class _FileJsonSettingsSource(PydanticBaseSettingsSource):

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        data = load_merged_file_config(Path.cwd())
        if field_name in data:
            return data[field_name], field_name, True
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return load_merged_file_config(Path.cwd())

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="COZMO_",
        extra="ignore",
        populate_by_name=True,
    )

    provider: str = "stub"
    model: str = "stub-model"
    worker_model: str | None = None
    workdir: Path = Field(default=Path("."))
    api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("api_key", "openai_api_key"),
    )
    base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("base_url", "openai_base_url"),
    )
    temperature: float = 0.2
    timeout_s: float = 120.0
    max_retries: int = 3
    # Cap completion size so low-credit OpenRouter accounts aren't rejected (402).
    max_tokens: int = 2048
    allow_write: bool = True
    allow_shell: bool = False
    max_agent_steps: int = 8
    memory_max_messages: int = 40
    max_messages_before_compact: int = 30
    context_token_budget: int = 24_000
    max_tool_calls_per_session: int = 40
    max_cost_usd: float | None = None
    session_timeout_s: float | None = 600.0
    max_subagent_depth: int = 1
    max_subagent_steps: int = 4
    shell_timeout_s: float = 60.0
    max_tool_result_chars: int = 24_000
    embedder: str = "auto"
    embedding_model: str = "text-embedding-3-small"
    vector_backend: str = "chroma"
    history_enabled: bool = True
    history_max_sessions: int = 50
    history_max_events_per_session: int = 2000
    history_rag: bool = True
    trace_enabled: bool = True
    mcp_servers: list[McpServerConfig] = Field(default_factory=list)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _FileJsonSettingsSource(settings_cls),
            file_secret_settings,
        )

def load_settings() -> Settings:
    return Settings()
