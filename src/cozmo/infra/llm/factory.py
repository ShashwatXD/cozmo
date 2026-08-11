"""Build LLMClient from Settings."""

from cozmo.domain.ports import LLMClient
from cozmo.infra.llm.anthropic_client import AnthropicClient
from cozmo.infra.llm.openai_compatible import OpenAICompatibleClient
from cozmo.infra.llm.retrying import RetryingLLMClient
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.settings import Settings

_OLLAMA_BASE = "http://127.0.0.1:11434/v1"
_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENAI_COMPAT = frozenset({"openai", "openrouter"})


def build_llm(settings: Settings) -> LLMClient:
    inner = _build_inner(settings)
    if settings.max_retries <= 1:
        return inner
    return RetryingLLMClient(inner, max_attempts=settings.max_retries)


def _compat_base_url(settings: Settings) -> str | None:
    if settings.base_url:
        return settings.base_url
    if settings.provider == "openrouter":
        return _OPENROUTER_BASE
    return None


def _build_inner(settings: Settings) -> LLMClient:
    if settings.provider == "stub":
        return StubLLMClient()

    if settings.provider == "anthropic":
        if not settings.api_key:
            raise ValueError("COZMO_API_KEY is required when provider=anthropic")
        return AnthropicClient(
            model=settings.model,
            api_key=settings.api_key,
            timeout_s=settings.timeout_s,
            max_tokens=settings.max_tokens,
        )

    if settings.provider in _OPENAI_COMPAT:
        if not settings.api_key:
            raise ValueError(
                f"COZMO_API_KEY is required when provider={settings.provider}"
            )
        return OpenAICompatibleClient(
            model=settings.model,
            api_key=settings.api_key,
            base_url=_compat_base_url(settings),
            timeout_s=settings.timeout_s,
            max_tokens=settings.max_tokens,
        )

    if settings.provider == "ollama":
        return OpenAICompatibleClient(
            model=settings.model,
            api_key=settings.api_key or "ollama",
            base_url=settings.base_url or _OLLAMA_BASE,
            timeout_s=settings.timeout_s,
            max_tokens=settings.max_tokens,
        )

    raise ValueError(
        f"Unknown provider '{settings.provider}'. "
        "Supported: stub, openai, anthropic, openrouter, ollama"
    )
