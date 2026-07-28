"""
Wiring: Settings → LLM adapter (+ retries).

What: factory that returns an LLMClient.
Why: swap stub/openai/ollama by config; wrap with retries (DI).
Layer: infra.
"""

from cozmo.domain.ports import LLMClient
from cozmo.infra.llm.openai_compatible import OpenAICompatibleClient
from cozmo.infra.llm.retrying import RetryingLLMClient
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.settings import Settings

_OLLAMA_BASE = "http://127.0.0.1:11434/v1"


def build_llm(settings: Settings) -> LLMClient:
    inner = _build_inner(settings)
    if settings.max_retries <= 1:
        return inner
    return RetryingLLMClient(inner, max_attempts=settings.max_retries)


def _build_inner(settings: Settings) -> LLMClient:
    if settings.provider == "stub":
        return StubLLMClient()

    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "COZMO_OPENAI_API_KEY is required when COZMO_PROVIDER=openai"
            )
        return OpenAICompatibleClient(
            model=settings.model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout_s=settings.timeout_s,
        )

    if settings.provider == "ollama":
        return OpenAICompatibleClient(
            model=settings.model,
            api_key=settings.openai_api_key or "ollama",
            base_url=settings.openai_base_url or _OLLAMA_BASE,
            timeout_s=settings.timeout_s,
        )

    raise ValueError(
        f"Unknown provider '{settings.provider}'. Supported: stub, openai, ollama"
    )
