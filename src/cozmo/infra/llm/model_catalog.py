"""Validate API keys and list models for setup."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

OPENAI_FAVORITES = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o4-mini",
    "o3-mini",
)

ANTHROPIC_FAVORITES = (
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-5-sonnet-20241022",
)

OPENROUTER_FAVORITES = (
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.5-pro",
    "meta-llama/llama-3.3-70b-instruct",
)

@dataclass(frozen=True)
class ModelCatalogResult:
    ok: bool
    models: tuple[str, ...] = ()
    error: str = ""
    favorites: tuple[str, ...] = ()

def _merge_favorites(fetched: list[str], favorites: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    fetched_set = set(fetched)
    for fav in favorites:
        if fav not in seen:
            out.append(fav)
            seen.add(fav)
    for mid in sorted(fetched):
        if mid not in seen:
            out.append(mid)
            seen.add(mid)
    if not fetched and not out:
        return favorites
    return tuple(out)

def validate_and_list_openai(
    api_key: str,
    *,
    base_url: str | None = None,
    timeout_s: float = 30.0,
) -> ModelCatalogResult:
    root = (base_url or "https://api.openai.com/v1").rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(
                f"{root}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code in {401, 403}:
            return ModelCatalogResult(ok=False, error="Invalid API key (unauthorized)")
        if resp.status_code >= 400:
            return ModelCatalogResult(
                ok=False, error=f"API error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json().get("data") or []
        ids = [str(m.get("id")) for m in data if m.get("id")]
        chatty = [
            i
            for i in ids
            if any(
                x in i
                for x in ("gpt", "o1", "o3", "o4", "chatgpt", "davinci")
            )
            and "embedding" not in i
            and "tts" not in i
            and "whisper" not in i
            and "dall-e" not in i
            and "moderation" not in i
        ]
        use = chatty or ids
        return ModelCatalogResult(
            ok=True,
            models=_merge_favorites(use, OPENAI_FAVORITES),
            favorites=OPENAI_FAVORITES,
        )
    except httpx.HTTPError as exc:
        return ModelCatalogResult(ok=False, error=f"Network error: {exc}")

def validate_and_list_anthropic(
    api_key: str,
    *,
    timeout_s: float = 30.0,
) -> ModelCatalogResult:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        if resp.status_code in {401, 403}:
            return ModelCatalogResult(ok=False, error="Invalid API key (unauthorized)")
        if resp.status_code >= 400:
            if resp.status_code == 404:
                return ModelCatalogResult(
                    ok=True,
                    models=ANTHROPIC_FAVORITES,
                    favorites=ANTHROPIC_FAVORITES,
                )
            return ModelCatalogResult(
                ok=False, error=f"API error {resp.status_code}: {resp.text[:200]}"
            )
        raw = resp.json()
        data = raw.get("data") or raw.get("models") or []
        ids = []
        for m in data:
            mid = m.get("id") or m.get("name")
            if mid:
                ids.append(str(mid))
        return ModelCatalogResult(
            ok=True,
            models=_merge_favorites(ids, ANTHROPIC_FAVORITES),
            favorites=ANTHROPIC_FAVORITES,
        )
    except httpx.HTTPError as exc:
        return ModelCatalogResult(ok=False, error=f"Network error: {exc}")

def validate_and_list_openrouter(
    api_key: str,
    *,
    timeout_s: float = 30.0,
) -> ModelCatalogResult:
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        if resp.status_code in {401, 403}:
            return ModelCatalogResult(ok=False, error="Invalid API key (unauthorized)")
        if resp.status_code >= 400:
            return ModelCatalogResult(
                ok=False, error=f"API error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json().get("data") or []
        ids = [str(m.get("id")) for m in data if m.get("id")]
        return ModelCatalogResult(
            ok=True,
            models=_merge_favorites(ids[:80], OPENROUTER_FAVORITES),
            favorites=OPENROUTER_FAVORITES,
        )
    except httpx.HTTPError as exc:
        return ModelCatalogResult(ok=False, error=f"Network error: {exc}")

def list_ollama_models(
    base_url: str = "http://127.0.0.1:11434",
    *,
    timeout_s: float = 10.0,
) -> ModelCatalogResult:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{root}/api/tags")
        if resp.status_code >= 400:
            return ModelCatalogResult(
                ok=False, error=f"Ollama not reachable ({resp.status_code})"
            )
        models = resp.json().get("models") or []
        names = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                names.append(str(name))
        if not names:
            return ModelCatalogResult(
                ok=False,
                error="No local models found. Run: ollama pull qwen2.5:3b",
            )
        return ModelCatalogResult(ok=True, models=tuple(sorted(names)))
    except httpx.HTTPError as exc:
        return ModelCatalogResult(
            ok=False, error=f"Cannot reach Ollama at {root}: {exc}"
        )

def validate_and_list(
    provider: str,
    api_key: str,
    *,
    base_url: str | None = None,
) -> ModelCatalogResult:
    if provider == "openai":
        return validate_and_list_openai(api_key, base_url=base_url)
    if provider == "anthropic":
        return validate_and_list_anthropic(api_key)
    if provider == "openrouter":
        return validate_and_list_openrouter(api_key)
    if provider == "ollama":
        return list_ollama_models(base_url or "http://127.0.0.1:11434")
    return ModelCatalogResult(ok=False, error=f"No model catalog for provider={provider}")
