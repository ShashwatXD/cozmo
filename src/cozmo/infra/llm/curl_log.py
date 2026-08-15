"""Format and print a curl for each LLM HTTP request (dev installs only)."""

from __future__ import annotations

import json
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower == "authorization" and value.lower().startswith("bearer "):
            out[key] = "Bearer $COZMO_API_KEY"
        elif lower in ("authorization", "x-api-key"):
            out[key] = "$COZMO_API_KEY"
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def is_dev_install() -> bool:
    try:
        import cozmo

        path = Path(cozmo.__file__).resolve()
    except Exception:
        return False

    parts_lower = {p.lower() for p in path.parts}
    return "site-packages" not in parts_lower and "dist-packages" not in parts_lower


def format_llm_curl(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    method: str = "POST",
) -> str:
    safe = _redact_headers(headers)
    parts = [
        "curl",
        "-sS",
        "-X",
        method,
        shlex.quote(url),
    ]
    for key, value in safe.items():
        header = f"{key}: {value}"
        if "$COZMO_API_KEY" in value:
            # Double-quote so the env var expands when pasted into a shell.
            parts.extend(["-H", '"' + header.replace('"', '\\"') + '"'])
        else:
            parts.extend(["-H", shlex.quote(header)])
    parts.extend(["-d", shlex.quote(json.dumps(body, ensure_ascii=False))])
    return " ".join(parts)


def log_llm_curl(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    method: str = "POST",
) -> None:
    if not is_dev_install():
        return
    curl = format_llm_curl(url=url, headers=headers, body=body, method=method)
    print(f"\n[cozmo llm curl]\n{curl}\n", file=sys.stderr, flush=True)
