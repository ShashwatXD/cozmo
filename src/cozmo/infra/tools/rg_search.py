"""Ripgrep-backed workspace search with Python fallback."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from cozmo.infra.workspace.ignore import IgnoreFilter

_DEFAULT_MAX_HITS = 40
_LINE_SNIPPET = 200


def rg_available() -> bool:
    return shutil.which("rg") is not None


def _flex_regex(query: str) -> str:
    """Allow optional whitespace between characters (e.g. a-b matches a - b)."""
    return r"\s*".join(re.escape(ch) for ch in query)


def _needs_flex(query: str) -> bool:
    if not query or " " in query:
        return False
    return bool(re.search(r"[^A-Za-z0-9_]", query))


def _line_matches(query: str, line: str) -> bool:
    if not query:
        return False
    if query in line:
        return True
    compact_q = re.sub(r"\s+", "", query)
    if not compact_q:
        return False
    return compact_q in re.sub(r"\s+", "", line)


def _normalize_glob(glob: str) -> str | None:
    g = (glob or "").strip()
    if not g:
        return None
    if g.startswith("*"):
        return g
    if g.startswith("."):
        return f"*{g}"
    return g


def search_with_rg(
    root: Path,
    query: str,
    *,
    glob: str = "",
    max_hits: int = _DEFAULT_MAX_HITS,
) -> list[str] | None:
    """
    Run ripgrep. Returns hit lines ``path:line:snippet``, or None if rg missing/failed hard.
    Empty list means no matches.
    """
    if not query or not rg_available():
        return None

    root = root.resolve()
    glob_pat = _normalize_glob(glob)
    hits: list[str] = []
    seen: set[str] = set()

    def _run(pattern: str, *, fixed: bool) -> None:
        nonlocal hits
        if len(hits) >= max_hits:
            return
        cmd = [
            "rg",
            "--line-number",
            "--with-filename",
            "--no-heading",
            "--color",
            "never",
            "--max-columns",
            "400",
            "--max-columns-preview",
        ]
        if fixed:
            cmd.append("--fixed-strings")
        if glob_pat:
            cmd.extend(["--glob", glob_pat])
        # Extra safety skips (rg already honors .gitignore).
        for skip in (".git", "node_modules", ".venv", "venv", ".cozmo"):
            cmd.extend(["--glob", f"!**/{skip}/**"])
        cmd.extend(["--", pattern, str(root)])
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        # 0 = matches, 1 = no matches, 2 = error
        if proc.returncode not in (0, 1):
            return
        for raw in (proc.stdout or "").splitlines():
            if len(hits) >= max_hits:
                break
            parsed = _parse_rg_line(raw, root)
            if parsed is None or parsed in seen:
                continue
            seen.add(parsed)
            hits.append(parsed)

    _run(query, fixed=True)
    if _needs_flex(query) and len(hits) < max_hits:
        _run(_flex_regex(query), fixed=False)
    return hits


def _parse_rg_line(raw: str, root: Path) -> str | None:
    # path:line:text  (path may contain colons on Windows rarely; split carefully)
    m = re.match(r"^(.*):(\d+):(.*)$", raw)
    if not m:
        return None
    path_s, line_s, text = m.group(1), m.group(2), m.group(3)
    try:
        path = Path(path_s)
        if path.is_absolute():
            rel = path.resolve().relative_to(root).as_posix()
        else:
            rel = path.as_posix()
    except ValueError:
        rel = path_s
    snippet = text.strip()[:_LINE_SNIPPET]
    return f"{rel}:{line_s}:{snippet}"


def search_fallback(
    root: Path,
    query: str,
    *,
    glob: str = "",
    max_hits: int = _DEFAULT_MAX_HITS,
    ignore: IgnoreFilter | None = None,
) -> list[str]:
    """Python walk with IgnoreFilter + whitespace-insensitive match."""
    root = root.resolve()
    filt = ignore or IgnoreFilter(root)
    suffix = (glob or "").strip()
    if suffix.startswith("*"):
        suffix = suffix[1:]
    hits: list[str] = []
    for file in filt.iter_files():
        if suffix and not file.name.endswith(suffix) and not str(file).endswith(suffix):
            continue
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = file.relative_to(root).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            if _line_matches(query, line):
                hits.append(f"{rel}:{i}:{line.strip()[:_LINE_SNIPPET]}")
                if len(hits) >= max_hits:
                    return hits
    return hits


def search_repo(
    root: Path,
    query: str,
    *,
    glob: str = "",
    max_hits: int = _DEFAULT_MAX_HITS,
) -> str:
    """Public entry: prefer ripgrep, fall back to Python walker."""
    query = query or ""
    if not query.strip():
        return "No matches."
    rg_hits = search_with_rg(root, query, glob=glob, max_hits=max_hits)
    if rg_hits is not None:
        return "\n".join(rg_hits) if rg_hits else "No matches."
    hits = search_fallback(root, query, glob=glob, max_hits=max_hits)
    return "\n".join(hits) if hits else "No matches."
