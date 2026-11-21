"""File + search + shell + git tools bound to a WorkspaceGuard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from cozmo.domain.tools import ToolSpec
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolRegistry
from cozmo.infra.tools.rg_search import search_repo as run_search_repo

# The model asks for more when it needs it, so defaults stay small: a full read of a
# large file would cost more tokens than the rest of the turn.
FULL_READ_MAX_CHARS = 12_000
HEAD_LINES = 150
DEFAULT_SEMANTIC_TOP_K = 5
MAX_SEMANTIC_TOP_K = 20


READ_SPEC = ToolSpec(
    name="read_file",
    description=(
        "Read a UTF-8 text file under the workspace. Small files come back whole. "
        "Large files return their first lines plus the total line count; ask for "
        "start_line/end_line or around_line to read the part you need."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path from workdir"},
            "start_line": {
                "type": "integer",
                "description": "Optional 1-based start line (inclusive)",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional 1-based end line (inclusive)",
            },
            "around_line": {
                "type": "integer",
                "description": "Optional center line; uses window lines each side",
            },
            "window": {
                "type": "integer",
                "description": "Lines before/after around_line (default 80)",
                "default": 80,
            },
        },
        "required": ["path"],
    },
)

WRITE_SPEC = ToolSpec(
    name="write_file",
    description="Write UTF-8 text to a file under the workspace (creates parents).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

SEARCH_SPEC = ToolSpec(
    name="search_repo",
    description=(
        "Exact / keyword search over the workspace (ripgrep when available). "
        "Returns path:line:snippet. Use for identifiers, errors, and literals. "
        "Prefer semantic_search for vague 'how does X work' questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "glob": {
                "type": "string",
                "description": "Optional glob or suffix filter, e.g. *.py or .py",
                "default": "",
            },
            "max_hits": {
                "type": "integer",
                "description": "Max hits to return (default 40)",
                "default": 40,
            },
        },
        "required": ["query"],
    },
)

SHELL_SPEC = ToolSpec(
    name="run_shell",
    description="Run a shell command inside the workspace (disabled unless allow_shell).",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
    },
)

GIT_STATUS_SPEC = ToolSpec(
    name="git_status",
    description="Run git status --short in the workspace.",
    parameters={"type": "object", "properties": {}},
)

GIT_DIFF_SPEC = ToolSpec(
    name="git_diff",
    description="Run git diff (optionally for one path) in the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Optional path"},
        },
    },
)

SEMANTIC_SPEC = ToolSpec(
    name="semantic_search",
    description=(
        "Hybrid retrieval over the indexed workspace: BM25+vector recall, "
        "lexical rerank, then surrounding context. Prefer for meaning / fuzzy "
        "questions; use search_repo for exact identifiers. Requires `cozmo index`."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {
                "type": "integer",
                "default": DEFAULT_SEMANTIC_TOP_K,
                "description": "Hits to return (default 5). Ask for more only if these miss.",
            },
        },
        "required": ["query"],
    },
)

HISTORY_SEARCH_SPEC = ToolSpec(
    name="search_history",
    description=(
        "Search prior Cozmo session turns (user/assistant/compact summaries) "
        "via hybrid retrieval. Use for past decisions or findings across sessions; "
        "use semantic_search for current repo code."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _read_file_ranged(path: Path, args: dict[str, Any], display_path: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    total = len(lines)
    if total == 0:
        return f"{display_path}: empty file"

    around = _as_int(args.get("around_line"))
    start = _as_int(args.get("start_line"))
    end = _as_int(args.get("end_line"))
    window = _as_int(args.get("window")) or 80
    window = max(1, min(window, 500))

    if around is not None:
        center = max(1, min(around, total))
        start = max(1, center - window)
        end = min(total, center + window)
    elif start is not None or end is not None:
        start = max(1, start or 1)
        end = min(total, end or total)
        if end < start:
            start, end = end, start
    else:
        # Small files come back whole. Large ones return the head and their size, so the
        # model can ask for the exact range instead of paying for the whole file.
        if len(text) <= FULL_READ_MAX_CHARS:
            return text
        head = "\n".join(lines[:HEAD_LINES])
        return (
            f"{display_path}:1-{min(HEAD_LINES, total)} (of {total})\n{head}\n"
            f"...[truncated: file has {total} lines / {len(text)} chars; "
            "pass start_line/end_line or around_line for the part you need]"
        )

    start = max(1, min(start, total))
    end = max(1, min(end, total))
    body = "\n".join(lines[start - 1 : end])
    return f"{display_path}:{start}-{end} (of {total})\n{body}"


def build_default_registry(
    guard: WorkspaceGuard,
    *,
    vector_store: Any | None = None,
    embedder: Any | None = None,
    sources: Any | None = None,
    history_rag: Any | None = None,
    shell_timeout_s: float = 60.0,
) -> ToolRegistry:
    """Register default workspace tools (read, write, search, git, optional RAG)."""
    reg = ToolRegistry()

    def read_file(args: dict[str, Any]) -> str:
        rel = args["path"]
        path = guard.resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {rel}")
        return _read_file_ranged(path, args, str(rel))

    def write_file(args: dict[str, Any]) -> str:
        guard.require_write()
        path = guard.resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {args['path']}"

    def search_repo(args: dict[str, Any]) -> str:
        query = str(args.get("query") or "")
        glob = str(args.get("glob") or "")
        max_hits = _as_int(args.get("max_hits")) or 40
        max_hits = max(1, min(max_hits, 200))
        return run_search_repo(guard.workdir, query, glob=glob, max_hits=max_hits)

    def run_shell(args: dict[str, Any]) -> str:
        guard.require_shell()
        command = args["command"]
        proc = subprocess.run(
            command,
            shell=True,
            cwd=guard.workdir,
            capture_output=True,
            text=True,
            timeout=shell_timeout_s,
            check=False,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"exit={proc.returncode}\n{out}"

    def git_status(_: dict[str, Any]) -> str:
        return _git(guard.workdir, ["status", "--short"])

    def git_diff(args: dict[str, Any]) -> str:
        cmd = ["diff"]
        if args.get("path"):
            guard.resolve(args["path"])
            cmd.append(args["path"])
        return _git(guard.workdir, cmd)

    def semantic_search(args: dict[str, Any]) -> str:
        if vector_store is None or embedder is None or len(vector_store) == 0:
            return "No RAG index loaded. Run: cozmo index -w <workdir> then retry semantic_search."
        top_k = _as_int(args.get("top_k")) or DEFAULT_SEMANTIC_TOP_K
        top_k = max(1, min(top_k, MAX_SEMANTIC_TOP_K))
        from cozmo.search.pipeline import RetrievalPipeline

        pipeline = RetrievalPipeline(
            vector_store,
            embedder,
            sources=sources,
            candidate_k=50,
            top_k=top_k,
        )
        hits = pipeline.retrieve(args["query"], top_k=top_k)
        if not hits:
            return "No semantic hits."
        stale = getattr(sources, "changed_since_index", None)
        lines: list[str] = []
        for h in hits:
            preview = h.text[:800]
            note = ""
            if stale is not None and stale(h.path):
                note = " [file changed since indexing: line numbers may have moved, re-read it]"
            lines.append(
                f"score={h.score:.3f} {h.path}:{h.start_line}-{h.end_line}{note}\n{preview}"
            )
        return "\n---\n".join(lines)

    def search_history(args: dict[str, Any]) -> str:
        if history_rag is None:
            return (
                "History RAG unavailable. Enable history_enabled and history_rag "
                "in config (defaults on)."
            )
        top_k = int(args.get("top_k") or 5)
        return history_rag.search(str(args.get("query") or ""), top_k=top_k)

    reg.register(READ_SPEC, read_file)
    reg.register(WRITE_SPEC, write_file)
    reg.register(SEARCH_SPEC, search_repo)
    reg.register(SEMANTIC_SPEC, semantic_search)
    reg.register(HISTORY_SEARCH_SPEC, search_history)
    reg.register(SHELL_SPEC, run_shell)
    reg.register(GIT_STATUS_SPEC, git_status)
    reg.register(GIT_DIFF_SPEC, git_diff)

    return reg


def _git(cwd: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and not out:
        raise RuntimeError(f"git {' '.join(args)} failed ({proc.returncode})")
    return out.strip() or "(clean)"
