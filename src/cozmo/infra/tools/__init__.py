"""File + search + shell + git tools bound to a WorkspaceGuard."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from cozmo.domain.tools import ToolSpec
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolRegistry

READ_SPEC = ToolSpec(
    name="read_file",
    description="Read a UTF-8 text file under the workspace.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path from workdir"},
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
    description="Search for a text pattern in files under the workspace (simple scan).",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "glob": {
                "type": "string",
                "description": "Optional suffix filter, e.g. .py",
                "default": "",
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
        "Semantic / RAG search over the indexed workspace. "
        "Prefer this to find code by meaning; use search_repo for exact text. "
        "Requires `cozmo index` first."
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


def build_default_registry(
    guard: WorkspaceGuard,
    *,
    vector_store: Any | None = None,
    embedder: Any | None = None,
    code_index: Any | None = None,
    sources: dict[str, str] | None = None,
) -> ToolRegistry:
    """Register default workspace tools (read, write, search, git, optional RAG)."""
    reg = ToolRegistry()

    def read_file(args: dict[str, Any]) -> str:
        path = guard.resolve(args["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {args['path']}")
        text = path.read_text(encoding="utf-8")
        # Cap huge files so we don't blow context
        if len(text) > 50_000:
            return text[:50_000] + "\n...[truncated]"
        return text

    def write_file(args: dict[str, Any]) -> str:
        guard.require_write()
        path = guard.resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"Wrote {len(args['content'])} chars to {args['path']}"

    def search_repo(args: dict[str, Any]) -> str:
        query = args["query"]
        suffix = args.get("glob") or ""
        hits: list[str] = []
        for file in guard.workdir.rglob("*"):
            if not file.is_file():
                continue
            if suffix and not file.name.endswith(suffix):
                continue
            # skip venv / git junk
            parts = set(file.parts)
            if ".venv" in parts or "node_modules" in parts or ".git" in parts:
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if query in line:
                    rel = file.relative_to(guard.workdir)
                    hits.append(f"{rel}:{i}:{line.strip()[:200]}")
                    if len(hits) >= 40:
                        return "\n".join(hits)
        return "\n".join(hits) if hits else "No matches."

    def run_shell(args: dict[str, Any]) -> str:
        guard.require_shell()
        command = args["command"]
        proc = subprocess.run(
            command,
            shell=True,
            cwd=guard.workdir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"exit={proc.returncode}\n{out[:20_000]}"

    def git_status(_: dict[str, Any]) -> str:
        return _git(guard.workdir, ["status", "--short"])

    def git_diff(args: dict[str, Any]) -> str:
        cmd = ["diff"]
        if args.get("path"):
            # still sandbox the path
            guard.resolve(args["path"])
            cmd.append(args["path"])
        return _git(guard.workdir, cmd)

    def semantic_search(args: dict[str, Any]) -> str:
        if vector_store is None or embedder is None or len(vector_store) == 0:
            return (
                "No RAG index loaded. Run: cozmo index -w <workdir> "
                "then retry semantic_search."
            )
        top_k = int(args.get("top_k") or 5)
        q = embedder.embed(args["query"])
        hits = vector_store.search(q, top_k=top_k)
        if not hits:
            return "No semantic hits."
        lines: list[str] = []
        for h in hits:
            preview = h.chunk.text[:300].replace("\n", " ")
            lines.append(
                f"score={h.score:.3f} {h.chunk.path}:{h.chunk.start_line}\n{preview}"
            )
        return "\n---\n".join(lines)

    reg.register(READ_SPEC, read_file)
    reg.register(WRITE_SPEC, write_file)
    reg.register(SEARCH_SPEC, search_repo)
    reg.register(SEMANTIC_SPEC, semantic_search)
    reg.register(SHELL_SPEC, run_shell)
    reg.register(GIT_STATUS_SPEC, git_status)
    reg.register(GIT_DIFF_SPEC, git_diff)

    # Code intelligence tools (optional - requires a built CodeIndex)
    if code_index is not None:
        from cozmo.infra.tools.code_intel import register_code_intel_tools

        register_code_intel_tools(
            reg,
            code_index,
            sources=sources or {},
            vector_store=vector_store,
            embedder=embedder,
        )

    return reg


def _git(cwd: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and not out:
        raise RuntimeError(f"git {' '.join(args)} failed ({proc.returncode})")
    return out.strip() or "(clean)"
