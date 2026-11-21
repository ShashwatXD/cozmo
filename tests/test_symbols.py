"""AST tools: exact definitions with full ranges, real call sites, file outlines."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from cozmo.app.groundedness import contexts_from_tool_results
from cozmo.domain.tools import ToolCall
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.infra.tools.registry import ToolExecutor

CODE = '''\
import subprocess


def add(a, b):
    """Add two numbers."""
    return a - b


class Calculator:
    @staticmethod
    def run(items):
        total = 0
        for item in items:
            total = add(total, item)
        return total


def shell():
    return subprocess.run(["ls"])
'''


def _call(root: Path, tool: str, **args) -> str:
    ex = ToolExecutor(build_default_registry(WorkspaceGuard(root)))
    return ex.execute(ToolCall(id="1", name=tool, arguments=json.dumps(args))).content


def test_definition_returns_the_whole_function_with_its_range(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(CODE, encoding="utf-8")
    out = _call(tmp_path, "find_definition", name="Calculator.run")
    assert out.startswith("calc.py:10-15  method Calculator.run")
    assert "@staticmethod" in out and "return total" in out
    assert "No Python definition" in _call(tmp_path, "find_definition", name="missing")


def test_references_are_real_calls_with_their_enclosing_function(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(CODE, encoding="utf-8")
    out = _call(tmp_path, "find_references", name="add")
    assert out.splitlines()[0] == "1 call(s) to add()"
    assert "calc.py:14  in Calculator.run" in out
    run = _call(tmp_path, "find_references", name="run")
    assert "in shell" in run and "matched by name only" in run, "subprocess.run is flagged"


def test_outline_lists_symbols_and_rejects_other_languages(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(CODE, encoding="utf-8")
    out = _call(tmp_path, "outline", path="calc.py")
    assert out.splitlines()[0] == "calc.py (19 lines)"
    assert "def add(a, b)   lines 4-6" in out
    assert "    def run(items)   lines 10-15" in out
    assert "Python files only" in _call(tmp_path, "outline", path="app.ts")


def test_edits_are_picked_up_and_a_broken_file_keeps_its_last_good_parse(tmp_path: Path) -> None:
    f = tmp_path / "calc.py"
    f.write_text(CODE, encoding="utf-8")
    ex = ToolExecutor(build_default_registry(WorkspaceGuard(tmp_path)))

    def run(tool: str, **args) -> str:
        return ex.execute(ToolCall(id="1", name=tool, arguments=json.dumps(args))).content

    assert "add" in run("outline", path="calc.py")

    def write_later(text: str) -> None:
        f.write_text(text, encoding="utf-8")
        later = time.time() + 5
        os.utime(f, (later, later))

    write_later(CODE + "\n\ndef multiply(a, b):\n    return a * b\n")
    assert "calc.py:22-23  def multiply" in run("find_definition", name="multiply")
    write_later(CODE + "\n\ndef broken(:\n")
    out = run("outline", path="calc.py")
    assert "syntax error at line" in out and "def multiply" in out


def test_ast_tool_output_counts_as_evidence_for_groundedness(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text(CODE, encoding="utf-8")
    out = _call(tmp_path, "find_definition", name="add")
    [ctx] = contexts_from_tool_results("find_definition", '{"name": "add"}', out)
    assert (ctx.path, ctx.start_line, ctx.end_line) == ("calc.py", 4, 6)
    assert "return a - b" in ctx.content


def test_definition_shows_the_imports_it_uses(tmp_path: Path) -> None:
    (tmp_path / "math_utils.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text(
        "import os\nfrom math_utils import add\n\n\ndef total(items):\n"
        "    r = 0\n    for i in items:\n        r = add(r, i)\n    return r\n",
        encoding="utf-8",
    )
    out = _call(tmp_path, "find_definition", name="total")
    assert "# imports it uses: from math_utils import add" in out
    assert "import os" not in out, "only imports the body actually uses"
