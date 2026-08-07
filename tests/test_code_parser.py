"""Tests for the code parser / symbol extraction engine."""

from __future__ import annotations

import textwrap

import pytest

from cozmo.domain.symbols import (
    FileSymbols,
    SymbolKind,
    Visibility,
)
from cozmo.parser.ast_parser import PythonASTParser
from cozmo.parser._regex_fallback import parse_with_regex
from cozmo.parser.symbol_extractor import SymbolExtractor


# ---- fixtures ----------------------------------------------------------

SAMPLE_PY = textwrap.dedent("""\
    \"\"\"Module docstring.\"\"\"
    import os
    from pathlib import Path

    MAX_SIZE = 1024

    class MyClass:
        \"\"\"A sample class.\"\"\"
        class_var = 42

        def public_method(self):
            pass

        def _private_method(self):
            pass

        class _Inner:
            pass

    def top_level():
        \"\"\"Top-level function.\"\"\"
        pass

    def _helper():
        pass
""")

SAMPLE_JS = textwrap.dedent("""\
    import { foo } from 'bar';
    export class Widget {}
    export function render() {}
    const API_URL = "http://example.com";
    const handler = async (req) => {};
""")

SAMPLE_RUST = textwrap.dedent("""\
    use std::io::Read;
    pub struct Config {}
    pub fn load() {}
    fn _internal() {}
""")


# ---- PythonASTParser ---------------------------------------------------

class TestPythonASTParser:
    parser = PythonASTParser()

    def test_classes_and_functions(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        names = [s.name for s in fs.symbols]
        assert "MyClass" in names
        assert "top_level" in names
        assert "_helper" in names

    def test_class_methods_are_children(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        cls = next(s for s in fs.symbols if s.name == "MyClass")
        child_names = [c.name for c in cls.children]
        assert "public_method" in child_names
        assert "_private_method" in child_names

    def test_nested_class(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        cls = next(s for s in fs.symbols if s.name == "MyClass")
        inner = [c for c in cls.children if c.kind == SymbolKind.CLASS]
        assert len(inner) == 1
        assert inner[0].name == "_Inner"
        assert inner[0].qualified_name == "MyClass._Inner"

    def test_visibility(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        helper = next(s for s in fs.symbols if s.name == "_helper")
        assert helper.visibility == Visibility.PRIVATE
        top = next(s for s in fs.symbols if s.name == "top_level")
        assert top.visibility == Visibility.PUBLIC

    def test_docstrings(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        cls = next(s for s in fs.symbols if s.name == "MyClass")
        assert cls.docstring == "A sample class."
        func = next(s for s in fs.symbols if s.name == "top_level")
        assert func.docstring == "Top-level function."

    def test_imports(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        assert len(fs.imports) == 2
        modules = {i.module for i in fs.imports}
        assert "os" in modules
        assert "pathlib" in modules

    def test_from_import_names(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        pathlib_imp = next(i for i in fs.imports if i.module == "pathlib")
        assert "Path" in pathlib_imp.names

    def test_module_variable(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        var = next(s for s in fs.symbols if s.name == "MAX_SIZE")
        assert var.kind == SymbolKind.VARIABLE

    def test_method_kind(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        cls = next(s for s in fs.symbols if s.name == "MyClass")
        method = next(c for c in cls.children if c.name == "public_method")
        assert method.kind == SymbolKind.METHOD

    def test_language_field(self) -> None:
        fs = self.parser.parse(SAMPLE_PY, "sample.py")
        assert fs.language == "python"


# ---- Regex fallback ----------------------------------------------------

class TestRegexFallback:
    def test_js_class_and_function(self) -> None:
        fs = parse_with_regex(SAMPLE_JS, "widget.js", "javascript")
        names = [s.name for s in fs.symbols]
        assert "Widget" in names
        assert "render" in names

    def test_js_import(self) -> None:
        fs = parse_with_regex(SAMPLE_JS, "widget.js", "javascript")
        assert any(i.module == "bar" for i in fs.imports)

    def test_js_arrow_function(self) -> None:
        fs = parse_with_regex(SAMPLE_JS, "widget.js", "javascript")
        names = [s.name for s in fs.symbols]
        assert "handler" in names

    def test_rust_symbols(self) -> None:
        fs = parse_with_regex(SAMPLE_RUST, "lib.rs", "rust")
        names = [s.name for s in fs.symbols]
        assert "Config" in names
        assert "load" in names
        assert "_internal" in names

    def test_rust_use_import(self) -> None:
        fs = parse_with_regex(SAMPLE_RUST, "lib.rs", "rust")
        assert len(fs.imports) >= 1

    def test_unknown_language(self) -> None:
        fs = parse_with_regex("whatever", "x.txt", "unknown")
        assert fs.symbols == ()
        assert fs.language == "unknown"


# ---- SymbolExtractor ---------------------------------------------------

class TestSymbolExtractor:
    extractor = SymbolExtractor()

    def test_dispatches_python(self) -> None:
        fs = self.extractor.extract("app.py", SAMPLE_PY)
        assert fs.language == "python"
        assert any(s.name == "MyClass" for s in fs.symbols)

    def test_dispatches_js(self) -> None:
        fs = self.extractor.extract("widget.js", SAMPLE_JS)
        assert fs.language == "javascript"
        assert any(s.name == "Widget" for s in fs.symbols)

    def test_explicit_language_override(self) -> None:
        fs = self.extractor.extract("noext", SAMPLE_PY, language="python")
        assert fs.language == "python"

    def test_unknown_extension(self) -> None:
        fs = self.extractor.extract("data.csv", "a,b,c")
        assert fs.language == "unknown"
