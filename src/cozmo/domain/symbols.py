"""Symbol types for the Code Intelligence Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

class SymbolKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"

class Visibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"

@dataclass(frozen=True)
class Location:
    path: str
    start_line: int
    end_line: int

@dataclass(frozen=True)
class SymbolNode:
    """A single extracted symbol from source code."""

    name: str
    qualified_name: str
    kind: SymbolKind
    location: Location
    docstring: str = ""
    visibility: Visibility = Visibility.PUBLIC
    children: tuple[SymbolNode, ...] = ()
    module: str = ""
    alias: str = ""

@dataclass(frozen=True)
class ImportInfo:
    """A single import statement."""

    module: str
    names: tuple[str, ...] = ()
    alias: str = ""
    is_relative: bool = False
    level: int = 0
    location: Location | None = None

@dataclass(frozen=True)
class FileSymbols:
    """All symbols extracted from a single file."""

    path: str
    symbols: tuple[SymbolNode, ...] = ()
    imports: tuple[ImportInfo, ...] = ()
    language: str = "unknown"
