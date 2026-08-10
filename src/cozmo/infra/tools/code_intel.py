"""Code Intelligence tools - symbol search, references, codebase graphs."""

from __future__ import annotations

from typing import Any

from cozmo.domain.tools import ToolSpec
from cozmo.infra.tools.registry import ToolRegistry


SYMBOL_SEARCH_SPEC = ToolSpec(
    name="symbol_search",
    description=(
        "Search the code index for symbol definitions (functions, classes, variables). "
        "Returns matching symbols with file location and kind."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Symbol name or substring to search for"},
            "kind": {
                "type": "string",
                "description": "Optional filter: function, class, method, variable, module",
            },
        },
        "required": ["query"],
    },
)

FIND_REFERENCES_SPEC = ToolSpec(
    name="find_references",
    description=(
        "Find all references (usages, imports, calls) of a symbol across the codebase."
    ),
    parameters={
        "type": "object",
        "properties": {
            "symbol_name": {
                "type": "string",
                "description": "Exact symbol name to find references for",
            },
        },
        "required": ["symbol_name"],
    },
)

GET_CODEBASE_GRAPH_SPEC = ToolSpec(
    name="get_codebase_graph",
    description=(
        "Retrieve a codebase graph: dependency, call, or import graph. "
        "Optionally scoped to a specific file or directory path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "graph_type": {
                "type": "string",
                "enum": ["dependency", "call", "import"],
                "description": "Type of graph to retrieve",
            },
            "path": {
                "type": "string",
                "description": "Optional file or directory to scope the graph to",
            },
        },
        "required": ["graph_type"],
    },
)


def _make_symbol_search_handler(
    code_index: Any,
    sources: dict[str, str],
) -> Any:
    """Return a handler that searches symbols in the code index."""

    def symbol_search(args: dict[str, Any]) -> str:
        query = args["query"]
        kind = args.get("kind")

        try:
            from cozmo.search.symbol_search import SymbolSearch

            searcher = SymbolSearch(code_index)
            results = searcher.search(query, kind=kind)
        except ImportError:
            results = _fallback_symbol_search(code_index, query, kind)

        if not results:
            return f"No symbols matching '{query}' found."

        lines: list[str] = []
        for sym in results[:20]:
            loc = f"{sym.location.path}:{sym.location.line}" if hasattr(sym, "location") else str(sym)
            lines.append(f"{sym.kind.value if hasattr(sym.kind, 'value') else sym.kind} {sym.name} @ {loc}")
        return "\n".join(lines)

    return symbol_search

def _fallback_symbol_search(code_index: Any, query: str, kind: str | None) -> list[Any]:
    """Simple fallback when search module isn't available yet."""
    matches = []
    for file_symbols in code_index.files.values():
        for sym in file_symbols.symbols:
            if query.lower() in sym.name.lower():
                if kind is None or (hasattr(sym.kind, "value") and sym.kind.value == kind):
                    matches.append(sym)
    return matches

def _make_find_references_handler(
    code_index: Any,
    sources: dict[str, str],
) -> Any:
    """Return a handler that finds references to a symbol."""

    def find_references(args: dict[str, Any]) -> str:
        symbol_name = args["symbol_name"]

        try:
            from cozmo.search.reference_search import ReferenceSearch

            searcher = ReferenceSearch(code_index, sources)
            refs = searcher.find(symbol_name)
        except ImportError:
            refs = _fallback_reference_search(sources, symbol_name)

        if not refs:
            return f"No references to '{symbol_name}' found."

        lines: list[str] = []
        for ref in refs[:30]:
            if isinstance(ref, dict):
                lines.append(f"{ref.get('path', '?')}:{ref.get('line', '?')} {ref.get('context', '')}")
            else:
                loc = f"{ref.location.path}:{ref.location.line}" if hasattr(ref, "location") else str(ref)
                lines.append(loc)
        return "\n".join(lines)

    return find_references

def _fallback_reference_search(sources: dict[str, str], symbol_name: str) -> list[dict[str, Any]]:
    """Simple text-based reference search when search module isn't available."""
    refs: list[dict[str, Any]] = []
    for path, content in sources.items():
        for i, line in enumerate(content.splitlines(), start=1):
            if symbol_name in line:
                refs.append({"path": path, "line": i, "context": line.strip()[:120]})
                if len(refs) >= 30:
                    return refs
    return refs

def _make_get_codebase_graph_handler(
    code_index: Any,
    sources: dict[str, str],
) -> Any:
    """Return a handler that retrieves codebase graph data."""

    def get_codebase_graph(args: dict[str, Any]) -> str:
        graph_type = args["graph_type"]
        scope_path = args.get("path")

        graph_builders = {
            "dependency": ("cozmo.graphs.dependency_graph", "DependencyGraph"),
            "call": ("cozmo.graphs.call_graph", "CallGraph"),
            "import": ("cozmo.graphs.import_graph", "ImportGraph"),
        }

        if graph_type not in graph_builders:
            return f"Unknown graph type: {graph_type}. Use: dependency, call, or import."

        module_path, class_name = graph_builders[graph_type]

        try:
            import importlib

            mod = importlib.import_module(module_path)
            graph_cls = getattr(mod, class_name)
            graph = graph_cls.from_index(code_index)

            if scope_path:
                summary = graph.subgraph(scope_path).summary()
            else:
                summary = graph.summary()
            return summary
        except ImportError:
            return _fallback_graph(code_index, graph_type, scope_path)
        except Exception as exc:
            return f"Graph error: {exc}"

    return get_codebase_graph

def _fallback_graph(code_index: Any, graph_type: str, scope_path: str | None) -> str:
    """Minimal graph info from the code index when graph modules aren't available."""
    if graph_type == "import":
        lines = [f"Import graph ({len(code_index.files)} files):"]
        for path, file_syms in list(code_index.files.items())[:20]:
            if scope_path and scope_path not in path:
                continue
            imports = [imp.module for imp in getattr(file_syms, "imports", [])]
            if imports:
                lines.append(f"  {path} -> {', '.join(imports[:10])}")
        return "\n".join(lines) if len(lines) > 1 else "No import data available."
    return f"{graph_type} graph: {len(code_index.files)} files indexed. Install graph modules for full support."


def register_code_intel_tools(
    registry: ToolRegistry,
    code_index: Any,
    sources: dict[str, str],
    vector_store: Any = None,
    embedder: Any = None,
) -> None:
    """
    Register code intelligence tools (symbol_search, find_references,
    get_codebase_graph) into an existing ToolRegistry.

    Args:
        registry: The ToolRegistry to add tools to.
        code_index: A CodeIndex instance with parsed symbol data.
        sources: Mapping of file path -> source text.
        vector_store: Optional VectorStore for hybrid search.
        embedder: Optional Embedder for hybrid search.
    """
    registry.register(
        SYMBOL_SEARCH_SPEC,
        _make_symbol_search_handler(code_index, sources),
    )
    registry.register(
        FIND_REFERENCES_SPEC,
        _make_find_references_handler(code_index, sources),
    )
    registry.register(
        GET_CODEBASE_GRAPH_SPEC,
        _make_get_codebase_graph_handler(code_index, sources),
    )
