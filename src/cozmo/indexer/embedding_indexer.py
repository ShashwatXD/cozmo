"""Embedding indexer - embeds symbol definitions into vector store."""

from __future__ import annotations

from cozmo.domain.ports_rag import Embedder
from cozmo.domain.rag import Chunk
from cozmo.domain.symbols import FileSymbols, SymbolKind, SymbolNode
from cozmo.infra.rag.store import VectorStore

def _extract_body(text: str, node: SymbolNode) -> str:
    """Extract the source text for a symbol using its location."""
    lines = text.splitlines()
    start = max(0, node.location.start_line - 1)
    end = min(len(lines), node.location.end_line)
    return "\n".join(lines[start:end])

def _symbol_passage(node: SymbolNode, body: str) -> str:
    """Build a dense passage combining symbol metadata and body."""
    parts = [f"{node.kind.value} {node.qualified_name}"]
    if node.docstring:
        parts.append(node.docstring)
    parts.append(body)
    return "\n".join(parts)

class EmbeddingIndexer:
    """Embeds symbol definitions and stores them in VectorStore."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index_symbols(self, file_symbols: FileSymbols, file_text: str) -> int:
        """Embed symbols from a file and add to store. Returns chunk count."""
        nodes = self._collect_nodes(file_symbols.symbols)
        if not nodes:
            return 0

        passages: list[str] = []
        chunks: list[Chunk] = []
        for i, node in enumerate(nodes):
            body = _extract_body(file_text, node)
            passage = _symbol_passage(node, body)
            passages.append(passage)
            chunks.append(
                Chunk(
                    id=f"{file_symbols.path}::{node.qualified_name}::{i}",
                    path=file_symbols.path,
                    start_line=node.location.start_line,
                    text=passage,
                )
            )

        embeddings = self._embedder.embed_many(passages)
        for chunk, emb in zip(chunks, embeddings, strict=True):
            self._store.add(chunk, emb)

        return len(chunks)

    def _collect_nodes(self, symbols: tuple[SymbolNode, ...]) -> list[SymbolNode]:
        """Flatten symbols and their children into a list of indexable nodes."""
        nodes: list[SymbolNode] = []
        for sym in symbols:
            if sym.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS):
                nodes.append(sym)
            for child in sym.children:
                if child.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.CLASS):
                    nodes.append(child)
        return nodes
