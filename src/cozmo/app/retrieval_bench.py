"""
Retrieval benchmark: labeled queries -> file-level Recall@k / MRR / latency.

Compares the four retrieval strategies Cozmo can use over the same index:

* ``bm25``     - lexical only
* ``vector``   - embeddings only
* ``hybrid``   - BM25 + vector fused with Reciprocal Rank Fusion
* ``pipeline`` - hybrid recall -> lexical rerank (what ``semantic_search`` runs)

Relevance is judged at file level: a query is a hit at k if any expected path
appears among the first k *distinct* files returned.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cozmo.domain.ports_rag import Embedder
from cozmo.infra.rag.indexer import RepoIndexer
from cozmo.infra.rag.store import JsonVectorStore
from cozmo.search.hybrid_search import HybridSearch
from cozmo.search.pipeline import RetrievalPipeline

STRATEGIES = ("bm25", "vector", "hybrid", "pipeline")
KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class BenchQuery:
    id: str
    query: str
    expected: tuple[str, ...]
    kind: str = "conceptual"


@dataclass
class StrategyScore:
    strategy: str
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    by_kind: dict[str, dict[str, float]] = field(default_factory=dict)
    misses: list[str] = field(default_factory=list)


@dataclass
class BenchReport:
    embedder: str
    corpus_files: int
    corpus_chunks: int
    queries: int
    index_seconds: float
    embed_requests: int
    scores: list[StrategyScore]
    tool_latency_ms_p50: float = 0.0
    tool_latency_ms_p95: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        head = "| Strategy | R@1 | R@3 | R@5 | R@10 | MRR@10 | p50 ms | p95 ms |"
        sep = "|---|---|---|---|---|---|---|---|"
        rows = [
            f"| {s.strategy} | "
            + " | ".join(f"{s.recall[k]:.2f}" for k in KS)
            + f" | {s.mrr:.3f} | {s.latency_ms_p50:.2f} | {s.latency_ms_p95:.2f} |"
            for s in self.scores
        ]
        meta = (
            f"embedder=`{self.embedder}` · {self.queries} queries · "
            f"{self.corpus_files} files / {self.corpus_chunks} chunks · "
            f"index {self.index_seconds:.1f}s in {self.embed_requests} embed requests · "
            f"semantic_search tool p50 {self.tool_latency_ms_p50:.2f} ms "
            f"/ p95 {self.tool_latency_ms_p95:.2f} ms"
        )
        return "\n".join([meta, "", head, sep, *rows])


def load_queries(path: Path) -> list[BenchQuery]:
    out: list[BenchQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        out.append(
            BenchQuery(
                id=str(raw["id"]),
                query=str(raw["query"]),
                expected=tuple(raw["expected"]),
                kind=str(raw.get("kind", "conceptual")),
            )
        )
    return out


class CachingEmbedder:
    """
    Disk-cached embedder wrapper so repeated benchmark runs skip re-embedding.

    Also lets latency numbers measure retrieval, not the embedding round-trip.
    """

    def __init__(self, inner: Embedder, *, cache_path: Path | None, namespace: str) -> None:
        self._inner = inner
        self._path = cache_path
        self._ns = namespace
        self._cache: dict[str, list[float]] = {}
        if cache_path and cache_path.is_file():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._ns}\0{text}".encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            for t, emb in zip(missing, self._inner.embed_many(missing), strict=True):
                self._cache[self._key(t)] = emb
        return [self._cache[self._key(t)] for t in texts]

    def save(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._cache), encoding="utf-8")


def _distinct_paths(paths: Iterable[str], limit: int) -> list[str]:
    seen: list[str] = []
    for p in paths:
        if p not in seen:
            seen.append(p)
            if len(seen) >= limit:
                break
    return seen


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _score(
    strategy: str,
    queries: list[BenchQuery],
    rank_fn: Callable[[str], list[str]],
    *,
    repeats: int,
) -> StrategyScore:
    hits_at = {k: 0 for k in KS}
    rr_total = 0.0
    latencies: list[float] = []
    kind_hits: dict[str, list[float]] = {}
    misses: list[str] = []

    for q in queries:
        ranked: list[str] = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            ranked = _distinct_paths(rank_fn(q.query), 10)
            latencies.append((time.perf_counter() - t0) * 1000)
        expected = set(q.expected)
        first = next((i for i, p in enumerate(ranked) if p in expected), None)
        for k in KS:
            if first is not None and first < k:
                hits_at[k] += 1
        rr = 1.0 / (first + 1) if first is not None else 0.0
        rr_total += rr
        kind_hits.setdefault(q.kind, []).append(1.0 if first is not None and first < 5 else 0.0)
        if first is None or first >= 5:
            misses.append(q.id)

    n = max(1, len(queries))
    return StrategyScore(
        strategy=strategy,
        recall={k: hits_at[k] / n for k in KS},
        mrr=rr_total / n,
        latency_ms_p50=statistics.median(latencies) if latencies else 0.0,
        latency_ms_p95=_percentile(latencies, 95),
        by_kind={kind: {"recall@5": sum(v) / len(v)} for kind, v in kind_hits.items()},
        misses=misses,
    )


def _tool_latency(
    store: JsonVectorStore,
    embedder: Embedder,
    queries: list[BenchQuery],
    *,
    repeats: int,
) -> tuple[float, float]:
    """End-to-end latency of the agent's ``semantic_search`` tool handler."""
    from cozmo.infra.tools import build_default_registry
    from cozmo.infra.tools.permissions import WorkspaceGuard

    reg = build_default_registry(WorkspaceGuard(Path.cwd()), vector_store=store, embedder=embedder)
    tool = reg.get("semantic_search")
    assert tool is not None
    handler = tool.handler
    latencies: list[float] = []
    for q in queries:
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            handler({"query": q.query, "top_k": 10})
            latencies.append((time.perf_counter() - t0) * 1000)
    return statistics.median(latencies), _percentile(latencies, 95)


def run_retrieval_bench(
    corpus_root: Path,
    queries: list[BenchQuery],
    embedder: Embedder,
    *,
    embedder_name: str,
    repeats: int = 3,
) -> BenchReport:
    """Index *corpus_root* into an in-memory store and score every strategy."""
    store = JsonVectorStore()
    indexer = RepoIndexer(embedder, store)
    t0 = time.perf_counter()
    report = indexer.index_dir(corpus_root, incremental=False)
    index_seconds = time.perf_counter() - t0
    # Index writes rag_meta.json under corpus_root/.cozmo; benchmarks must not
    # leave that behind in the source tree.
    meta = corpus_root / ".cozmo" / "rag_meta.json"
    if meta.is_file():
        meta.unlink()
        if not any(meta.parent.iterdir()):
            meta.parent.rmdir()

    # Pre-embed queries so latency measures retrieval, not network round-trips.
    embedder.embed_many([q.query for q in queries])

    hybrid = HybridSearch(store, embedder)
    hybrid.build()
    pipeline = RetrievalPipeline(store, embedder, candidate_k=50, top_k=10)

    def bm25(q: str) -> list[str]:
        return [h.path for h in hybrid.lexical_search(q, top_k=50)]

    def vector(q: str) -> list[str]:
        return [h.chunk.path for h in store.search(embedder.embed(q), top_k=50)]

    def fused(q: str) -> list[str]:
        return [h.path for h in hybrid.search(q, top_k=50)]

    def piped(q: str) -> list[str]:
        return [h.path for h in pipeline.retrieve(q, top_k=10)]

    fns = {"bm25": bm25, "vector": vector, "hybrid": fused, "pipeline": piped}
    scores = [_score(name, queries, fns[name], repeats=repeats) for name in STRATEGIES]
    tool_p50, tool_p95 = _tool_latency(store, embedder, queries, repeats=repeats)
    return BenchReport(
        tool_latency_ms_p50=tool_p50,
        tool_latency_ms_p95=tool_p95,
        embedder=embedder_name,
        corpus_files=report.files_embedded,
        corpus_chunks=report.chunks,
        queries=len(queries),
        index_seconds=index_seconds,
        embed_requests=report.embed_requests,
        scores=scores,
    )
