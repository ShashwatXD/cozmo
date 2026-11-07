"""Regression tests for the engineering-hardening pass (bugs + performance work)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import httpx
import pytest

from cozmo.app.history import SessionHistory
from cozmo.app.model_router import ModelRouter
from cozmo.app.team import TeamService
from cozmo.domain.completion import CompletionResult, TransientLLMError, Usage
from cozmo.domain.events import EventKind
from cozmo.domain.guardrails import AgentPolicy
from cozmo.domain.messages import Message, Role
from cozmo.domain.rag import Chunk, RetrievalCandidate
from cozmo.domain.roles import ModelRole
from cozmo.domain.tools import ToolSpec
from cozmo.infra.history.null import NullEventStore
from cozmo.infra.llm.anthropic_client import AnthropicClient
from cozmo.infra.llm.retrying import RetryingLLMClient
from cozmo.infra.llm.stub import StubLLMClient
from cozmo.infra.rag.embedder import StubEmbedder
from cozmo.infra.rag.indexer import RepoIndexer
from cozmo.infra.rag.store import JsonVectorStore, cosine
from cozmo.infra.tools import build_default_registry
from cozmo.infra.tools.permissions import WorkspaceGuard
from cozmo.search import hybrid_search
from cozmo.search.hybrid_search import HybridSearch, _tokenize
from cozmo.search.rerank import LexicalReranker

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_repo"


# --- session history ---------------------------------------------------------


class _ListStore(NullEventStore):
    def __init__(self) -> None:
        self.events: list = []

    def append(self, event) -> None:
        self.events.append(event)


def test_history_team_event_does_not_collide_with_kind() -> None:
    """history.team() passed kind="team" into emit(kind, **data) -> TypeError."""
    store = _ListStore()
    event = SessionHistory(store).team("find add()", team_id="t1", workers=["researcher"])
    assert event.kind is EventKind.SUBAGENT
    assert event.data["kind"] == "team"
    assert store.events == [event]


def test_team_run_with_history_enabled() -> None:
    class Orchestrator:
        def complete(self, messages, *, temperature=0.2, json_mode=False, tools=None):
            content = (
                json.dumps({"plan": "p", "orders": [{"agentId": "researcher", "goal": "g"}]})
                if json_mode
                else "merged"
            )
            return CompletionResult(content=content, usage=Usage(), finish_reason="stop")

        def stream(self, messages, *, temperature=0.2):
            yield "merged"

    store = _ListStore()
    svc = TeamService(
        models=ModelRouter(
            {ModelRole.ORCHESTRATOR: Orchestrator(), ModelRole.WORKER: StubLLMClient()}
        ),
        parent_registry=build_default_registry(WorkspaceGuard(FIXTURE)),
        policy=AgentPolicy(max_subagent_depth=1, max_subagent_steps=2),
        history=SessionHistory(store),
    )
    out = json.loads(svc.run("Where is add()?", worker_names=["researcher"], parallel=False))
    assert out["ok"] is True
    assert any(e.data.get("kind") == "team" for e in store.events)


# --- vector store ------------------------------------------------------------


def _random_store(n: int = 200, dim: int = 32, seed: int = 7) -> JsonVectorStore:
    rng = random.Random(seed)
    store = JsonVectorStore()
    for i in range(n):
        store.add(
            Chunk(id=f"f{i % 17}.py::{i}", path=f"f{i % 17}.py", start_line=i, text=f"chunk {i}"),
            [rng.uniform(-1, 1) for _ in range(dim)],
        )
    return store


def test_vectorized_search_matches_reference_cosine() -> None:
    store = _random_store()
    rng = random.Random(1)
    for _ in range(10):
        q = [rng.uniform(-1, 1) for _ in range(32)]
        got = store.search(q, top_k=15)
        ref = sorted(store.items(), key=lambda it: cosine(q, it[1]), reverse=True)[:15]
        assert [h.chunk.id for h in got] == [c.id for c, _ in ref]
        for h, (_, emb) in zip(got, ref, strict=True):
            assert math.isclose(h.score, cosine(q, emb), abs_tol=1e-5)


def test_search_cache_invalidated_on_mutation() -> None:
    store = JsonVectorStore()
    store.add(Chunk("a::0", "a.py", 1, "a"), [1.0, 0.0])
    assert store.search([1.0, 0.0], top_k=1)[0].chunk.path == "a.py"
    v0 = store.version
    store.add(Chunk("b::0", "b.py", 1, "b"), [0.0, 1.0])
    assert store.version > v0
    assert store.search([0.0, 1.0], top_k=1)[0].chunk.path == "b.py"
    store.drop_paths({"b.py"})
    assert store.search([0.0, 1.0], top_k=1)[0].chunk.path == "a.py"


def test_search_handles_mixed_dimensions_and_empty() -> None:
    store = JsonVectorStore()
    assert store.search([1.0], top_k=3) == []
    store.add(Chunk("a::0", "a.py", 1, "a"), [1.0, 0.0])
    store.add(Chunk("b::0", "b.py", 1, "b"), [1.0, 0.0, 0.0])
    assert len(store.search([1.0, 0.0], top_k=2)) == 2


def test_chroma_drop_paths_removes_stale_chunks(tmp_path: Path) -> None:
    """Chroma lacked drop_paths, so incremental re-index kept deleted files forever."""
    from cozmo.infra.rag.chroma_store import ChromaVectorStore

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (repo / "gone.py").write_text("def gone_function_marker():\n    return 2\n", encoding="utf-8")

    embedder = StubEmbedder()
    store = ChromaVectorStore(tmp_path / "chroma")
    indexer = RepoIndexer(embedder, store)
    indexer.index_dir(repo)
    assert {"keep.py", "gone.py"} <= {c.path for c, _ in store.items()}

    (repo / "gone.py").unlink()
    report = indexer.index_dir(repo, incremental=True)
    assert report.files_removed >= 1
    assert all(h.chunk.path != "gone.py" for h in store.search(embedder.embed("gone"), top_k=10))
    assert "gone.py" not in {c.path for c, _ in store.items()}


# --- indexer -----------------------------------------------------------------


class _CountingEmbedder(StubEmbedder):
    def __init__(self, poison: str | None = None) -> None:
        super().__init__()
        self.requests = 0
        self.poison = poison

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        self.requests += 1
        if self.poison and any(self.poison in t for t in texts):
            raise RuntimeError("HTTP 402 payment required")
        return super().embed_many(texts)


def _small_repo(root: Path, files: int) -> None:
    for i in range(files):
        (root / f"mod_{i}.py").write_text(f"def fn_{i}():\n    return {i}\n", encoding="utf-8")


def test_indexer_batches_across_files(tmp_path: Path) -> None:
    _small_repo(tmp_path, 40)
    emb = _CountingEmbedder()
    report = RepoIndexer(emb, JsonVectorStore()).index_dir(tmp_path)
    assert report.files_embedded == 40
    # 40 one-chunk files fit in a single cross-file batch (was 40 requests).
    assert emb.requests == report.embed_requests == 1


def test_indexer_isolates_failing_file(tmp_path: Path) -> None:
    _small_repo(tmp_path, 10)
    (tmp_path / "bad.py").write_text("POISON = True\n", encoding="utf-8")
    store = JsonVectorStore()
    report = RepoIndexer(_CountingEmbedder(poison="POISON"), store).index_dir(tmp_path)
    assert report.partial is True
    assert report.files_embedded == 10
    assert any("bad.py" in e for e in report.errors)
    assert "bad.py" not in store.paths()
    meta = json.loads((tmp_path / ".cozmo" / "rag_meta.json").read_text(encoding="utf-8"))
    assert "bad.py" not in meta["files"]  # retried on the next run


# --- lexical search ----------------------------------------------------------


def test_tokenize_splits_identifiers() -> None:
    toks = _tokenize("WorkspaceGuard detect_stuck HTTPServer")
    assert {"workspaceguard", "workspace", "guard", "detect_stuck", "detect", "stuck"} <= set(toks)
    assert "http" in toks and "server" in toks


def test_bm25_index_is_shared_and_invalidated() -> None:
    store = JsonVectorStore()
    emb = StubEmbedder()
    store.add(Chunk("a::0", "a.py", 1, "alpha beta"), emb.embed("alpha beta"))

    first = HybridSearch(store, emb)
    first.build()
    second = HybridSearch(store, emb)
    second.build()
    assert first._index is second._index  # no rebuild per pipeline / tool call

    store.add(Chunk("b::0", "b.py", 1, "gamma delta"), emb.embed("gamma delta"))
    hits = second.lexical_search("gamma", top_k=5)
    assert [h.path for h in hits] == ["b.py"]
    assert hybrid_search._INDEX_CACHE[store].size == 2


def test_hybrid_ordering_is_deterministic() -> None:
    store = _random_store(n=60)
    hs = HybridSearch(store, StubEmbedder(dim=32))
    a = [h.path for h in hs.search("chunk", top_k=20)]
    b = [h.path for h in hs.search("chunk", top_k=20)]
    assert a == b


# --- reranker ----------------------------------------------------------------


def _cand(path: str, text: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(path=path, start_line=1, text=text, score=score)


def test_reranker_keeps_recall_order_without_lexical_evidence() -> None:
    """Old reranker scaled the prior by 0.15, so stopword overlap reordered everything."""
    cands = [
        _cand("guard.py", "class Sandbox: ...", 0.033),
        _cand("noise.py", "how does the agent do this when it is the same", 0.016),
    ]
    out = LexicalReranker().rerank("how does the agent stay inside the folder", cands, top_k=2)
    assert [c.path for c in out] == ["guard.py", "noise.py"]


def test_reranker_promotes_exact_identifier_match() -> None:
    cands = [
        _cand("a.py", "unrelated text", 0.033),
        _cand("b.py", "def detect_stuck(steps):", 0.030),
    ]
    out = LexicalReranker().rerank("detect_stuck", cands, top_k=2)
    assert out[0].path == "b.py"


# --- retries -----------------------------------------------------------------


class _Flaky:
    def __init__(self, exc: Exception, fails: int) -> None:
        self.exc, self.fails, self.calls = exc, fails, 0

    def complete(self, messages, *, temperature=0.2, json_mode=False, tools=None):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return CompletionResult(content="ok", usage=Usage())

    def stream(self, messages, *, temperature=0.2):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        yield "a"
        yield "b"


MSG = [Message(role=Role.USER, content="hi")]


def test_transient_error_retried_and_retry_after_honored() -> None:
    sleeps: list[float] = []
    inner = _Flaky(TransientLLMError("429", status_code=429, retry_after_s=2.5), fails=2)
    out = RetryingLLMClient(inner, max_attempts=3, sleep=sleeps.append).complete(MSG)
    assert out.content == "ok"
    assert sleeps == [2.5, 2.5]


def test_non_transient_error_not_retried() -> None:
    inner = _Flaky(ValueError("bad request"), fails=1)
    with pytest.raises(ValueError):
        RetryingLLMClient(inner, max_attempts=3, sleep=lambda _s: None).complete(MSG)
    assert inner.calls == 1


def test_stream_retries_before_first_chunk() -> None:
    """Generators fail lazily; the old wrapper retried a call that never raised."""
    inner = _Flaky(ConnectionError("reset"), fails=2)
    chunks = list(RetryingLLMClient(inner, max_attempts=3, sleep=lambda _s: None).stream(MSG))
    assert chunks == ["a", "b"]
    assert inner.calls == 3


# --- anthropic adapter -------------------------------------------------------


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    real_client = httpx.Client

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    monkeypatch.setattr("cozmo.infra.llm.anthropic_client.log_llm_curl", lambda **_: None)
    return seen


def test_anthropic_rate_limit_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(
        monkeypatch, lambda r: httpx.Response(429, headers={"retry-after": "3"}, text="slow")
    )
    with pytest.raises(TransientLLMError) as info:
        AnthropicClient(model="m", api_key="k").complete(MSG)
    assert info.value.status_code == 429
    assert info.value.retry_after_s == 3.0


def test_anthropic_client_error_is_not_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch, lambda r: httpx.Response(400, text="bad"))
    with pytest.raises(RuntimeError) as info:
        AnthropicClient(model="m", api_key="k").complete(MSG)
    assert not isinstance(info.value, TransientLLMError)


def test_anthropic_parses_tool_use_and_forces_json(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "content": [
            {"type": "text", "text": "reading"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 5},
        "stop_reason": "tool_use",
    }
    seen = _patch_httpx(monkeypatch, lambda r: httpx.Response(200, json=body))
    tool = ToolSpec(name="read_file", description="d", parameters={"type": "object"})
    out = AnthropicClient(model="m", api_key="k").complete(MSG, json_mode=True, tools=[tool])
    assert out.content == "reading"
    assert out.tool_calls[0].name == "read_file"
    assert json.loads(out.tool_calls[0].arguments) == {"path": "a.py"}
    assert out.usage.prompt_tokens == 11
    sent = json.loads(seen[0].content)
    assert "JSON" in sent["system"]  # json_mode without a system prompt used to be dropped
    assert sent["tools"][0]["name"] == "read_file"


# --- embedder batching -------------------------------------------------------


def test_openai_embedder_caps_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from cozmo.infra.rag.openai_embedder import OpenAICompatibleEmbedder

    emb = OpenAICompatibleEmbedder(model="m", api_key="k", max_batch=4)
    sizes: list[int] = []

    def create(*, model, input):
        items = input if isinstance(input, list) else [input]
        sizes.append(len(items))
        data = [SimpleNamespace(index=i, embedding=[float(len(t))]) for i, t in enumerate(items)]
        return SimpleNamespace(data=list(reversed(data)))

    monkeypatch.setattr(emb._client.embeddings, "create", create)
    texts = [f"t{'x' * i}" for i in range(10)]
    out = emb.embed_many(texts)
    assert sizes == [4, 4, 2]
    assert out == [[float(len(t))] for t in texts]  # order restored from .index
