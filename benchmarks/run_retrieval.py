"""
Run the Cozmo retrieval benchmark.

    python benchmarks/run_retrieval.py                     # offline stub embedder
    python benchmarks/run_retrieval.py --embedder ollama   # local nomic-embed-text
    python benchmarks/run_retrieval.py --embedder openai   # text-embedding-3-small

Corpus is Cozmo's own source tree (src/cozmo) pinned at CORPUS_REF, so scores
stay comparable as the code evolves; labels live in queries.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

CORPUS_REF = "429fc96"  # upstream main when the benchmark was introduced
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from cozmo.app.retrieval_bench import (  # noqa: E402
    CachingEmbedder,
    load_queries,
    run_retrieval_bench,
)
from cozmo.infra.rag.embedder import StubEmbedder  # noqa: E402


def build(name: str):
    if name == "stub":
        return StubEmbedder(), "stub"
    from cozmo.infra.rag.openai_embedder import OpenAICompatibleEmbedder

    if name == "ollama":
        model = os.environ.get("COZMO_BENCH_MODEL", "nomic-embed-text")
        inner = OpenAICompatibleEmbedder(
            model=model, api_key="ollama", base_url="http://127.0.0.1:11434/v1"
        )
        return inner, f"ollama/{model}"
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("COZMO_API_KEY")
        if not key:
            raise SystemExit("set OPENAI_API_KEY for --embedder openai")
        model = os.environ.get("COZMO_BENCH_MODEL", "text-embedding-3-small")
        return OpenAICompatibleEmbedder(model=model, api_key=key), f"openai/{model}"
    raise SystemExit(f"unknown embedder {name}")


def pinned_corpus(ref: str, dest: Path) -> Path:
    """Extract src/cozmo at *ref* into *dest* (reproducible corpus)."""
    tar_path = dest / "corpus.tar"
    with tar_path.open("wb") as fh:
        subprocess.run(["git", "-C", str(ROOT), "archive", ref, "src/cozmo"], stdout=fh, check=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")
    return dest / "src" / "cozmo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="stub", choices=["stub", "ollama", "openai"])
    ap.add_argument("--corpus", type=Path, help="index this directory instead of the pinned ref")
    ap.add_argument("--ref", default=CORPUS_REF, help="git ref of src/cozmo to index")
    ap.add_argument("--queries", type=Path, default=HERE / "queries.jsonl")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", type=Path, help="write the full report as JSON")
    args = ap.parse_args()

    inner, label = build(args.embedder)
    embedder = inner
    if args.embedder != "stub":
        cache = HERE / ".cache" / f"{label.replace('/', '_')}.json"
        embedder = CachingEmbedder(inner, cache_path=cache, namespace=label)

    with tempfile.TemporaryDirectory() as tmp:
        corpus = args.corpus or pinned_corpus(args.ref, Path(tmp))
        report = run_retrieval_bench(
            corpus,
            load_queries(args.queries),
            embedder,
            embedder_name=label,
            repeats=args.repeats,
        )
    if isinstance(embedder, CachingEmbedder):
        embedder.save()
    print(report.to_markdown())
    for s in report.scores:
        print(f"\n{s.strategy}: by kind {s.by_kind}  misses@5 {s.misses}")
    if args.json:
        args.json.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
