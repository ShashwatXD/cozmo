"""Groundedness + citation eval scoring."""

from cozmo.app.groundedness import (
    Citation,
    GroundingContext,
    extract_citations,
    score_groundedness,
    split_claims,
    validate_citation,
)


def _contexts() -> list[GroundingContext]:
    return [
        GroundingContext(
            path="server/src/rag/indexer.ts",
            start_line=200,
            end_line=240,
            content=(
                'const insert = sqlite.prepare("INSERT INTO rag_chunks '
                '(id, path, start_line, end_line, content, embedding_json, indexed_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)");'
            ),
        ),
        GroundingContext(
            path="server/src/rag/retrieve.ts",
            start_line=20,
            end_line=55,
            content=(
                "const hits = withVectors.map((c) => ({ path: c.path, "
                "score: cosineSimilarity(qVec, c.embedding) }))"
            ),
        ),
    ]


def test_split_claims() -> None:
    claims = split_claims(
        "Chunks are inserted into rag_chunks.\n- Ranking uses cosine similarity.\nOk."
    )
    assert len(claims) == 2
    assert claims[1].startswith("Ranking uses cosine")


def test_supported_claim() -> None:
    report = score_groundedness(
        answer="Chunks are inserted into rag_chunks with an embedding_json column.",
        contexts=_contexts(),
    )
    assert not report.unsupported
    assert report.score >= 60


def test_unsupported_claim() -> None:
    report = score_groundedness(
        answer=(
            "Vectors are sharded across a Redis cluster using HNSW graphs "
            "for approximate search."
        ),
        contexts=_contexts(),
    )
    assert len(report.unsupported) >= 1
    assert report.score < 60


def test_empty_claims_score_zero() -> None:
    report = score_groundedness(answer="Ok.", contexts=_contexts())
    assert report.claims == []
    assert report.score == 0


def test_extract_citations() -> None:
    found = extract_citations(
        "See server/src/rag/retrieve.ts:20-55 and server/src/rag/indexer.ts."
    )
    assert len(found) == 2
    assert found[0].start_line == 20
    assert found[0].end_line == 55
    assert found[1].start_line is None


def test_validate_citation_in_range() -> None:
    ctx = _contexts()
    assert validate_citation(
        Citation(
            raw="x",
            path="server/src/rag/retrieve.ts",
            start_line=30,
            end_line=40,
        ),
        ctx,
    )


def test_validate_citation_unknown_file() -> None:
    assert not validate_citation(
        Citation(raw="x", path="server/src/rag/pinecone.ts", start_line=None, end_line=None),
        _contexts(),
    )


def test_validate_citation_out_of_range() -> None:
    assert not validate_citation(
        Citation(
            raw="x",
            path="server/src/rag/retrieve.ts",
            start_line=900,
            end_line=950,
        ),
        _contexts(),
    )


def test_fabricated_citation_lowers_score() -> None:
    ctx = _contexts()
    honest = score_groundedness(
        answer=(
            "Ranking uses cosine similarity over stored vectors, "
            "see server/src/rag/retrieve.ts:20-55."
        ),
        contexts=ctx,
    )
    fabricated = score_groundedness(
        answer=(
            "Ranking uses cosine similarity over stored vectors, "
            "see server/src/rag/pinecone.ts:20-55."
        ),
        contexts=ctx,
    )
    assert honest.score > fabricated.score
