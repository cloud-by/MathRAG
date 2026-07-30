from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.infrastructure.embedding import provider as embedding_provider_module
from app.modules.knowledge.errors import EmbeddingUnavailableError
from scripts import evaluate_pgvector_retrieval as evaluator_module
from scripts.evaluate_pgvector_retrieval import (
    EvaluationInputError,
    EvaluationThresholdError,
    RetrievalMetrics,
    _percentile,
    assert_thresholds,
    build_artifact,
    calculate_metrics,
    evaluate_questions,
    hash_provider_origin,
    run_evaluation,
    write_success_artifact,
)
from scripts.legacy_faiss_retriever import LegacyFaissRetriever


def _question_row(
    *,
    expected: list[str],
    legacy: list[str],
    pgvector: list[str],
    latency_ms: float,
    top_k: int,
    question_id: str = "rq-0001",
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "expected_legacy_ids": expected,
        "legacy_source_ids": legacy,
        "pgvector_source_ids": pgvector,
        "pgvector_latency_ms": latency_ms,
        "top_k": top_k,
    }


def _passing_metrics(**overrides: object) -> RetrievalMetrics:
    values: dict[str, object] = {
        "total_questions": 26,
        "expected_hit_count": 24,
        "expected_hit_rate": 24 / 26,
        "average_top_k_overlap": 24 / 26,
        "pgvector_p50_ms": 10.0,
        "pgvector_p95_ms": 100.0,
    }
    values.update(overrides)
    return RetrievalMetrics(**values)


def _artifact_question() -> dict[str, object]:
    return {
        "question_id": "rq-0001",
        "expected_legacy_ids": ["k0001"],
        "legacy_source_ids": ["k0001", "k0002", "k0003"],
        "pgvector_source_ids": ["k0001", "k0002", "k0004"],
        "pgvector_latency_ms": 4.5,
        "expected_hit": True,
        "top_k_overlap": 2 / 3,
        "top_k": 3,
    }


def _artifact_questions() -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for index in range(1, 27):
        expected_id = f"k{index:04d}"
        legacy_ids = [expected_id, f"k{index + 100:04d}", f"k{index + 200:04d}"]
        expected_hit = index <= 24
        pgvector_ids = (
            list(legacy_ids)
            if expected_hit
            else [
                f"k{index + 300:04d}",
                f"k{index + 400:04d}",
                f"k{index + 500:04d}",
            ]
        )
        questions.append(
            {
                "question_id": f"rq-{index:04d}",
                "expected_legacy_ids": [expected_id],
                "legacy_source_ids": legacy_ids,
                "pgvector_source_ids": pgvector_ids,
                "pgvector_latency_ms": 10.0 if index <= 24 else 100.0,
                "expected_hit": expected_hit,
                "top_k_overlap": 1.0 if expected_hit else 0.0,
                "top_k": 3,
            }
        )
    return questions


def test_calculate_metrics_uses_expected_set_and_each_rows_top_k() -> None:
    rows = [
        _question_row(
            expected=["k0001"],
            legacy=["k0001", "k0002", "k0003"],
            pgvector=["k0001", "k0002", "k0004"],
            latency_ms=4.0,
            top_k=3,
        ),
        _question_row(
            question_id="rq-0002",
            expected=["k0005"],
            legacy=["k0005", "k0006"],
            pgvector=["k0008", "k0006"],
            latency_ms=6.0,
            top_k=2,
        ),
    ]

    metrics = calculate_metrics(rows)

    assert metrics == RetrievalMetrics(
        total_questions=2,
        expected_hit_count=1,
        expected_hit_rate=pytest.approx(0.5),
        average_top_k_overlap=pytest.approx(((2 / 3) + (1 / 2)) / 2),
        pgvector_p50_ms=pytest.approx(4.0),
        pgvector_p95_ms=pytest.approx(6.0),
    )


def test_metrics_are_frozen() -> None:
    metrics = _passing_metrics()

    with pytest.raises(FrozenInstanceError):
        metrics.total_questions = 25  # type: ignore[misc]


def test_percentile_uses_documented_nearest_rank_definition() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert _percentile(values, 0) == pytest.approx(1.0)
    assert _percentile(values, 50) == pytest.approx(2.0)
    assert _percentile(values, 95) == pytest.approx(4.0)
    assert _percentile(values, 100) == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("values", "percentile"),
    [([], 50), ([1.0, float("nan")], 50), ([1.0], -1), ([1.0], 101)],
)
def test_percentile_rejects_undefined_inputs(
    values: list[float], percentile: float
) -> None:
    with pytest.raises(EvaluationInputError):
        _percentile(values, percentile)


def test_threshold_boundary_values_pass() -> None:
    assert_thresholds(_passing_metrics())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_questions": 25}, "26"),
        ({"expected_hit_count": 23, "expected_hit_rate": 23 / 26}, "90%"),
        ({"average_top_k_overlap": 0.799999}, "80%"),
        ({"pgvector_p95_ms": 100.000001}, "100 ms"),
    ],
)
def test_thresholds_fail_without_lowering_contract(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(EvaluationThresholdError, match=message):
        assert_thresholds(_passing_metrics(**overrides))


@pytest.mark.parametrize(
    "field_name",
    ["expected_hit_rate", "average_top_k_overlap", "pgvector_p95_ms"],
)
def test_thresholds_reject_nan_metrics(field_name: str) -> None:
    with pytest.raises(EvaluationThresholdError):
        assert_thresholds(_passing_metrics(**{field_name: float("nan")}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_hit_rate": -0.1},
        {"expected_hit_rate": 1.1},
        {"average_top_k_overlap": -0.1},
        {"average_top_k_overlap": 1.1},
        {"pgvector_p50_ms": -0.1},
        {"pgvector_p95_ms": -0.1},
    ],
)
def test_thresholds_reject_out_of_range_metrics(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(EvaluationThresholdError):
        assert_thresholds(_passing_metrics(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"expected_hit_rate": 0.90},
        {"expected_hit_count": True},
        {"expected_hit_count": -1, "expected_hit_rate": -1 / 26},
        {"expected_hit_count": 27, "expected_hit_rate": 27 / 26},
        {"pgvector_p50_ms": 50.0, "pgvector_p95_ms": 10.0},
    ],
)
def test_thresholds_reject_internally_inconsistent_metrics(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(EvaluationThresholdError):
        assert_thresholds(_passing_metrics(**overrides))


def test_provider_origin_is_canonicalized_then_only_hashed() -> None:
    expected = hashlib.sha256(b"https://example.test").hexdigest()

    digest = hash_provider_origin(
        "HTTPS://user:password@EXAMPLE.TEST:443/v1/embeddings?token=hidden"
    )

    assert digest == expected
    assert "example.test" not in digest
    assert len(digest) == 64


@pytest.mark.parametrize(
    "url",
    ["", "example.test/v1", "ftp://example.test/v1", "https://", "https://x:bad/v1"],
)
def test_provider_origin_rejects_invalid_urls_without_echoing_them(url: str) -> None:
    with pytest.raises(EvaluationInputError) as captured:
        hash_provider_origin(url)

    if url:
        assert url not in str(captured.value)


def test_artifact_contains_only_auditable_safe_fields() -> None:
    generated_at = datetime(2026, 7, 30, 1, 2, 3, tzinfo=timezone.utc)
    questions = _artifact_questions()

    artifact = build_artifact(
        metrics=_passing_metrics(),
        questions=questions,
        git_sha="a" * 40,
        git_tree_sha="1" * 40,
        fixture_sha256="b" * 64,
        seed_sha256="c" * 64,
        faiss_sha256="d" * 64,
        id_map_sha256="e" * 64,
        embedding_model="embedding-test",
        dimensions=1024,
        provider_origin_sha256="f" * 64,
        generated_at=generated_at,
    )

    assert artifact == {
        "schema_version": "1.1",
        "generated_at": "2026-07-30T01:02:03Z",
        "git_sha": "a" * 40,
        "git_tree_sha": "1" * 40,
        "inputs": {
            "fixture_sha256": "b" * 64,
            "seed_sha256": "c" * 64,
            "faiss_sha256": "d" * 64,
            "id_map_sha256": "e" * 64,
        },
        "embedding": {
            "model": "embedding-test",
            "dimensions": 1024,
            "provider_origin_sha256": "f" * 64,
        },
        "methodology": {
            "top_k": 3,
            "warmup_queries": 1,
            "warmup_strategy": "first_query_vector",
            "timed_queries": 26,
            "latency_scope": "repository_sql_only",
        },
        "metrics": {
            "total_questions": 26,
            "expected_hit_count": 24,
            "expected_hit_rate": 24 / 26,
            "average_top_k_overlap": 24 / 26,
            "pgvector_p50_ms": 10.0,
            "pgvector_p95_ms": 100.0,
        },
        "questions": questions,
    }
    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    for forbidden in (
        "api_key",
        "base_url",
        "authorization",
        "password",
        "secret",
        "正文",
        "://",
    ):
        assert forbidden not in serialized


def test_artifact_records_commit_and_tree_evidence() -> None:
    artifact = build_artifact(
        metrics=_passing_metrics(),
        questions=_artifact_questions(),
        git_sha="a" * 40,
        git_tree_sha="b" * 40,
        fixture_sha256="c" * 64,
        seed_sha256="d" * 64,
        faiss_sha256="e" * 64,
        id_map_sha256="f" * 64,
        embedding_model="embedding-test",
        dimensions=1024,
        provider_origin_sha256="1" * 64,
        generated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    assert artifact["git_sha"] == "a" * 40
    assert artifact["git_tree_sha"] == "b" * 40


def _build_test_artifact(
    questions: list[dict[str, object]],
    metrics: RetrievalMetrics,
) -> dict[str, object]:
    return build_artifact(
        metrics=metrics,
        questions=questions,
        git_sha="a" * 40,
        git_tree_sha="b" * 40,
        fixture_sha256="c" * 64,
        seed_sha256="d" * 64,
        faiss_sha256="e" * 64,
        id_map_sha256="f" * 64,
        embedding_model="embedding-test",
        dimensions=1024,
        provider_origin_sha256="1" * 64,
        generated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("case", ["too_few", "too_many", "duplicate_id"])
def test_artifact_requires_exactly_26_unique_question_ids(case: str) -> None:
    questions = _artifact_questions()
    if case == "too_few":
        questions.pop()
    elif case == "too_many":
        extra = dict(questions[-1])
        extra["question_id"] = "rq-0027"
        questions.append(extra)
    else:
        questions[-1]["question_id"] = questions[0]["question_id"]

    with pytest.raises(EvaluationInputError):
        _build_test_artifact(questions, calculate_metrics(questions))


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("total_questions", 25),
        ("expected_hit_count", 23),
        ("expected_hit_rate", 0.91),
        ("average_top_k_overlap", 0.91),
        ("pgvector_p50_ms", 11.0),
        ("pgvector_p95_ms", 99.0),
    ],
)
def test_artifact_rejects_metrics_not_recalculated_from_details(
    field_name: str,
    forged_value: object,
) -> None:
    questions = _artifact_questions()
    values = vars(calculate_metrics(questions)) | {field_name: forged_value}

    with pytest.raises(EvaluationInputError, match="指标"):
        _build_test_artifact(questions, RetrievalMetrics(**values))


@pytest.mark.parametrize(
    "case",
    ["string_latency", "string_overlap", "duplicate_source_id", "non_top_three"],
)
def test_artifact_rejects_non_strict_question_result_fields(case: str) -> None:
    questions = _artifact_questions()
    first = questions[0]
    if case == "string_latency":
        first["pgvector_latency_ms"] = "10.0"
    elif case == "string_overlap":
        first["top_k_overlap"] = "1.0"
    elif case == "duplicate_source_id":
        first["legacy_source_ids"] = ["k0001", "k0001", "k0101"]
        first["top_k_overlap"] = 2 / 3
    else:
        first["top_k"] = 2
        first["legacy_source_ids"] = first["legacy_source_ids"][:2]
        first["pgvector_source_ids"] = first["pgvector_source_ids"][:2]
        first["top_k_overlap"] = 1.0

    with pytest.raises(EvaluationInputError):
        _build_test_artifact(questions, calculate_metrics(questions))


@pytest.mark.parametrize(
    "field_name",
    [
        "fixture_sha256",
        "seed_sha256",
        "faiss_sha256",
        "id_map_sha256",
        "provider_origin_sha256",
    ],
)
def test_artifact_rejects_every_invalid_sha256(field_name: str) -> None:
    kwargs: dict[str, object] = {
        "metrics": _passing_metrics(),
        "questions": _artifact_questions(),
        "git_sha": "a" * 40,
        "git_tree_sha": "1" * 40,
        "fixture_sha256": "b" * 64,
        "seed_sha256": "c" * 64,
        "faiss_sha256": "d" * 64,
        "id_map_sha256": "e" * 64,
        "embedding_model": "embedding-test",
        "dimensions": 1024,
        "provider_origin_sha256": "f" * 64,
        "generated_at": datetime(2026, 7, 30, tzinfo=timezone.utc),
    }
    kwargs[field_name] = "not-a-sha"

    with pytest.raises(EvaluationInputError, match="SHA-256"):
        build_artifact(**kwargs)


@pytest.mark.parametrize(
    "unsafe_update",
    [
        {"question": "完整问题不应写入"},
        {"content": "知识正文不应写入"},
        {"api_key": "hidden"},
        {"legacy_source_ids": ["https://secret.example/item"]},
    ],
)
def test_artifact_rejects_unknown_or_secret_question_payloads(
    unsafe_update: dict[str, object],
) -> None:
    questions = _artifact_questions()
    questions[0].update(unsafe_update)

    with pytest.raises(EvaluationInputError):
        build_artifact(
            metrics=calculate_metrics(questions),
            questions=questions,
            git_sha="a" * 40,
            git_tree_sha="1" * 40,
            fixture_sha256="b" * 64,
            seed_sha256="c" * 64,
            faiss_sha256="d" * 64,
            id_map_sha256="e" * 64,
            embedding_model="embedding-test",
            dimensions=1024,
            provider_origin_sha256="f" * 64,
            generated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )


class _FakeIndex:
    def __init__(self, *, ntotal: int, indices: list[object]) -> None:
        self.ntotal = ntotal
        self.d = 1024
        self._indices = indices
        self.calls: list[tuple[np.ndarray, int]] = []

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, object]:
        self.calls.append((query, top_k))
        return (
            np.zeros((1, len(self._indices)), dtype="float32"),
            SimpleNamespace(tolist=lambda: [list(self._indices)]),
        )


def _build_legacy_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    index: _FakeIndex,
    id_map_text: str,
) -> LegacyFaissRetriever:
    from scripts import legacy_faiss_retriever as module

    index_path = tmp_path / "faiss.index"
    id_map_path = tmp_path / "id_map.json"
    index_path.write_bytes(b"read-only-fixture")
    id_map_path.write_text(id_map_text, encoding="utf-8")
    monkeypatch.setattr(module.faiss, "read_index", lambda path: index)
    return LegacyFaissRetriever(index_path=index_path, id_map_path=id_map_path)


def test_legacy_adapter_searches_valid_vector_and_accepts_numpy_integer_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = _FakeIndex(ntotal=3, indices=[0, np.int64(1), 2])
    retriever = _build_legacy_adapter(
        tmp_path,
        monkeypatch,
        index=index,
        id_map_text=json.dumps(
            {
                "0": {"source_id": "k0001", "content": "不得返回"},
                "1": {"source_id": "k0002"},
                "2": {"source_id": "k0003"},
            },
            ensure_ascii=False,
        ),
    )
    vector = [1.0] + [0.0] * 1023

    source_ids = retriever.search_vector(vector, top_k=10)

    assert source_ids == ["k0001", "k0002", "k0003"]
    assert len(index.calls) == 1
    query, requested_k = index.calls[0]
    assert query.shape == (1, 1024)
    assert query.dtype == np.dtype("float32")
    assert requested_k == 3
    assert not hasattr(retriever, "add")
    assert not hasattr(retriever, "write")


@pytest.mark.parametrize("raw_index", [1.0, "1", True, -1, 2])
def test_legacy_adapter_rejects_invalid_raw_index_before_id_map_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_index: object,
) -> None:
    index = _FakeIndex(ntotal=2, indices=[raw_index])
    retriever = _build_legacy_adapter(
        tmp_path,
        monkeypatch,
        index=index,
        id_map_text=json.dumps(
            {"0": {"source_id": "k0001"}, "1": {"source_id": "k0002"}}
        ),
    )

    class NoLookup(dict[str, dict[str, str]]):
        def get(self, key: str, default: object = None) -> dict[str, str] | None:
            raise AssertionError("无效索引不得查询 id_map")

    retriever._id_map = NoLookup(retriever._id_map)

    with pytest.raises(ValueError, match="FAISS"):
        retriever.search_vector([1.0] + [0.0] * 1023, top_k=1)


@pytest.mark.parametrize(
    "id_map_text",
    [
        "[]",
        '{"0": {"source_id": "k0001"}, "0": {"source_id": "k0002"}}',
        '{"0": {"source_id": "k0001"}}',
        '{"0": {"source_id": "k0001"}, "1": []}',
        '{"0": {"source_id": "k0001"}, "1": {"source_id": ""}}',
        '{"0": {"source_id": "k0001"}, "1": {"source_id": "k0001"}}',
    ],
)
def test_legacy_adapter_rejects_malformed_duplicate_or_incomplete_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    id_map_text: str,
) -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        _build_legacy_adapter(
            tmp_path,
            monkeypatch,
            index=_FakeIndex(ntotal=2, indices=[]),
            id_map_text=id_map_text,
        )


@pytest.mark.parametrize(
    ("vector", "top_k"),
    [
        ([1.0] * 1023, 3),
        ([0.0] * 1024, 3),
        ([float("nan")] + [0.0] * 1023, 3),
        ([True] + [0.0] * 1023, 3),
        (None, 3),
        ([1.0] + [0.0] * 1023, 0),
        ([1.0] + [0.0] * 1023, 11),
        ([1.0] + [0.0] * 1023, True),
    ],
)
def test_legacy_adapter_rejects_invalid_vector_or_top_k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    vector: object,
    top_k: int,
) -> None:
    retriever = _build_legacy_adapter(
        tmp_path,
        monkeypatch,
        index=_FakeIndex(ntotal=1, indices=[0]),
        id_map_text='{"0": {"source_id": "k0001"}}',
    )

    with pytest.raises(ValueError):
        retriever.search_vector(vector, top_k=top_k)


@pytest.mark.parametrize("value", [1e39, 1e-50])
def test_legacy_adapter_rejects_float32_overflow_or_underflow_before_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    index = _FakeIndex(ntotal=1, indices=[0])
    retriever = _build_legacy_adapter(
        tmp_path,
        monkeypatch,
        index=index,
        id_map_text='{"0": {"source_id": "k0001"}}',
    )

    with pytest.raises(ValueError) as captured:
        retriever.search_vector([value] + [0.0] * 1023, top_k=1)

    assert str(value) not in str(captured.value)
    assert index.calls == []


def test_evaluator_rejects_invalid_provider_contract() -> None:
    questions = [
        {
            "id": f"rq-{index:04d}",
            "question": f"问题 {index}",
            "expected_legacy_ids": [f"k{index:04d}"],
        }
        for index in range(1, 27)
    ]

    class InvalidProvider:
        model = None
        dimensions = 1024

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise AssertionError("无效 Provider 不应发起向量化")

        async def aclose(self) -> None:
            return None

    with pytest.raises(EvaluationInputError):
        asyncio.run(
            evaluate_questions(
                questions=questions,
                top_k=3,
                provider=InvalidProvider(),
                legacy_retriever=object(),
                session_factory=lambda: None,
            )
        )


class _FakeProvider:
    def __init__(self, vectors: list[list[float]], events: list[object]) -> None:
        self.model = "embedding-test"
        self.dimensions = 1024
        self.vectors = vectors
        self.events = events
        self.closed = False

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.events.append(("embed", list(texts)))
        return self.vectors

    async def aclose(self) -> None:
        self.events.append("provider-close")
        self.closed = True


class _GlobalFakeProvider:
    model = "embedding-test"
    dimensions = 1024

    def __init__(self, *, close_error: BaseException | None = None) -> None:
        self.close_calls = 0
        self.close_error = close_error

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _install_global_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: _GlobalFakeProvider,
) -> None:
    monkeypatch.setattr(embedding_provider_module, "_embedding_provider", provider)
    monkeypatch.setattr(
        evaluator_module,
        "get_embedding_provider",
        embedding_provider_module.get_embedding_provider,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator_module,
        "dispose_embedding_provider",
        embedding_provider_module.dispose_embedding_provider,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator_module,
        "OpenAIEmbeddingProvider",
        lambda: provider,
        raising=False,
    )
    monkeypatch.setattr(
        evaluator_module,
        "_git_evidence",
        lambda: ("a" * 40, "b" * 40),
        raising=False,
    )


def test_evaluator_batches_provider_and_times_only_repository_sql() -> None:
    events: list[object] = []
    questions = [
        {
            "id": f"rq-{index:04d}",
            "question": f"问题 {index}",
            "expected_legacy_ids": [f"k{index:04d}"],
        }
        for index in range(1, 27)
    ]
    vectors = [[float(index)] + [0.0] * 1023 for index in range(1, 27)]
    provider = _FakeProvider(vectors, events)
    vector_positions = {id(vector): index for index, vector in enumerate(vectors, start=1)}

    class FakeLegacy:
        def search_vector(self, vector: list[float], *, top_k: int) -> list[str]:
            position = vector_positions[id(vector)]
            events.append(("legacy", position, id(vector), top_k))
            return [f"k{position:04d}"]

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            events.append("session-enter")
            return object()

        async def __aexit__(self, *args: object) -> None:
            events.append("session-exit")

    def session_factory() -> FakeSessionContext:
        events.append("session-create")
        return FakeSessionContext()

    class FakeRepository:
        def __init__(self, session: object) -> None:
            events.append("repository-create")

        async def search_ready_chunks(
            self,
            *,
            query_vector: list[float],
            embedding_model: str,
            limit: int,
        ) -> list[SimpleNamespace]:
            position = vector_positions[id(query_vector)]
            events.append(("repository-search", position, id(query_vector), limit))
            assert embedding_model == "embedding-test"
            return [SimpleNamespace(legacy_source_id=f"k{position:04d}")]

    ticks = iter(value for index in range(26) for value in (index, index + 0.004))

    def timer() -> float:
        value = next(ticks)
        events.append(("timer", value))
        return value

    rows, metrics = asyncio.run(
        evaluate_questions(
            questions=questions,
            top_k=3,
            provider=provider,
            legacy_retriever=FakeLegacy(),
            session_factory=session_factory,
            repository_factory=FakeRepository,
            timer=timer,
        )
    )

    assert events[0] == ("embed", [question["question"] for question in questions])
    assert len([event for event in events if isinstance(event, tuple) and event[0] == "embed"]) == 1
    warmup_events = events[1:6]
    assert [event if isinstance(event, str) else event[0] for event in warmup_events] == [
        "session-create",
        "session-enter",
        "repository-create",
        "repository-search",
        "session-exit",
    ]
    assert isinstance(warmup_events[3], tuple)
    assert warmup_events[3][2] == id(vectors[0])

    first_question_events = events[6:14]
    assert [event if isinstance(event, str) else event[0] for event in first_question_events] == [
        "legacy",
        "session-create",
        "session-enter",
        "repository-create",
        "timer",
        "repository-search",
        "timer",
        "session-exit",
    ]
    legacy_event = first_question_events[0]
    repository_event = first_question_events[5]
    assert isinstance(legacy_event, tuple)
    assert isinstance(repository_event, tuple)
    assert legacy_event[2] == repository_event[2] == id(vectors[0])
    assert rows[0] == {
        "question_id": "rq-0001",
        "expected_legacy_ids": ["k0001"],
        "legacy_source_ids": ["k0001"],
        "pgvector_source_ids": ["k0001"],
        "pgvector_latency_ms": pytest.approx(4.0),
        "expected_hit": True,
        "top_k_overlap": pytest.approx(1 / 3),
        "top_k": 3,
    }
    assert metrics.total_questions == 26
    assert metrics.expected_hit_count == 26
    assert metrics.average_top_k_overlap == pytest.approx(1 / 3)
    assert len(
        [
            event
            for event in events
            if isinstance(event, tuple) and event[0] == "repository-search"
        ]
    ) == 27
    assert len(
        [event for event in events if isinstance(event, tuple) and event[0] == "timer"]
    ) == 52


def test_git_evidence_checks_tracked_and_index_before_reading_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        outputs = {
            ("git", "rev-parse", "HEAD"): "a" * 40,
            ("git", "rev-parse", "HEAD^{tree}"): "b" * 40,
        }
        return SimpleNamespace(returncode=0, stdout=outputs.get(tuple(command), ""))

    monkeypatch.setattr(evaluator_module.subprocess, "run", run)

    assert evaluator_module._git_evidence() == ("a" * 40, "b" * 40)
    assert calls == [
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "HEAD^{tree}"],
    ]


@pytest.mark.parametrize(
    "dirty_command",
    [
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ],
)
def test_git_evidence_rejects_dirty_tracked_or_index_state(
    monkeypatch: pytest.MonkeyPatch,
    dirty_command: list[str],
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=1 if command == dirty_command else 0, stdout="")

    monkeypatch.setattr(evaluator_module.subprocess, "run", run)

    with pytest.raises(EvaluationInputError, match="Git"):
        evaluator_module._git_evidence()

    assert ["git", "rev-parse", "HEAD"] not in calls


def test_cli_rejects_dirty_evidence_before_external_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_dirty() -> tuple[str, str]:
        events.append("git-evidence")
        raise EvaluationInputError("Git tracked/index 必须干净")

    def open_provider() -> object:
        events.append("provider")
        raise AssertionError("不应创建 Provider")

    monkeypatch.setattr(evaluator_module, "_git_evidence", reject_dirty, raising=False)
    monkeypatch.setattr(evaluator_module, "get_embedding_provider", open_provider)

    with pytest.raises(EvaluationInputError, match="Git"):
        asyncio.run(
            evaluator_module._run_cli(
                Path("fixture.json"), tmp_path / "parity.json"
            )
        )

    assert events == ["git-evidence"]


def test_cli_existing_output_is_preserved_without_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "parity.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    calls: list[str] = []

    async def run_cli(fixture_path: Path, requested_output: Path) -> RetrievalMetrics:
        calls.append("run")
        return _passing_metrics()

    monkeypatch.setattr(evaluator_module, "_run_cli", run_cli)

    exit_code = evaluator_module.main(
        ["--fixture", "fixture.json", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert calls == []
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"old": True}
    assert "output_exists" in captured.err
    assert captured.out == ""


def test_cli_replace_failure_removes_old_target_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "parity.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    async def fail(
        fixture_path: Path,
        requested_output: Path,
        *,
        git_evidence: tuple[str, str] | None = None,
    ) -> RetrievalMetrics:
        assert not requested_output.exists()
        assert git_evidence == ("a" * 40, "b" * 40)
        raise EvaluationThresholdError("threshold")

    monkeypatch.setattr(
        evaluator_module, "_git_evidence", lambda: ("a" * 40, "b" * 40)
    )
    monkeypatch.setattr(evaluator_module, "_run_cli", fail)

    exit_code = evaluator_module.main(
        [
            "--fixture",
            "fixture.json",
            "--output",
            str(output_path),
            "--replace-existing",
        ]
    )

    assert exit_code == 4
    assert not output_path.exists()


def test_cli_replace_success_atomically_writes_new_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "parity.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    async def succeed(
        fixture_path: Path,
        requested_output: Path,
        *,
        git_evidence: tuple[str, str] | None = None,
    ) -> RetrievalMetrics:
        assert not requested_output.exists()
        assert git_evidence == ("a" * 40, "b" * 40)
        evaluator_module._atomic_write_json(requested_output, {"new": True})
        return _passing_metrics()

    monkeypatch.setattr(
        evaluator_module, "_git_evidence", lambda: ("a" * 40, "b" * 40)
    )
    monkeypatch.setattr(evaluator_module, "_run_cli", succeed)

    exit_code = evaluator_module.main(
        [
            "--fixture",
            "fixture.json",
            "--output",
            str(output_path),
            "--replace-existing",
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"new": True}


def test_cli_replace_freezes_clean_evidence_before_removing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "parity.json"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    events: list[str] = []

    def git_evidence() -> tuple[str, str]:
        assert output_path.exists()
        events.append("evidence")
        return "a" * 40, "b" * 40

    async def succeed(
        fixture_path: Path,
        requested_output: Path,
        *,
        git_evidence: tuple[str, str] | None = None,
    ) -> RetrievalMetrics:
        assert not requested_output.exists()
        assert git_evidence == ("a" * 40, "b" * 40)
        events.append("evaluation")
        return _passing_metrics()

    monkeypatch.setattr(evaluator_module, "_git_evidence", git_evidence)
    monkeypatch.setattr(evaluator_module, "_run_cli", succeed)

    exit_code = evaluator_module.main(
        [
            "--fixture",
            "fixture.json",
            "--output",
            str(output_path),
            "--replace-existing",
        ]
    )

    assert exit_code == 0
    assert events == ["evidence", "evaluation"]


@pytest.mark.parametrize("failure_stage", ["fixture", "session", "evaluation"])
def test_cli_failure_cleans_global_resources_once_and_redacts_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
) -> None:
    provider = _GlobalFakeProvider()
    _install_global_provider(monkeypatch, provider)
    dispose_calls: list[str] = []

    async def dispose_database() -> None:
        dispose_calls.append("database")

    def load_fixture(path: Path) -> tuple[dict[str, object], str]:
        if failure_stage == "fixture":
            raise RuntimeError("fixture-secret")
        return {"questions": [], "top_k": 3}, "a" * 64

    def session_factory() -> object:
        if failure_stage == "session":
            raise RuntimeError("session-secret")
        return object()

    async def evaluate(**kwargs: object) -> tuple[list[dict[str, object]], RetrievalMetrics]:
        if failure_stage == "evaluation":
            raise RuntimeError("evaluation-secret")
        raise AssertionError("未选择失败阶段")

    monkeypatch.setattr(evaluator_module, "dispose_engine", dispose_database)
    monkeypatch.setattr(evaluator_module, "_load_evaluation_fixture", load_fixture)
    monkeypatch.setattr(evaluator_module, "hash_provider_origin", lambda value: "f" * 64)
    monkeypatch.setattr(evaluator_module, "LegacyFaissRetriever", lambda **kwargs: object())
    monkeypatch.setattr(evaluator_module, "get_session_factory", session_factory)
    monkeypatch.setattr(evaluator_module, "run_evaluation", evaluate)
    output_path = tmp_path / "parity.json"

    exit_code = evaluator_module.main(
        ["--fixture", "fixture.json", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "secret" not in captured.err.lower()
    assert "RuntimeError" in captured.err
    assert provider.close_calls == 1
    assert embedding_provider_module._embedding_provider is None
    assert dispose_calls == ["database"]
    assert not output_path.exists()


def test_cli_cleanup_errors_do_not_override_business_exit_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = _GlobalFakeProvider(close_error=RuntimeError("cleanup-secret"))
    _install_global_provider(monkeypatch, provider)
    dispose_calls: list[str] = []

    async def failing_dispose_database() -> None:
        dispose_calls.append("database")
        raise RuntimeError("database-cleanup-secret")

    async def fail_business(**kwargs: object) -> tuple[list[dict[str, object]], RetrievalMetrics]:
        raise EmbeddingUnavailableError("business-secret")

    monkeypatch.setattr(evaluator_module, "dispose_engine", failing_dispose_database)
    monkeypatch.setattr(
        evaluator_module,
        "_load_evaluation_fixture",
        lambda path: ({"questions": [], "top_k": 3}, "a" * 64),
    )
    monkeypatch.setattr(evaluator_module, "hash_provider_origin", lambda value: "f" * 64)
    monkeypatch.setattr(evaluator_module, "LegacyFaissRetriever", lambda **kwargs: object())
    monkeypatch.setattr(evaluator_module, "get_session_factory", lambda: object())
    monkeypatch.setattr(evaluator_module, "run_evaluation", fail_business)
    output_path = tmp_path / "parity.json"

    exit_code = evaluator_module.main(
        ["--fixture", "fixture.json", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 3
    assert captured.out == ""
    assert "EmbeddingUnavailableError" in captured.err
    assert "secret" not in captured.err.lower()
    assert provider.close_calls == 1
    assert embedding_provider_module._embedding_provider is None
    assert dispose_calls == ["database"]
    assert not output_path.exists()


def test_cli_success_closes_global_resources_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _GlobalFakeProvider()
    _install_global_provider(monkeypatch, provider)
    dispose_calls: list[str] = []
    written_metrics: list[RetrievalMetrics] = []

    async def dispose_database() -> None:
        dispose_calls.append("database")

    async def evaluate(**kwargs: object) -> tuple[list[dict[str, object]], RetrievalMetrics]:
        return [], _passing_metrics()

    def write_artifact(output_path: Path, **kwargs: object) -> dict[str, object]:
        written_metrics.append(kwargs["metrics"])
        return {}

    monkeypatch.setattr(evaluator_module, "dispose_engine", dispose_database)
    monkeypatch.setattr(
        evaluator_module,
        "_load_evaluation_fixture",
        lambda path: ({"questions": [], "top_k": 3}, "a" * 64),
    )
    monkeypatch.setattr(evaluator_module, "hash_provider_origin", lambda value: "f" * 64)
    monkeypatch.setattr(evaluator_module, "LegacyFaissRetriever", lambda **kwargs: object())
    monkeypatch.setattr(evaluator_module, "get_session_factory", lambda: object())
    monkeypatch.setattr(evaluator_module, "run_evaluation", evaluate)
    monkeypatch.setattr(evaluator_module, "write_success_artifact", write_artifact)
    monkeypatch.setattr(evaluator_module, "sha256_file", lambda path: "b" * 64)

    metrics = asyncio.run(
        evaluator_module._run_cli(Path("fixture.json"), tmp_path / "parity.json")
    )

    assert metrics == _passing_metrics()
    assert written_metrics == [metrics]
    assert provider.close_calls == 1
    assert embedding_provider_module._embedding_provider is None
    assert dispose_calls == ["database"]


def test_threshold_failure_does_not_write_success_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "parity.json"

    with pytest.raises(EvaluationThresholdError):
        write_success_artifact(
            output_path,
            metrics=_passing_metrics(expected_hit_rate=0.5),
            questions=[],
            git_sha="a" * 40,
            git_tree_sha="1" * 40,
            fixture_sha256="b" * 64,
            seed_sha256="c" * 64,
            faiss_sha256="d" * 64,
            id_map_sha256="e" * 64,
            embedding_model="embedding-test",
            dimensions=1024,
            provider_origin_sha256="f" * 64,
            generated_at=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_threshold_failure_is_redacted_and_returns_four(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "parity.json"

    async def fail_threshold(
        fixture_path: Path, requested_output: Path
    ) -> RetrievalMetrics:
        raise EvaluationThresholdError(
            "https://provider.test/v1 api_key=must-not-be-printed"
        )

    monkeypatch.setattr(evaluator_module, "_run_cli", fail_threshold)

    exit_code = evaluator_module.main(
        ["--fixture", "fixture.json", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert not output_path.exists()
    assert captured.out == ""
    assert "https://" not in captured.err
    assert "api_key" not in captured.err
    assert "EvaluationThresholdError" in captured.err
