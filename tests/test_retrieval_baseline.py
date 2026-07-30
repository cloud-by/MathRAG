from __future__ import annotations

import inspect
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_questions.json"
SEED_PATH = PROJECT_ROOT / "data" / "raw" / "math_knowledge_seed.jsonl"


def _baseline_module():
    from scripts import capture_retrieval_baseline

    return capture_retrieval_baseline


def _valid_fixture_payload() -> dict:
    questions = [
        {
            "id": f"rq-{index:04d}",
            "question": f"测试问题 {index}",
            "expected_legacy_ids": ["k0001"],
            "rationale": "用于验证固定问题集结构。",
        }
        for index in range(20)
    ]
    return {
        "schema_version": "1.0",
        "dataset_sha256": "a" * 64,
        "top_k": 3,
        "questions": questions,
    }


def test_retrieval_fixture_is_pinned_to_current_seed() -> None:
    assert FIXTURE_PATH.exists(), "固定检索问题集尚未创建"

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    seed_rows = [
        json.loads(line)
        for line in SEED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    legacy_ids = {row["id"] for row in seed_rows}

    assert fixture["schema_version"] == "1.0"
    normalized_seed = SEED_PATH.read_text(encoding="utf-8").encode("utf-8")
    assert fixture["dataset_sha256"] == hashlib.sha256(normalized_seed).hexdigest()
    assert fixture["top_k"] == 3
    assert len(fixture["questions"]) >= 20

    question_ids = [question["id"] for question in fixture["questions"]]
    assert len(question_ids) == len(set(question_ids))

    for question in fixture["questions"]:
        assert question["id"].strip()
        assert question["question"].strip()
        assert question["expected_legacy_ids"]
        assert set(question["expected_legacy_ids"]) <= legacy_ids
        assert question["rationale"].strip()


def test_validate_fixture_rejects_missing_question_field() -> None:
    baseline = _baseline_module()
    fixture = _valid_fixture_payload()
    fixture["questions"][0].pop("rationale")

    with pytest.raises(ValueError, match="rationale"):
        baseline.validate_fixture(
            fixture,
            valid_legacy_ids={"k0001"},
            dataset_sha256="a" * 64,
        )


def test_validate_fixture_rejects_unknown_expected_id() -> None:
    baseline = _baseline_module()
    fixture = _valid_fixture_payload()
    fixture["questions"][0]["expected_legacy_ids"] = ["k9999"]

    with pytest.raises(ValueError, match="k9999"):
        baseline.validate_fixture(
            fixture,
            valid_legacy_ids={"k0001"},
            dataset_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("mutate", "expected_error"),
    [
        (lambda fixture: fixture.update({"extra": True}), "未知字段"),
        (
            lambda fixture: fixture["questions"][0].update({"extra": True}),
            "未知字段",
        ),
        (lambda fixture: fixture.update({"top_k": 3.0}), "top_k"),
        (lambda fixture: fixture["questions"][0].update({"id": 1}), "id"),
        (
            lambda fixture: fixture["questions"][0].update({"id": "question-1"}),
            "id",
        ),
        (
            lambda fixture: fixture["questions"][0].update(
                {"question": "what is sine?"}
            ),
            "中文",
        ),
    ],
)
def test_validate_fixture_rejects_noncanonical_shapes(
    mutate,
    expected_error: str,
) -> None:
    baseline = _baseline_module()
    fixture = _valid_fixture_payload()
    mutate(fixture)

    with pytest.raises(ValueError, match=expected_error):
        baseline.validate_fixture(
            fixture,
            valid_legacy_ids={"k0001"},
            dataset_sha256="a" * 64,
        )


def test_normalize_results_keeps_only_top_three_public_fields() -> None:
    baseline = _baseline_module()
    raw_results = [
        {
            "rank": 99,
            "chunk_id": f"k000{index}_chunk_0",
            "source_id": f"k000{index}",
            "title": f"知识点 {index}",
            "score": str(1 - index / 10),
            "content": "不应写入基线的完整知识正文",
        }
        for index in range(1, 5)
    ]

    normalized = baseline.normalize_results(raw_results, top_k=3)

    assert normalized == [
        {
            "rank": 1,
            "chunk_id": "k0001_chunk_0",
            "source_id": "k0001",
            "title": "知识点 1",
            "score": 0.9,
        },
        {
            "rank": 2,
            "chunk_id": "k0002_chunk_0",
            "source_id": "k0002",
            "title": "知识点 2",
            "score": 0.8,
        },
        {
            "rank": 3,
            "chunk_id": "k0003_chunk_0",
            "source_id": "k0003",
            "title": "知识点 3",
            "score": 0.7,
        },
    ]


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_normalize_results_rejects_non_finite_scores(score: float) -> None:
    baseline = _baseline_module()

    with pytest.raises(ValueError, match="有限数值"):
        baseline.normalize_results(
            [{"chunk_id": "chunk", "source_id": "k0001", "score": score}],
            top_k=3,
        )


def test_calculate_summary_counts_expected_hits() -> None:
    baseline = _baseline_module()
    question_results = [
        {
            "expected_legacy_ids": ["k0001"],
            "results": [{"source_id": "k0001"}],
        },
        {
            "expected_legacy_ids": ["k0002", "k0003"],
            "results": [{"source_id": "k0003"}],
        },
        {
            "expected_legacy_ids": ["k0004"],
            "results": [{"source_id": "k0005"}],
        },
    ]

    summary = baseline.calculate_summary(question_results)

    assert summary == {
        "total_questions": 3,
        "expected_hit_count": 2,
        "expected_hit_rate": pytest.approx(2 / 3),
    }


def test_embedding_metadata_excludes_credentials() -> None:
    baseline = _baseline_module()

    class FakeSettings:
        EMBEDDING_API_KEY = "do-not-export"
        EMBEDDING_BASE_URL = "https://user:password@example.test/v1"
        EMBEDDING_MODEL = "embedding-test"
        EMBEDDING_DIMENSIONS = 1024
        EMBEDDING_BATCH_SIZE = 10
        EMBEDDING_TIMEOUT = 60
        EMBEDDING_NORMALIZE = True
        USE_INNER_PRODUCT = True

    metadata = baseline.build_embedding_metadata(FakeSettings())
    serialized = json.dumps(metadata).lower()

    assert metadata == {
        "model": "embedding-test",
        "dimensions": 1024,
        "batch_size": 10,
        "timeout_seconds": 60,
        "normalize": True,
        "similarity": "inner_product",
        "provider_origin_sha256": hashlib.sha256(
            b"https://example.test"
        ).hexdigest(),
    }
    assert "api_key" not in serialized
    assert "base_url" not in serialized
    assert "password" not in serialized


def test_cli_writes_safe_baseline_with_injected_retriever(tmp_path: Path) -> None:
    baseline = _baseline_module()
    output_path = tmp_path / "baseline.json"
    timestamps = iter(
        [
            datetime(2026, 7, 29, 8, 0, tzinfo=timezone.utc),
            datetime(2026, 7, 29, 8, 1, tzinfo=timezone.utc),
        ]
    )
    calls: list[tuple[str, int]] = []

    def fake_retrieve(question: str, top_k: int) -> list[dict]:
        calls.append((question, top_k))
        return [
            {
                "chunk_id": "k0001_chunk_0",
                "source_id": "k0001",
                "title": "任意角及其终边关系",
                "score": 0.95,
                "content": "不会进入输出",
            }
        ]

    exit_code = baseline.main(
        ["--fixture", str(FIXTURE_PATH), "--output", str(output_path)],
        retrieve_fn=fake_retrieve,
        now_fn=lambda: next(timestamps),
    )

    document = json.loads(output_path.read_text(encoding="utf-8"))
    serialized = json.dumps(document, ensure_ascii=False).lower()

    assert exit_code == 0
    assert len(calls) == 26
    assert all(top_k == 3 for _, top_k in calls)
    assert document["started_at"] == "2026-07-29T08:00:00Z"
    assert document["finished_at"] == "2026-07-29T08:01:00Z"
    assert document["summary"]["total_questions"] == 26
    assert document["summary"]["expected_hit_count"] == 1
    assert set(document["artifacts"]) == {"faiss_index", "id_map", "chunk_file"}
    assert all(len(item["sha256"]) == 64 for item in document["artifacts"].values())
    for sensitive_key in ("api_key", "base_url", "authorization", "password", "secret", "token"):
        assert sensitive_key not in serialized


def test_default_capture_boundary_uses_new_provider_and_legacy_adapter() -> None:
    baseline = _baseline_module()
    source = inspect.getsource(baseline)

    assert "app.services.retriever" not in source
    assert "app.services.embedding_service" not in source
    assert "LegacyFaissRetriever" in source
    assert "OpenAIEmbeddingProvider" in source


def test_default_capture_batches_vectors_and_closes_provider() -> None:
    baseline = _baseline_module()
    fixture = {
        "top_k": 3,
        "questions": [
            {"question": "问题一"},
            {"question": "问题二"},
        ],
    }
    vectors = [
        [1.0] + [0.0] * 1023,
        [2.0] + [0.0] * 1023,
    ]
    provider_calls: list[list[str]] = []
    search_calls: list[tuple[list[float], int]] = []

    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False

        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            provider_calls.append(list(texts))
            return vectors

        async def aclose(self) -> None:
            self.closed = True

    class FakeLegacy:
        def search_vector(self, vector: list[float], *, top_k: int) -> list[str]:
            search_calls.append((vector, top_k))
            return [f"k{len(search_calls):04d}"]

    provider = FakeProvider()
    retrieve_fn = baseline.build_default_retrieve_fn(
        fixture,
        provider=provider,
        legacy_retriever=FakeLegacy(),
    )

    assert provider.closed is True
    assert provider_calls == [["问题一", "问题二"]]
    assert search_calls == [(vectors[0], 3), (vectors[1], 3)]
    assert retrieve_fn("问题一", 3) == [{"source_id": "k0001"}]
    assert retrieve_fn("问题二", 3) == [{"source_id": "k0002"}]
    with pytest.raises(RuntimeError, match="次数"):
        retrieve_fn("问题三", 3)


def test_default_capture_validates_legacy_before_opening_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _baseline_module()
    provider_constructions: list[bool] = []

    class FakeProvider:
        def __init__(self) -> None:
            provider_constructions.append(True)

    class InvalidLegacy:
        def __init__(self, **kwargs: object) -> None:
            raise ValueError("invalid legacy artifacts")

    monkeypatch.setattr(baseline, "OpenAIEmbeddingProvider", FakeProvider)
    monkeypatch.setattr(baseline, "LegacyFaissRetriever", InvalidLegacy)

    with pytest.raises(ValueError, match="invalid legacy"):
        baseline.build_default_retrieve_fn(
            {"top_k": 3, "questions": [{"question": "问题"}]}
        )

    assert provider_constructions == []
