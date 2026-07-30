from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.core.config import settings
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from scripts.legacy_faiss_retriever import LegacyFaissRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCHEMA_VERSION = "1.0"
BASELINE_SCHEMA_VERSION = "1.0"
FIXED_TOP_K = 3
FIXTURE_FIELDS = frozenset({"schema_version", "dataset_sha256", "top_k", "questions"})
QUESTION_FIELDS = frozenset({"id", "question", "expected_legacy_ids", "rationale"})
QUESTION_ID_PATTERN = re.compile(r"rq-\d{4}")
CHINESE_CHARACTER_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_normalized_utf8_text(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_fixture(path: str | Path) -> dict[str, Any]:
    fixture_path = Path(path)
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"固定问题集 JSON 解析失败：{exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("固定问题集根节点必须是 JSON 对象")
    return payload


def load_legacy_ids(path: str | Path) -> set[str]:
    legacy_ids: set[str] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"seed 第 {line_no} 行 JSON 解析失败：{exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"seed 第 {line_no} 行必须是 JSON 对象")
        legacy_id = str(row.get("id", "")).strip()
        if not legacy_id:
            raise ValueError(f"seed 第 {line_no} 行缺少 id")
        if legacy_id in legacy_ids:
            raise ValueError(f"seed 中存在重复 id：{legacy_id}")
        legacy_ids.add(legacy_id)
    return legacy_ids


def validate_fixture(
    fixture: Mapping[str, Any],
    *,
    valid_legacy_ids: set[str],
    dataset_sha256: str,
) -> None:
    for field in FIXTURE_FIELDS:
        if field not in fixture:
            raise ValueError(f"固定问题集缺少字段：{field}")
    unknown_fields = sorted(set(fixture) - FIXTURE_FIELDS)
    if unknown_fields:
        raise ValueError(f"固定问题集包含未知字段：{', '.join(unknown_fields)}")

    if fixture["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"不支持的固定问题集 schema_version：{fixture['schema_version']}"
        )
    if fixture["dataset_sha256"] != dataset_sha256:
        raise ValueError(
            "固定问题集 dataset_sha256 与当前 seed 文件不一致："
            f"fixture={fixture['dataset_sha256']}, actual={dataset_sha256}"
        )
    if type(fixture["top_k"]) is not int or fixture["top_k"] != FIXED_TOP_K:
        raise ValueError(f"固定问题集 top_k 必须为 {FIXED_TOP_K}")

    questions = fixture["questions"]
    if not isinstance(questions, list) or len(questions) < 20:
        raise ValueError("固定问题集 questions 必须是至少包含 20 题的数组")

    question_ids: set[str] = set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"固定问题集第 {index} 题必须是 JSON 对象")
        for field in QUESTION_FIELDS:
            if field not in question:
                raise ValueError(f"固定问题集第 {index} 题缺少字段：{field}")
        unknown_fields = sorted(set(question) - QUESTION_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"固定问题集第 {index} 题包含未知字段：{', '.join(unknown_fields)}"
            )

        if not isinstance(question["id"], str):
            raise ValueError(f"固定问题集第 {index} 题 id 必须是字符串")
        question_id = question["id"].strip()
        if not QUESTION_ID_PATTERN.fullmatch(question_id):
            raise ValueError(f"固定问题集第 {index} 题 id 必须符合 rq-0001 格式")
        if question_id in question_ids:
            raise ValueError(f"固定问题集存在重复 id：{question_id}")
        question_ids.add(question_id)

        if not isinstance(question["question"], str) or not question["question"].strip():
            raise ValueError(f"固定问题集第 {index} 题 question 不能为空")
        if not CHINESE_CHARACTER_PATTERN.search(question["question"]):
            raise ValueError(f"固定问题集第 {index} 题 question 必须包含中文")
        if not isinstance(question["rationale"], str) or not question["rationale"].strip():
            raise ValueError(f"固定问题集第 {index} 题 rationale 不能为空")

        expected_ids = question["expected_legacy_ids"]
        if not isinstance(expected_ids, list) or not expected_ids:
            raise ValueError(
                f"固定问题集第 {index} 题 expected_legacy_ids 必须是非空数组"
            )
        if any(not isinstance(item, str) for item in expected_ids):
            raise ValueError(
                f"固定问题集第 {index} 题 expected_legacy_ids 只能包含字符串"
            )
        normalized_expected_ids = [item.strip() for item in expected_ids]
        if any(not item for item in normalized_expected_ids):
            raise ValueError(f"固定问题集第 {index} 题包含空的 expected legacy ID")
        unknown_ids = sorted(set(normalized_expected_ids) - valid_legacy_ids)
        if unknown_ids:
            raise ValueError(
                f"固定问题集第 {index} 题引用未知 legacy ID：{', '.join(unknown_ids)}"
            )


def normalize_results(
    results: Iterable[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    normalized: list[dict[str, Any]] = []
    for rank, result in enumerate(results, start=1):
        if rank > top_k:
            break
        score = float(result.get("score", 0.0))
        if not math.isfinite(score):
            raise ValueError(f"第 {rank} 条检索结果 score 必须是有限数值")
        normalized.append(
            {
                "rank": rank,
                "chunk_id": str(result.get("chunk_id", "")),
                "source_id": str(result.get("source_id", "")),
                "title": str(result.get("title", "")),
                "score": score,
            }
        )
    return normalized


def calculate_summary(question_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_hit_count = 0
    for question_result in question_results:
        expected_ids = {
            str(item) for item in question_result.get("expected_legacy_ids", [])
        }
        retrieved_ids = {
            str(result.get("source_id", ""))
            for result in question_result.get("results", [])
            if isinstance(result, Mapping)
        }
        if expected_ids & retrieved_ids:
            expected_hit_count += 1

    total_questions = len(question_results)
    return {
        "total_questions": total_questions,
        "expected_hit_count": expected_hit_count,
        "expected_hit_rate": (
            expected_hit_count / total_questions if total_questions else 0.0
        ),
    }


def build_embedding_metadata(settings_obj: Any) -> dict[str, Any]:
    parsed_base_url = urlsplit(settings_obj.EMBEDDING_BASE_URL)
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.hostname:
        raise ValueError("EMBEDDING_BASE_URL 必须是有效的 HTTP(S) URL")
    host = parsed_base_url.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 443 if parsed_base_url.scheme == "https" else 80
    port_suffix = (
        f":{parsed_base_url.port}"
        if parsed_base_url.port and parsed_base_url.port != default_port
        else ""
    )
    provider_origin = f"{parsed_base_url.scheme.lower()}://{host}{port_suffix}"

    return {
        "model": settings_obj.EMBEDDING_MODEL,
        "dimensions": settings_obj.EMBEDDING_DIMENSIONS,
        "batch_size": settings_obj.EMBEDDING_BATCH_SIZE,
        "timeout_seconds": settings_obj.EMBEDDING_TIMEOUT,
        "normalize": settings_obj.EMBEDDING_NORMALIZE,
        "similarity": (
            "inner_product" if settings_obj.USE_INNER_PRODUCT else "l2_distance"
        ),
        "provider_origin_sha256": hashlib.sha256(
            provider_origin.encode("utf-8")
        ).hexdigest(),
    }


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _is_expected_hit(question_result: Mapping[str, Any]) -> bool:
    expected_ids = {
        str(item) for item in question_result.get("expected_legacy_ids", [])
    }
    retrieved_ids = {
        str(result.get("source_id", ""))
        for result in question_result.get("results", [])
        if isinstance(result, Mapping)
    }
    return bool(expected_ids & retrieved_ids)


def capture_baseline(
    fixture: Mapping[str, Any],
    *,
    fixture_path: str | Path,
    retrieve_fn: Callable[[str, int], Sequence[Mapping[str, Any]]],
    embedding_metadata: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    now_fn: Callable[[], datetime],
) -> dict[str, Any]:
    started_at = now_fn()
    top_k = int(fixture["top_k"])
    question_results: list[dict[str, Any]] = []

    for question in fixture["questions"]:
        normalized = normalize_results(
            retrieve_fn(question["question"], top_k),
            top_k=top_k,
        )
        question_result = {
            "id": question["id"],
            "question": question["question"],
            "expected_legacy_ids": list(question["expected_legacy_ids"]),
            "rationale": question["rationale"],
            "results": normalized,
        }
        question_result["expected_hit"] = _is_expected_hit(question_result)
        question_results.append(question_result)

    finished_at = now_fn()
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "started_at": _format_utc(started_at),
        "finished_at": _format_utc(finished_at),
        "fixture": {
            "path": _portable_path(fixture_path),
            "schema_version": fixture["schema_version"],
            "dataset_sha256": fixture["dataset_sha256"],
            "top_k": top_k,
        },
        "embedding": dict(embedding_metadata),
        "artifacts": dict(artifacts),
        "summary": calculate_summary(question_results),
        "questions": question_results,
    }


def build_artifact_metadata(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    return {
        name: {
            "path": _portable_path(path),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_default_retrieve_fn(
    fixture: Mapping[str, Any],
    *,
    provider: EmbeddingProvider | None = None,
    legacy_retriever: LegacyFaissRetriever | None = None,
) -> Callable[[str, int], Sequence[Mapping[str, Any]]]:
    """批量生成查询向量并关闭 Provider，再返回顺序只读 FAISS 调用器。"""
    questions = fixture.get("questions")
    top_k = fixture.get("top_k")
    if not isinstance(questions, list) or not questions:
        raise ValueError("固定问题集 questions 必须是非空数组")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("固定问题集 top_k 必须是 1 到 10 的整数")
    question_texts: list[str] = []
    for question in questions:
        if not isinstance(question, Mapping):
            raise ValueError("固定问题集问题必须是对象")
        text = question.get("question")
        if type(text) is not str or not text.strip():
            raise ValueError("固定问题集 question 不能为空")
        question_texts.append(text)

    active_retriever = legacy_retriever or LegacyFaissRetriever(
        index_path=settings.FAISS_INDEX_PATH,
        id_map_path=settings.ID_MAP_PATH,
    )
    active_provider = provider or OpenAIEmbeddingProvider()

    async def embed_once() -> list[list[float]]:
        business_error: BaseException | None = None
        try:
            return await active_provider.embed_texts(question_texts)
        except BaseException as exc:
            business_error = exc
            raise
        finally:
            try:
                await active_provider.aclose()
            except BaseException:
                if business_error is None:
                    raise

    vectors = asyncio.run(embed_once())
    if type(vectors) is not list or len(vectors) != len(question_texts):
        raise ValueError("Embedding 返回数量与固定问题集不一致")

    prepared = deque(
        (
            question_text,
            [
                {"source_id": source_id}
                for source_id in active_retriever.search_vector(vector, top_k=top_k)
            ],
        )
        for question_text, vector in zip(question_texts, vectors, strict=True)
    )

    def retrieve(question: str, requested_top_k: int) -> Sequence[Mapping[str, Any]]:
        if not prepared:
            raise RuntimeError("默认检索调用次数超过固定题集")
        expected_question, results = prepared.popleft()
        if question != expected_question or requested_top_k != top_k:
            raise RuntimeError("默认检索调用顺序或 top_k 与固定题集不一致")
        return results

    return retrieve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="捕获当前 FAISS Top-K 检索基线")
    parser.add_argument("--fixture", required=True, type=Path, help="固定问题集 JSON 路径")
    parser.add_argument("--output", required=True, type=Path, help="基线输出 JSON 路径")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    retrieve_fn: Callable[[str, int], Sequence[Mapping[str, Any]]] | None = None,
    now_fn: Callable[[], datetime] = _utc_now,
) -> int:
    args = build_parser().parse_args(argv)
    fixture = load_fixture(args.fixture)
    dataset_sha256 = sha256_normalized_utf8_text(settings.RAW_KB_PATH)
    validate_fixture(
        fixture,
        valid_legacy_ids=load_legacy_ids(settings.RAW_KB_PATH),
        dataset_sha256=dataset_sha256,
    )

    if retrieve_fn is None:
        retrieve_fn = build_default_retrieve_fn(fixture)

    artifacts = build_artifact_metadata(
        {
            "faiss_index": settings.FAISS_INDEX_PATH,
            "id_map": settings.ID_MAP_PATH,
            "chunk_file": settings.PROCESSED_KB_PATH,
        }
    )
    document = capture_baseline(
        fixture,
        fixture_path=args.fixture,
        retrieve_fn=retrieve_fn,
        embedding_metadata=build_embedding_metadata(settings),
        artifacts=artifacts,
        now_fn=now_fn,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"检索基线已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
