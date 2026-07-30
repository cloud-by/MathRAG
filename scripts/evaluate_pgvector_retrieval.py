"""使用同一批查询向量对账 legacy FAISS 与 pgvector 精确检索。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.core.config import settings
from app.infrastructure.database.session import dispose_engine, get_session_factory
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from app.modules.knowledge.repository import KnowledgeRepository
from scripts.capture_retrieval_baseline import (
    load_fixture,
    load_legacy_ids,
    sha256_file,
    sha256_normalized_utf8_text,
    validate_fixture,
)
from scripts.legacy_faiss_retriever import LegacyFaissRetriever


SCHEMA_VERSION = "1.0"
FIXED_QUESTION_COUNT = 26
FIXED_TOP_K = 3
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
GIT_SHA_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
QUESTION_ID_PATTERN = re.compile(r"rq-\d{4}")
SAFE_QUESTION_FIELDS = frozenset(
    {
        "question_id",
        "expected_legacy_ids",
        "legacy_source_ids",
        "pgvector_source_ids",
        "pgvector_latency_ms",
        "expected_hit",
        "top_k_overlap",
        "top_k",
    }
)
FORBIDDEN_SERIALIZED_MARKERS = (
    "api_key",
    "base_url",
    "authorization",
    "password",
    "secret",
    "token",
    "正文",
    "://",
)


class EvaluationInputError(ValueError):
    """对账输入、冻结工件或配置不满足严格契约。"""


class EvaluationThresholdError(RuntimeError):
    """真实检索指标未达到 M3 切换门槛。"""


@dataclass(frozen=True)
class RetrievalMetrics:
    """FAISS/pgvector 对账的聚合指标。"""

    total_questions: int
    expected_hit_count: int
    expected_hit_rate: float
    average_top_k_overlap: float
    pgvector_p50_ms: float
    pgvector_p95_ms: float


def _percentile(values: Sequence[float], percentile: float) -> float:
    """使用 nearest-rank 定义计算百分位，0 分位固定返回最小值。"""
    try:
        ordered = sorted(float(value) for value in values)
        requested = float(percentile)
    except Exception:
        raise EvaluationInputError("百分位输入必须是有限数值") from None
    if not ordered or not all(math.isfinite(value) for value in ordered):
        raise EvaluationInputError("百分位输入必须是非空有限数值数组")
    if not math.isfinite(requested) or not 0.0 <= requested <= 100.0:
        raise EvaluationInputError("百分位必须介于 0 和 100 之间")
    if requested == 0.0:
        return ordered[0]
    rank = math.ceil((requested / 100.0) * len(ordered))
    return ordered[rank - 1]


def _validated_source_ids(value: object, field_name: str) -> list[str]:
    """验证并复制 legacy source ID 数组。"""
    if type(value) is not list:
        raise EvaluationInputError(f"{field_name} 必须是字符串数组")
    output: list[str] = []
    for item in value:
        if type(item) is not str or not item or item != item.strip():
            raise EvaluationInputError(f"{field_name} 包含无效 source ID")
        if item not in output:
            output.append(item)
    return output


def _validated_metric_row(row: Mapping[str, object]) -> tuple[
    list[str], list[str], list[str], float, int
]:
    """提取计算指标所需的严格字段。"""
    if not isinstance(row, Mapping):
        raise EvaluationInputError("问题结果必须是对象")
    expected = _validated_source_ids(
        row.get("expected_legacy_ids"), "expected_legacy_ids"
    )
    legacy = _validated_source_ids(row.get("legacy_source_ids"), "legacy_source_ids")
    pgvector = _validated_source_ids(
        row.get("pgvector_source_ids"), "pgvector_source_ids"
    )
    top_k = row.get("top_k")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise EvaluationInputError("每题 top_k 必须是 1 到 10 的整数")
    if len(legacy) > top_k or len(pgvector) > top_k:
        raise EvaluationInputError("检索 source ID 数量不能超过 top_k")
    try:
        latency_ms = float(row.get("pgvector_latency_ms"))
    except Exception:
        raise EvaluationInputError("pgvector_latency_ms 必须是有限非负数") from None
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise EvaluationInputError("pgvector_latency_ms 必须是有限非负数")
    return expected, legacy, pgvector, latency_ms, top_k


def calculate_metrics(
    question_results: Sequence[Mapping[str, object]],
) -> RetrievalMetrics:
    """按 pgvector 期望命中和每题 Top-K 集合交集计算纯指标。"""
    rows = list(question_results)
    if not rows:
        return RetrievalMetrics(0, 0, 0.0, 0.0, 0.0, 0.0)

    expected_hit_count = 0
    overlaps: list[float] = []
    latencies: list[float] = []
    for row in rows:
        expected, legacy, pgvector, latency_ms, top_k = _validated_metric_row(row)
        if set(expected) & set(pgvector):
            expected_hit_count += 1
        overlaps.append(len(set(legacy) & set(pgvector)) / top_k)
        latencies.append(latency_ms)

    total_questions = len(rows)
    return RetrievalMetrics(
        total_questions=total_questions,
        expected_hit_count=expected_hit_count,
        expected_hit_rate=expected_hit_count / total_questions,
        average_top_k_overlap=math.fsum(overlaps) / total_questions,
        pgvector_p50_ms=_percentile(latencies, 50),
        pgvector_p95_ms=_percentile(latencies, 95),
    )


def assert_thresholds(metrics: RetrievalMetrics) -> None:
    """要求真实 26 题对账满足冻结的 M3 门槛；等于门槛时通过。"""
    if not isinstance(metrics, RetrievalMetrics):
        raise EvaluationThresholdError("检索指标类型无效")
    numeric_metrics = (
        metrics.expected_hit_rate,
        metrics.average_top_k_overlap,
        metrics.pgvector_p50_ms,
        metrics.pgvector_p95_ms,
    )
    if not all(math.isfinite(value) for value in numeric_metrics):
        raise EvaluationThresholdError("检索指标必须是有限数值")
    if not (
        0.0 <= metrics.expected_hit_rate <= 1.0
        and 0.0 <= metrics.average_top_k_overlap <= 1.0
        and metrics.pgvector_p50_ms >= 0.0
        and metrics.pgvector_p95_ms >= 0.0
    ):
        raise EvaluationThresholdError("检索指标超出有效范围")
    if metrics.total_questions != FIXED_QUESTION_COUNT:
        raise EvaluationThresholdError("固定题集必须包含 26 题")
    if metrics.expected_hit_rate < 0.90:
        raise EvaluationThresholdError("pgvector Top-3 期望命中率低于 90%")
    if metrics.average_top_k_overlap < 0.80:
        raise EvaluationThresholdError("FAISS/pgvector Top-3 平均重合率低于 80%")
    if metrics.pgvector_p95_ms > 100.0:
        raise EvaluationThresholdError("pgvector 精确检索 P95 超过 100 ms")


def hash_provider_origin(base_url: str) -> str:
    """规范化 HTTP(S) origin 后仅返回 SHA-256，不返回 URL 本身。"""
    if type(base_url) is not str or not base_url.strip():
        raise EvaluationInputError("Embedding Provider URL 配置无效")
    try:
        parsed = urlsplit(base_url.strip())
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        raise EvaluationInputError("Embedding Provider URL 配置无效") from None
    if scheme not in {"http", "https"} or not host:
        raise EvaluationInputError("Embedding Provider URL 配置无效")
    if any(character.isspace() for character in host) or any(
        character in host for character in "/\\"
    ):
        raise EvaluationInputError("Embedding Provider URL 配置无效")
    try:
        canonical_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise EvaluationInputError("Embedding Provider URL 配置无效") from None
    if ":" in canonical_host:
        canonical_host = f"[{canonical_host}]"
    default_port = 443 if scheme == "https" else 80
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    origin = f"{scheme}://{canonical_host}{port_suffix}"
    return hashlib.sha256(origin.encode("utf-8")).hexdigest()


def _format_utc(value: datetime) -> str:
    """稳定输出带 Z 后缀的 UTC ISO-8601 时间。"""
    if not isinstance(value, datetime):
        raise EvaluationInputError("generated_at 必须是 datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validated_sha256(value: object) -> str:
    """要求输入摘要是完整的 64 位十六进制 SHA-256。"""
    if type(value) is not str or not SHA256_PATTERN.fullmatch(value):
        raise EvaluationInputError("输入摘要必须是 64 位 SHA-256")
    return value.lower()


def _validated_git_sha(value: object) -> str:
    """要求 Git 提交摘要为完整 SHA-1 或 SHA-256。"""
    if type(value) is not str or not GIT_SHA_PATTERN.fullmatch(value):
        raise EvaluationInputError("git_sha 格式无效")
    return value.lower()


def _safe_question(question: Mapping[str, object]) -> dict[str, object]:
    """只保留可审计 ID、时延和布尔指标，拒绝额外载荷。"""
    if not isinstance(question, Mapping) or set(question) != SAFE_QUESTION_FIELDS:
        raise EvaluationInputError("artifact 问题结果字段不符合白名单")
    question_id = question.get("question_id")
    if type(question_id) is not str or not QUESTION_ID_PATTERN.fullmatch(question_id):
        raise EvaluationInputError("artifact question_id 格式无效")
    expected, legacy, pgvector, latency_ms, top_k = _validated_metric_row(question)
    expected_hit = question.get("expected_hit")
    if type(expected_hit) is not bool or expected_hit != bool(set(expected) & set(pgvector)):
        raise EvaluationInputError("artifact expected_hit 与 source ID 不一致")
    try:
        top_k_overlap = float(question.get("top_k_overlap"))
    except Exception:
        raise EvaluationInputError("artifact top_k_overlap 无效") from None
    calculated_overlap = len(set(legacy) & set(pgvector)) / top_k
    if (
        not math.isfinite(top_k_overlap)
        or not 0.0 <= top_k_overlap <= 1.0
        or not math.isclose(top_k_overlap, calculated_overlap, abs_tol=1e-12)
    ):
        raise EvaluationInputError("artifact top_k_overlap 与 source ID 不一致")
    return {
        "question_id": question_id,
        "expected_legacy_ids": expected,
        "legacy_source_ids": legacy,
        "pgvector_source_ids": pgvector,
        "pgvector_latency_ms": latency_ms,
        "expected_hit": expected_hit,
        "top_k_overlap": top_k_overlap,
        "top_k": top_k,
    }


def build_artifact(
    *,
    metrics: RetrievalMetrics,
    questions: Sequence[Mapping[str, object]],
    git_sha: str,
    fixture_sha256: str,
    seed_sha256: str,
    faiss_sha256: str,
    id_map_sha256: str,
    embedding_model: str,
    dimensions: int,
    provider_origin_sha256: str,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """构造 schema 1.0 的脱敏对账 artifact。"""
    if not isinstance(metrics, RetrievalMetrics):
        raise EvaluationInputError("metrics 类型无效")
    if type(embedding_model) is not str or not embedding_model.strip():
        raise EvaluationInputError("embedding_model 必须是非空字符串")
    if type(dimensions) is not int or dimensions != 1024:
        raise EvaluationInputError("Embedding 维度必须为 1024")

    safe_questions = [_safe_question(question) for question in questions]
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _format_utc(generated_at or datetime.now(timezone.utc)),
        "git_sha": _validated_git_sha(git_sha),
        "inputs": {
            "fixture_sha256": _validated_sha256(fixture_sha256),
            "seed_sha256": _validated_sha256(seed_sha256),
            "faiss_sha256": _validated_sha256(faiss_sha256),
            "id_map_sha256": _validated_sha256(id_map_sha256),
        },
        "embedding": {
            "model": embedding_model.strip(),
            "dimensions": dimensions,
            "provider_origin_sha256": _validated_sha256(
                provider_origin_sha256
            ),
        },
        "metrics": asdict(metrics),
        "questions": safe_questions,
    }
    serialized = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ).lower()
    if any(marker in serialized for marker in FORBIDDEN_SERIALIZED_MARKERS):
        raise EvaluationInputError("artifact 包含禁止序列化的敏感字段或载荷")
    return document


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    """在目标目录内完成 UTF-8 JSON 临时写入和原子替换。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file_obj:
            json.dump(
                document,
                file_obj,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def write_success_artifact(
    output_path: str | Path,
    *,
    metrics: RetrievalMetrics,
    questions: Sequence[Mapping[str, object]],
    git_sha: str,
    fixture_sha256: str,
    seed_sha256: str,
    faiss_sha256: str,
    id_map_sha256: str,
    embedding_model: str,
    dimensions: int,
    provider_origin_sha256: str,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """仅在门槛通过后构造并原子写入成功 artifact。"""
    assert_thresholds(metrics)
    document = build_artifact(
        metrics=metrics,
        questions=questions,
        git_sha=git_sha,
        fixture_sha256=fixture_sha256,
        seed_sha256=seed_sha256,
        faiss_sha256=faiss_sha256,
        id_map_sha256=id_map_sha256,
        embedding_model=embedding_model,
        dimensions=dimensions,
        provider_origin_sha256=provider_origin_sha256,
        generated_at=generated_at,
    )
    _atomic_write_json(Path(output_path), document)
    return document


def _validated_questions(
    questions: Sequence[Mapping[str, object]],
) -> list[tuple[str, str, list[str]]]:
    """从已验证 fixture 提取恰好 26 题的运行时字段。"""
    selected = list(questions)
    if len(selected) != FIXED_QUESTION_COUNT:
        raise EvaluationInputError("固定题集必须包含 26 题")
    output: list[tuple[str, str, list[str]]] = []
    seen_ids: set[str] = set()
    for question in selected:
        if not isinstance(question, Mapping):
            raise EvaluationInputError("固定题集问题必须是对象")
        question_id = question.get("id")
        text = question.get("question")
        if type(question_id) is not str or not QUESTION_ID_PATTERN.fullmatch(question_id):
            raise EvaluationInputError("固定题集 question ID 无效")
        if question_id in seen_ids:
            raise EvaluationInputError("固定题集包含重复 question ID")
        seen_ids.add(question_id)
        if type(text) is not str or not text.strip():
            raise EvaluationInputError("固定题集 question 不能为空")
        expected = _validated_source_ids(
            question.get("expected_legacy_ids"), "expected_legacy_ids"
        )
        if not expected:
            raise EvaluationInputError("expected_legacy_ids 不能为空")
        output.append((question_id, text, expected))
    return output


async def evaluate_questions(
    *,
    questions: Sequence[Mapping[str, object]],
    top_k: int,
    provider: EmbeddingProvider,
    legacy_retriever: LegacyFaissRetriever,
    session_factory: Callable[[], Any],
    repository_factory: Callable[[Any], Any] = KnowledgeRepository,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[list[dict[str, object]], RetrievalMetrics]:
    """批量向量化后逐题顺序对账；计时只包 Repository SQL await。"""
    if type(top_k) is not int or top_k != FIXED_TOP_K:
        raise EvaluationInputError("固定题集 top_k 必须为 3")
    prepared_questions = _validated_questions(questions)
    provider_model = getattr(provider, "model", None)
    provider_dimensions = getattr(provider, "dimensions", None)
    if (
        type(provider_dimensions) is not int
        or provider_dimensions != 1024
        or type(provider_model) is not str
        or not provider_model.strip()
    ):
        raise EvaluationInputError("Embedding Provider 配置与固定契约不一致")
    provider_model = provider_model.strip()

    vectors = await provider.embed_texts(
        [question_text for _question_id, question_text, _expected in prepared_questions]
    )
    if type(vectors) is not list or len(vectors) != len(prepared_questions):
        raise EvaluationInputError("Embedding 返回数量与固定题集不一致")

    # 用首题的同一向量预热连接、查询计划和数据页；不调用计时器，也不写入结果。
    async with session_factory() as session:
        repository = repository_factory(session)
        await repository.search_ready_chunks(
            query_vector=vectors[0],
            embedding_model=provider_model,
            limit=top_k,
        )

    rows: list[dict[str, object]] = []
    for (question_id, _question_text, expected), vector in zip(
        prepared_questions, vectors, strict=True
    ):
        legacy_ids = legacy_retriever.search_vector(vector, top_k=top_k)
        legacy_ids = _validated_source_ids(legacy_ids, "legacy_source_ids")

        async with session_factory() as session:
            repository = repository_factory(session)
            started = timer()
            hits = await repository.search_ready_chunks(
                query_vector=vector,
                embedding_model=provider_model,
                limit=top_k,
            )
            finished = timer()
        try:
            latency_ms = (float(finished) - float(started)) * 1000.0
        except Exception:
            raise EvaluationInputError("Repository 计时器返回无效数值") from None
        if not math.isfinite(latency_ms) or latency_ms < 0.0:
            raise EvaluationInputError("Repository 计时器返回无效数值")

        pgvector_ids: list[str] = []
        for hit in hits:
            source_id = getattr(hit, "legacy_source_id", None)
            if type(source_id) is not str:
                raise EvaluationInputError("pgvector 命中缺少 legacy_source_id")
            pgvector_ids.append(source_id)
        pgvector_ids = _validated_source_ids(
            pgvector_ids, "pgvector_source_ids"
        )
        if len(pgvector_ids) > top_k:
            raise EvaluationInputError("pgvector 命中数量超过 top_k")

        expected_hit = bool(set(expected) & set(pgvector_ids))
        overlap = len(set(legacy_ids) & set(pgvector_ids)) / top_k
        rows.append(
            {
                "question_id": question_id,
                "expected_legacy_ids": expected,
                "legacy_source_ids": legacy_ids,
                "pgvector_source_ids": pgvector_ids,
                "pgvector_latency_ms": latency_ms,
                "expected_hit": expected_hit,
                "top_k_overlap": overlap,
                "top_k": top_k,
            }
        )
    return rows, calculate_metrics(rows)


async def run_evaluation(
    *,
    questions: Sequence[Mapping[str, object]],
    top_k: int,
    provider: EmbeddingProvider,
    legacy_retriever: LegacyFaissRetriever,
    session_factory: Callable[[], Any],
    repository_factory: Callable[[Any], Any] = KnowledgeRepository,
    timer: Callable[[], float] = time.perf_counter,
    dispose_database: Callable[[], Any] = dispose_engine,
) -> tuple[list[dict[str, object]], RetrievalMetrics]:
    """运行对账，并在成功或失败时都关闭 Provider 与数据库连接池。"""
    business_error: BaseException | None = None
    try:
        return await evaluate_questions(
            questions=questions,
            top_k=top_k,
            provider=provider,
            legacy_retriever=legacy_retriever,
            session_factory=session_factory,
            repository_factory=repository_factory,
            timer=timer,
        )
    except BaseException as exc:
        business_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            await provider.aclose()
        except BaseException as exc:
            cleanup_error = exc
        try:
            await dispose_database()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if business_error is None and cleanup_error is not None:
            raise cleanup_error


def _load_evaluation_fixture(path: Path) -> tuple[dict[str, Any], str]:
    """加载现有 fixture，并同时按规范化 seed SHA 严格验证。"""
    fixture = load_fixture(path)
    seed_sha256 = sha256_normalized_utf8_text(settings.RAW_KB_PATH)
    try:
        validate_fixture(
            fixture,
            valid_legacy_ids=load_legacy_ids(settings.RAW_KB_PATH),
            dataset_sha256=seed_sha256,
        )
    except ValueError as exc:
        raise EvaluationInputError("固定题集验证失败") from exc
    questions = fixture.get("questions")
    if not isinstance(questions, list) or len(questions) != FIXED_QUESTION_COUNT:
        raise EvaluationInputError("固定题集必须包含 26 题")
    if fixture.get("top_k") != FIXED_TOP_K:
        raise EvaluationInputError("固定题集 top_k 必须为 3")
    return fixture, seed_sha256


def _git_sha() -> str:
    """读取当前工作树 HEAD 的完整提交摘要。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError):
        raise EvaluationInputError("无法读取 Git 提交摘要") from None
    return _validated_git_sha(completed.stdout.strip())


async def _run_cli(fixture_path: Path, output_path: Path) -> RetrievalMetrics:
    """执行真实 Provider/数据库对账，并仅在门槛通过后写 artifact。"""
    fixture, seed_sha256 = _load_evaluation_fixture(fixture_path)
    provider_origin_sha256 = hash_provider_origin(settings.EMBEDDING_BASE_URL)
    legacy_retriever = LegacyFaissRetriever(
        index_path=settings.FAISS_INDEX_PATH,
        id_map_path=settings.ID_MAP_PATH,
    )
    provider = OpenAIEmbeddingProvider()
    rows, metrics = await run_evaluation(
        questions=fixture["questions"],
        top_k=fixture["top_k"],
        provider=provider,
        legacy_retriever=legacy_retriever,
        session_factory=get_session_factory(),
    )
    write_success_artifact(
        output_path,
        metrics=metrics,
        questions=rows,
        git_sha=_git_sha(),
        fixture_sha256=sha256_file(fixture_path),
        seed_sha256=seed_sha256,
        faiss_sha256=sha256_file(settings.FAISS_INDEX_PATH),
        id_map_sha256=sha256_file(settings.ID_MAP_PATH),
        embedding_model=provider.model,
        dimensions=provider.dimensions,
        provider_origin_sha256=provider_origin_sha256,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    """构造真实对账 CLI 参数。"""
    parser = argparse.ArgumentParser(description="对账 FAISS 与 pgvector Top-3 检索")
    parser.add_argument("--fixture", required=True, type=Path, help="固定问题集 JSON")
    parser.add_argument("--output", required=True, type=Path, help="成功 artifact JSON")
    return parser


def _write_error(error: str, exception: BaseException) -> None:
    """只输出稳定错误分类和类型，不回显异常消息、URL 或密钥。"""
    sys.stderr.write(
        json.dumps(
            {"detail": type(exception).__name__, "error": error},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """运行真实对账，阈值失败固定退出 4 且不写成功 artifact。"""
    args = build_parser().parse_args(argv)
    try:
        metrics = asyncio.run(_run_cli(args.fixture, args.output))
    except EvaluationThresholdError as exc:
        _write_error("threshold_failed", exc)
        return 4
    except (EvaluationInputError, EmbeddingInputError, FileNotFoundError) as exc:
        _write_error("invalid_input", exc)
        return 2
    except (EmbeddingUnavailableError, EmbeddingResponseError) as exc:
        _write_error("embedding_unavailable", exc)
        return 3
    except Exception as exc:
        _write_error("evaluation_failed", exc)
        return 1
    sys.stdout.write(json.dumps(asdict(metrics), ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
