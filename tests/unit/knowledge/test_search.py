"""知识向量检索 DTO、合并和输入校验的单元测试。"""

from __future__ import annotations

import asyncio
import math
from copy import deepcopy
from uuid import UUID

import pytest

from app.modules.knowledge.errors import KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.search import (
    KnowledgeSearchHit,
    merge_search_hits,
    search_hit_from_row,
)


CHUNK_A = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_B = UUID("00000000-0000-0000-0000-000000000002")
CHUNK_C = UUID("00000000-0000-0000-0000-000000000003")


def make_hit(
    database_chunk_id: UUID = CHUNK_A,
    *,
    legacy_chunk_id: str = "legacy-chunk-a",
    distance: float = 0.2,
    metadata: dict[str, object] | None = None,
) -> KnowledgeSearchHit:
    """构造一个字段完整的检索命中。"""
    return KnowledgeSearchHit(
        database_chunk_id=database_chunk_id,
        legacy_chunk_id=legacy_chunk_id,
        source_id="legacy-source-a",
        category="algebra",
        title="一元一次方程",
        keywords=("方程", "移项"),
        content="方程知识正文",
        example="2x + 1 = 5",
        steps=("移项", "求解"),
        difficulty="easy",
        answer_context="回答上下文",
        retrieval_text="检索文本",
        source_line=7,
        metadata=metadata if metadata is not None else {"nested": {"labels": ["初始"]}},
        distance=distance,
    )


def make_row() -> tuple[KnowledgeChunk, KnowledgeItem, float]:
    """构造与 M2 导入结果一致的 ORM 行。"""
    item = KnowledgeItem(
        id=UUID("10000000-0000-0000-0000-000000000001"),
        legacy_id="source-from-model",
        category="model-category",
        title="模型标题",
        keywords=["模型关键词"],
        content="模型正文",
        example="模型示例",
        steps=["模型步骤"],
        difficulty="medium",
        visibility="public",
        status="ready",
    )
    chunk = KnowledgeChunk(
        id=CHUNK_A,
        knowledge_item_id=item.id,
        chunk_index=0,
        retrieval_text="模型检索文本",
        answer_context="模型回答上下文",
        embedding=[1.0, *([0.0] * 1023)],
        embedding_model="embedding-test",
        metadata_={
            "legacy_chunk_id": "legacy-chunk-from-metadata",
            "legacy_source_id": "source-from-metadata",
            "source_line": 11,
            "category": "不能覆盖模型列",
            "nested": {"labels": ["初始"]},
        },
        status="ready",
    )
    return chunk, item, 0.125


def test_hit_copies_metadata_and_builds_legacy_reference() -> None:
    """构造与引用映射均不得共享 metadata，且引用字段兼容旧检索结果。"""
    metadata = {"nested": {"labels": ["初始"]}}
    hit = make_hit(metadata=metadata)
    metadata["nested"]["labels"].append("外部篡改")  # type: ignore[index]

    reference = hit.to_reference(rank=2)
    reference["metadata"]["nested"]["labels"].append("引用篡改")  # type: ignore[index]
    reference["keywords"].append("引用篡改")  # type: ignore[union-attr]
    reference["steps"].append("引用篡改")  # type: ignore[union-attr]

    assert hit.score == pytest.approx(0.8)
    assert hit.metadata == {"nested": {"labels": ["初始"]}}
    assert hit.keywords == ("方程", "移项")
    assert hit.steps == ("移项", "求解")
    assert reference == {
        "rank": 2,
        "score": pytest.approx(0.8),
        "index": None,
        "chunk_id": "legacy-chunk-a",
        "source_id": "legacy-source-a",
        "category": "algebra",
        "title": "一元一次方程",
        "keywords": ["方程", "移项", "引用篡改"],
        "content": "方程知识正文",
        "example": "2x + 1 = 5",
        "steps": ["移项", "求解", "引用篡改"],
        "difficulty": "easy",
        "answer_context": "回答上下文",
        "retrieval_text": "检索文本",
        "source_line": 7,
        "metadata": {"nested": {"labels": ["初始", "引用篡改"]}},
    }


@pytest.mark.parametrize("rank", [0, -1, True, 1.5, "1"])
def test_to_reference_rejects_invalid_rank(rank: object) -> None:
    """引用排名必须是从 1 开始的真整数。"""
    with pytest.raises(KnowledgeSearchError):
        make_hit().to_reference(rank)  # type: ignore[arg-type]


def test_merge_deduplicates_database_uuid_and_keeps_highest_score() -> None:
    """跨查询重复项按数据库 UUID 合并，保留分数最高的命中。"""
    lower = make_hit(CHUNK_A, legacy_chunk_id="duplicate", distance=0.4)
    higher = make_hit(CHUNK_A, legacy_chunk_id="duplicate", distance=0.1)
    same_legacy_different_database_id = make_hit(
        CHUNK_B,
        legacy_chunk_id="duplicate",
        distance=0.2,
    )

    merged = merge_search_hits(
        [[lower, same_legacy_different_database_id], [higher]],
        top_k=3,
    )

    assert [(hit.database_chunk_id, hit.distance) for hit in merged] == [
        (CHUNK_A, 0.1),
        (CHUNK_B, 0.2),
    ]


def test_merge_uses_uuid_as_stable_tie_break_and_applies_top_k() -> None:
    """同分命中按数据库 UUID 字符串稳定排序，再截取 Top-K。"""
    merged = merge_search_hits(
        [[make_hit(CHUNK_C, distance=0.25), make_hit(CHUNK_A, distance=0.25)],
         [make_hit(CHUNK_B, distance=0.25)]],
        top_k=2,
    )

    assert [hit.database_chunk_id for hit in merged] == [CHUNK_A, CHUNK_B]
    assert merge_search_hits([], top_k=2) == []
    assert merge_search_hits([[]], top_k=2) == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "1"])
def test_merge_rejects_invalid_top_k(top_k: object) -> None:
    """Top-K 必须是正的真整数。"""
    with pytest.raises(KnowledgeSearchError):
        merge_search_hits([], top_k=top_k)  # type: ignore[arg-type]


def test_search_hit_from_row_uses_model_columns_and_removes_audit_metadata() -> None:
    """行映射以真实模型列为准，并从返回 metadata 移除 M2 审计键。"""
    chunk, item, distance = make_row()
    original_metadata = deepcopy(chunk.metadata_)

    hit = search_hit_from_row(chunk, item, distance)
    chunk.metadata_["nested"]["labels"].append("ORM 篡改")  # type: ignore[index]

    assert hit.database_chunk_id == CHUNK_A
    assert hit.legacy_chunk_id == "legacy-chunk-from-metadata"
    assert hit.source_id == "source-from-model"
    assert hit.category == "model-category"
    assert hit.title == "模型标题"
    assert hit.keywords == ("模型关键词",)
    assert hit.content == "模型正文"
    assert hit.example == "模型示例"
    assert hit.steps == ("模型步骤",)
    assert hit.difficulty == "medium"
    assert hit.retrieval_text == "模型检索文本"
    assert hit.answer_context == "模型回答上下文"
    assert hit.source_line == 11
    assert hit.metadata == {
        "category": "不能覆盖模型列",
        "nested": {"labels": ["初始"]},
    }
    assert original_metadata == {
        "legacy_chunk_id": "legacy-chunk-from-metadata",
        "legacy_source_id": "source-from-metadata",
        "source_line": 11,
        "category": "不能覆盖模型列",
        "nested": {"labels": ["初始"]},
    }
    assert hit.distance == pytest.approx(0.125)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda chunk, item: setattr(chunk, "metadata_", []),
        lambda chunk, item: setattr(item, "keywords", ("不是列表",)),
        lambda chunk, item: setattr(item, "steps", [1]),
        lambda chunk, item: chunk.metadata_.pop("legacy_chunk_id"),
        lambda chunk, item: (
            setattr(item, "legacy_id", None),
            chunk.metadata_.pop("legacy_source_id"),
        ),
        lambda chunk, item: chunk.metadata_.update({"source_line": "11"}),
    ],
)
def test_search_hit_from_row_rejects_malformed_json_without_leaking_values(
    mutate: object,
) -> None:
    """损坏持久化行须安全失败，错误只能携带数据库分块 UUID。"""
    chunk, item, distance = make_row()
    item.content = "https://example.invalid/private?token=secret"
    chunk.metadata_["secret"] = "metadata-secret"
    mutate(chunk, item)  # type: ignore[operator]

    with pytest.raises(KnowledgeSearchError) as captured:
        search_hit_from_row(chunk, item, distance)

    message = str(captured.value)
    assert str(CHUNK_A) in message
    assert "metadata-secret" not in message
    assert "example.invalid" not in message
    assert "token=secret" not in message


@pytest.mark.parametrize("distance", [math.nan, math.inf, -math.inf, "not-a-number"])
def test_search_hit_from_row_rejects_invalid_distance_safely(distance: object) -> None:
    """数据库距离必须可转换为有限浮点数。"""
    chunk, item, _distance = make_row()

    with pytest.raises(KnowledgeSearchError) as captured:
        search_hit_from_row(chunk, item, distance)  # type: ignore[arg-type]

    assert str(CHUNK_A) in str(captured.value)
    assert "not-a-number" not in str(captured.value)


class NoExecuteSession:
    """确认无效输入在触发 SQL 前即被拒绝。"""

    def __init__(self) -> None:
        self.execute_called = False

    async def execute(self, statement: object) -> None:
        del statement
        self.execute_called = True
        raise AssertionError("无效输入不应执行 SQL")


@pytest.mark.parametrize(
    ("query_vector", "embedding_model", "limit"),
    [
        ([0.0] * 1023, "embedding-test", 1),
        ([0.0] * 1025, "embedding-test", 1),
        ([0.0] * 1023 + [math.nan], "embedding-test", 1),
        ([0.0] * 1023 + [math.inf], "embedding-test", 1),
        ([0.0] * 1024, "", 1),
        ([0.0] * 1024, "   ", 1),
        ([0.0] * 1024, "embedding-test", 0),
        ([0.0] * 1024, "embedding-test", 11),
        ([0.0] * 1024, "embedding-test", True),
    ],
)
def test_repository_rejects_unsafe_search_inputs_before_sql(
    query_vector: list[float],
    embedding_model: str,
    limit: object,
) -> None:
    """检索向量、模型和在线 limit 必须遵循固定边界。"""
    session = NoExecuteSession()
    repository = KnowledgeRepository(session)  # type: ignore[arg-type]

    async def exercise() -> None:
        with pytest.raises(KnowledgeSearchError):
            await repository.search_ready_chunks(
                query_vector=query_vector,
                embedding_model=embedding_model,
                limit=limit,  # type: ignore[arg-type]
            )

    asyncio.run(exercise())
    assert session.execute_called is False
