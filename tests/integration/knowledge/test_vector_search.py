"""知识 pgvector 精确余弦检索的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import math
import os
import sys
from collections.abc import Sequence
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.knowledge.errors import KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository
from tests.integration.database_safety import require_test_database_url


MODEL = "integration-embedding-model"
CHUNK_A = UUID("00000000-0000-0000-0000-000000000100")
CHUNK_B = UUID("00000000-0000-0000-0000-000000000200")


def vector(*coordinates: float) -> list[float]:
    """构造固定 1024 维向量。"""
    return [*coordinates, *([0.0] * (1024 - len(coordinates)))]


def make_item_with_chunk(
    *,
    number: int,
    legacy_id: str,
    embedding: Sequence[float],
    embedding_model: str = MODEL,
    visibility: str = "public",
    item_status: str = "ready",
    chunk_status: str = "ready",
) -> KnowledgeItem:
    """创建一个具有固定 UUID 的真实数据库候选项。"""
    item_id = UUID(f"10000000-0000-0000-0000-{number:012d}")
    chunk_id = UUID(f"00000000-0000-0000-0000-{number:012d}")
    item = KnowledgeItem(
        id=item_id,
        legacy_id=legacy_id,
        category="vector-search-test",
        title=f"向量检索条目 {legacy_id}",
        keywords=["向量", legacy_id],
        content=f"{legacy_id} 正文",
        example=f"{legacy_id} 示例",
        steps=[f"{legacy_id} 步骤"],
        difficulty="easy",
        visibility=visibility,
        status=item_status,
    )
    item.chunks.append(
        KnowledgeChunk(
            id=chunk_id,
            chunk_index=0,
            retrieval_text=f"{legacy_id} 检索文本",
            answer_context=f"{legacy_id} 回答上下文",
            embedding=list(embedding),
            embedding_model=embedding_model,
            metadata_={
                "legacy_chunk_id": f"{legacy_id}-chunk-0",
                "legacy_source_id": legacy_id,
                "source_line": number,
                "origin": "integration-test",
            },
            status=chunk_status,
        )
    )
    return item


async def cleanup(session: AsyncSession) -> None:
    """按外键顺序清空专用测试库知识表。"""
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def assert_database_restored(session: AsyncSession) -> None:
    """确认测试库恢复到迁移 0003 且知识表均为空。"""
    revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
    item_count = await session.scalar(select(func.count()).select_from(KnowledgeItem))
    chunk_count = await session.scalar(select(func.count()).select_from(KnowledgeChunk))
    assert revision == "0003_enforce_vector_readiness"
    assert item_count == 0
    assert chunk_count == 0


async def exercise_vector_search(database_url: str) -> None:
    """在真实 pgvector 上验证 SQL 过滤、精确排序、稳定并列和 limit。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    query = vector(1.0)
    try:
        async with session_factory() as session:
            async with session.begin():
                await cleanup(session)
                await assert_database_restored(session)

        candidates = [
            make_item_with_chunk(
                number=100,
                legacy_id="public-ready-a",
                embedding=query,
            ),
            make_item_with_chunk(
                number=200,
                legacy_id="public-ready-b",
                embedding=vector(0.8, 0.6),
            ),
            make_item_with_chunk(
                number=1,
                legacy_id="private-ready",
                embedding=query,
                visibility="private",
            ),
            make_item_with_chunk(
                number=2,
                legacy_id="item-not-ready",
                embedding=query,
                item_status="indexing",
            ),
            make_item_with_chunk(
                number=3,
                legacy_id="chunk-pending",
                embedding=query,
                chunk_status="pending",
            ),
            make_item_with_chunk(
                number=4,
                legacy_id="chunk-failed",
                embedding=query,
                chunk_status="failed",
            ),
            make_item_with_chunk(
                number=5,
                legacy_id="wrong-model-ready",
                embedding=query,
                embedding_model="wrong-model",
            ),
        ]

        async with session_factory() as session:
            async with session.begin():
                session.add_all(candidates)

        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            hits = await repository.search_ready_chunks(
                query_vector=query,
                embedding_model=MODEL,
                limit=10,
            )

            assert [hit.database_chunk_id for hit in hits] == [CHUNK_A, CHUNK_B]
            assert [hit.legacy_chunk_id for hit in hits] == [
                "public-ready-a-chunk-0",
                "public-ready-b-chunk-0",
            ]
            assert [hit.legacy_source_id for hit in hits] == [
                "public-ready-a",
                "public-ready-b",
            ]
            assert hits[0].distance == pytest.approx(0.0, abs=1e-7)
            assert hits[1].distance == pytest.approx(0.2, abs=1e-6)
            assert [hit.metadata for hit in hits] == [
                {"origin": "integration-test"},
                {"origin": "integration-test"},
            ]

            limited = await repository.search_ready_chunks(
                query_vector=query,
                embedding_model=MODEL,
                limit=1,
            )
            assert [hit.database_chunk_id for hit in limited] == [CHUNK_A]

            candidate_b = candidates[1].chunks[0]
            candidate_b.embedding = query
            session.add(candidate_b)
            await session.flush()
            tied = await repository.search_ready_chunks(
                query_vector=query,
                embedding_model=MODEL,
                limit=2,
            )
            assert [hit.database_chunk_id for hit in tied] == [CHUNK_A, CHUNK_B]
            assert [hit.distance for hit in tied] == pytest.approx([0.0, 0.0], abs=1e-7)

            for invalid_vector in ([0.0] * 1023, [*query[:-1], math.nan]):
                with pytest.raises(KnowledgeSearchError):
                    await repository.search_ready_chunks(
                        query_vector=invalid_vector,
                        embedding_model=MODEL,
                        limit=2,
                    )
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            async with session_factory() as session:
                async with session.begin():
                    await cleanup(session)
                    await assert_database_restored(session)
        except BaseException:
            cleanup_failed = True
            if not original_exception_pending:
                raise
        finally:
            try:
                await engine.dispose()
            except BaseException:
                if not original_exception_pending and not cleanup_failed:
                    raise


def test_exact_vector_search_filters_in_sql_and_orders_stably() -> None:
    """只返回当前模型的公开就绪分块，并由数据库完成过滤、排序和 limit。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_vector_search(database_url))
