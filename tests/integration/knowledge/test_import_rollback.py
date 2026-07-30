"""旧知识批量导入真实事务回滚测试。"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable

import pytest
from sqlalchemy import delete, event, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.knowledge.errors import LegacyKnowledgeConflictError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.schemas import (
    LegacyKnowledgeBundle,
    LegacyKnowledgeChunkInput,
    LegacyKnowledgeItemInput,
)
from app.modules.knowledge.service import LegacyKnowledgeImportService
from tests.integration.database_safety import require_test_database_url


def make_bundle(legacy_id: str, retrieval_text: str | None = None) -> LegacyKnowledgeBundle:
    """构造用于回滚验证的合法旧知识。"""
    item = LegacyKnowledgeItemInput(
        id=legacy_id,
        category="rollback-test",
        title=f"回滚条目 {legacy_id}",
        keywords=["事务"],
        content=f"{legacy_id} 内容",
        example="示例",
        steps=["验证"],
        difficulty="easy",
    )
    return LegacyKnowledgeBundle(
        item=item,
        chunk=LegacyKnowledgeChunkInput(
            chunk_id=f"{legacy_id}-chunk-0",
            source_id=legacy_id,
            category=item.category,
            title=item.title,
            keywords=item.keywords,
            content=item.content,
            example=item.example,
            steps=item.steps,
            difficulty=item.difficulty,
            source_line=1,
            retrieval_text=retrieval_text or f"{legacy_id} 检索文本",
            answer_context=f"{legacy_id} 回答上下文",
            metadata={"origin": "rollback-test"},
        ),
    )


async def cleanup(session: AsyncSession) -> None:
    """按照外键顺序清空专用测试库。"""
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def assert_counts(session: AsyncSession, items: int, chunks: int) -> None:
    """断言知识表精确行数。"""
    assert await session.scalar(select(func.count()).select_from(KnowledgeItem)) == items
    assert await session.scalar(select(func.count()).select_from(KnowledgeChunk)) == chunks


async def exercise_rollback(database_url: str) -> None:
    """验证已 flush 的首条新增记录仍会在后续冲突时整体回滚。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as session:
            async with session.begin():
                await cleanup(session)
                await assert_counts(session, 0, 0)

        original = make_bundle("k0002")
        async with session_factory() as session:
            service = LegacyKnowledgeImportService(session, KnowledgeRepository(session))
            await service.import_bundles([original])

        async with session_factory() as session:
            stored = await KnowledgeRepository(session).get_by_legacy_id("k0002")
            assert stored is not None
            assert len(stored.chunks) == 1
            assert stored.chunks[0].retrieval_text == "k0002 检索文本"

        flushed_new_ids: list[str] = []
        async with session_factory() as session:
            def record_flush(sync_session: object, context: object) -> None:
                del context
                flushed_new_ids.extend(
                    item.legacy_id
                    for item in sync_session.new  # type: ignore[attr-defined]
                    if isinstance(item, KnowledgeItem) and item.legacy_id == "k0001"
                )

            event.listen(session.sync_session, "after_flush", record_flush)
            try:
                service = LegacyKnowledgeImportService(session, KnowledgeRepository(session))
                with pytest.raises(LegacyKnowledgeConflictError, match="k0002"):
                    await service.import_bundles(
                        [make_bundle("k0001"), make_bundle("k0002", "冲突检索文本")]
                    )
            finally:
                event.remove(session.sync_session, "after_flush", record_flush)

        assert flushed_new_ids == ["k0001"]
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            assert await repository.get_by_legacy_id("k0001") is None
            stored = await repository.get_by_legacy_id("k0002")
            assert stored is not None
            assert stored.chunks[0].retrieval_text == "k0002 检索文本"
            await assert_counts(session, 1, 1)
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            async with session_factory() as session:
                async with session.begin():
                    await cleanup(session)
                    await assert_counts(session, 0, 0)
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


def test_import_conflict_rolls_back_previously_flushed_insert() -> None:
    """真实 PostgreSQL 中，冲突批次不能遗留先前已 flush 的首条记录。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_rollback(database_url))
