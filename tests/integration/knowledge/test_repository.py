"""知识仓储的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import ast
import os
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_PATH = PROJECT_ROOT / "app" / "modules" / "knowledge" / "repository.py"
LEGACY_IDS = ("k9001", "k9002")
TEST_TITLES = ("Repository 遗留条目 k9001", "Repository 遗留条目 k9002", "Repository 非遗留条目")


def make_item(legacy_id: str | None, title: str) -> KnowledgeItem:
    """构造带一个分块的可持久化知识条目。"""
    item = KnowledgeItem(
        legacy_id=legacy_id,
        category="repository-test",
        title=title,
        keywords=["repository"],
        content=f"{title} 内容",
        example="示例",
        steps=["步骤"],
        difficulty="easy",
        status="ready",
    )
    item.chunks.append(
        KnowledgeChunk(
            chunk_index=0,
            retrieval_text=f"{title} 检索文本",
            answer_context=f"{title} 回答上下文",
            status="ready",
        )
    )
    return item


async def cleanup(session: AsyncSession) -> None:
    """按外键依赖顺序清理本测试写入的数据。"""
    target_item_ids = select(KnowledgeItem.id).where(KnowledgeItem.title.in_(TEST_TITLES))
    await session.execute(
        delete(KnowledgeChunk).where(KnowledgeChunk.knowledge_item_id.in_(target_item_ids))
    )
    await session.execute(delete(KnowledgeItem).where(KnowledgeItem.title.in_(TEST_TITLES)))


async def exercise_repository(database_url: str) -> None:
    """通过独立会话验证仓储查询、加载及事务边界。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as cleanup_session:
            async with cleanup_session.begin():
                await cleanup(cleanup_session)

        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            async with session.begin():
                repository.add(make_item("k9002", "Repository 遗留条目 k9002"))
                repository.add(make_item("k9001", "Repository 遗留条目 k9001"))
                repository.add(make_item(None, "Repository 非遗留条目"))

        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            item = await repository.get_by_legacy_id("k9001")
            assert item is not None
            assert item.legacy_id == "k9001"
            assert item.title == "Repository 遗留条目 k9001"
            assert len(item.chunks) == 1
            assert item.chunks[0].retrieval_text == "Repository 遗留条目 k9001 检索文本"
            assert await repository.get_by_legacy_id("unknown") is None
            assert await repository.count_legacy_items() == 2
            assert await repository.count_legacy_chunks() == 2
            items = await repository.list_legacy_items_ordered()
            assert [listed.legacy_id for listed in items] == ["k9001", "k9002"]
            assert [len(listed.chunks) for listed in items] == [1, 1]

        assert [chunk.answer_context for item in items for chunk in item.chunks] == [
            "Repository 遗留条目 k9001 回答上下文",
            "Repository 遗留条目 k9002 回答上下文",
        ]
    finally:
        async with session_factory() as cleanup_session:
            async with cleanup_session.begin():
                await cleanup(cleanup_session)
        await engine.dispose()


def test_repository_reads_and_writes_legacy_items_without_owning_transactions() -> None:
    """仓储仅操作会话，遗留查询不会包含普通知识条目。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")

    asyncio.run(exercise_repository(database_url))


def test_repository_does_not_control_session_transactions() -> None:
    """仓储源代码不得调用会话事务或生命周期方法。"""
    tree = ast.parse(REPOSITORY_PATH.read_text(encoding="utf-8"))
    invoked_methods = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {"commit", "rollback", "close", "begin"}.isdisjoint(invoked_methods)
