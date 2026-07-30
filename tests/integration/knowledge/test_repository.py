"""知识仓储的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import ast
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from tests.integration.database_safety import require_test_database_url

from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_PATH = PROJECT_ROOT / "app" / "modules" / "knowledge" / "repository.py"
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
            status="pending",
        )
    )
    return item


async def cleanup(session: AsyncSession) -> None:
    """按外键依赖顺序清空专用测试库的知识表。"""
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def assert_knowledge_tables_empty(session: AsyncSession) -> None:
    """确认专用测试库未遗留任何知识条目或分块。"""
    item_count = await session.scalar(select(func.count()).select_from(KnowledgeItem))
    chunk_count = await session.scalar(select(func.count()).select_from(KnowledgeChunk))
    assert item_count == 0
    assert chunk_count == 0


async def exercise_repository(database_url: str) -> None:
    """通过独立会话验证仓储查询、加载及事务边界。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as cleanup_session:
            async with cleanup_session.begin():
                await cleanup(cleanup_session)

        async with session_factory() as stale_session:
            async with stale_session.begin():
                stale_key = uuid4().hex
                stale_session.add(
                    make_item(
                        f"stale-{stale_key}",
                        f"Repository 陈旧遗留条目 {stale_key}",
                    )
                )

        async with session_factory() as cleanup_session:
            async with cleanup_session.begin():
                await cleanup(cleanup_session)
                await assert_knowledge_tables_empty(cleanup_session)

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
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            async with session_factory() as cleanup_session:
                async with cleanup_session.begin():
                    await cleanup(cleanup_session)
                    await assert_knowledge_tables_empty(cleanup_session)
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


def test_repository_reads_and_writes_legacy_items_without_owning_transactions() -> None:
    """仓储仅操作会话，遗留查询不会包含普通知识条目。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_repository(database_url))


def is_repository_session(node: ast.expr) -> bool:
    """判断 AST 节点是否精确指向 ``self._session``。"""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_session"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def session_lifecycle_violations(tree: ast.AST) -> tuple[set[str], bool]:
    """收集仅由仓储会话发起的禁用调用及直接上下文管理。"""
    invoked_methods: set[str] = set()
    session_managed_directly = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and is_repository_session(node.func.value)
            and node.func.attr in {"commit", "rollback", "close", "begin"}
        ):
            invoked_methods.add(node.func.attr)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            session_managed_directly |= any(
                is_repository_session(item.context_expr) for item in node.items
            )
    return invoked_methods, session_managed_directly


def test_session_ast_helper_ignores_unrelated_objects() -> None:
    """同名方法或其他成员不得触发仓储会话违规判断。"""
    tree = ast.parse(
        """
class Repository:
    async def query(self, other):
        other.begin()
        self._other.close()
        self._session.execute("SELECT 1")
        async with self._session:
            pass
"""
    )

    invoked_methods, session_managed_directly = session_lifecycle_violations(tree)

    assert invoked_methods == set()
    assert session_managed_directly is True


def test_repository_does_not_control_session_transactions() -> None:
    """仓储不得调用或直接管理它持有的会话。"""
    tree = ast.parse(REPOSITORY_PATH.read_text(encoding="utf-8"))
    invoked_methods, session_managed_directly = session_lifecycle_violations(tree)

    assert invoked_methods == set()
    assert session_managed_directly is False
