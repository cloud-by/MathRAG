"""知识管理读取仓储的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.management_repository import KnowledgeManagementRepository
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from tests.integration.database_safety import require_test_database_url


def _principal(role: str) -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=UUID(int=100 if role == "admin" else 101),
        session_id=UUID(int=200 if role == "admin" else 201),
        username=role,
        role=role,  # type: ignore[arg-type]
        session_token_hash=b"integration-session-token",
    )


def _item(
    item_id: int,
    *,
    title: str,
    updated_at: datetime,
    category: str = "management-algebra",
    visibility: str = "public",
    status: str = "ready",
) -> KnowledgeItem:
    return KnowledgeItem(
        id=UUID(int=item_id),
        legacy_id=None,
        owner_id=None,
        category=category,
        title=title,
        keywords=["management"],
        content=f"{title}内容",
        example="",
        steps=["步骤一"],
        difficulty="easy",
        visibility=visibility,
        status=status,
        revision=1,
        created_at=updated_at,
        updated_at=updated_at,
    )


async def _cleanup(session: AsyncSession) -> None:
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def _exercise_repository(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    first_time = datetime(2026, 7, 31, 8, 0, tzinfo=UTC)
    latest_time = datetime(2026, 7, 31, 9, 0, tzinfo=UTC)
    public_old = _item(1001, title="公开旧条目", updated_at=first_time)
    public_tie_low = _item(1002, title="公开同刻低 ID", updated_at=latest_time)
    public_tie_high = _item(
        1003,
        title="公开同刻高 ID",
        updated_at=latest_time,
        category="management-geometry",
    )
    private_ready = _item(
        1004,
        title="私有就绪条目",
        updated_at=latest_time,
        visibility="private",
    )
    public_failed = _item(
        1005,
        title="公开失败条目",
        updated_at=latest_time,
        status="failed",
    )
    public_archived = _item(
        1006,
        title="公开归档条目",
        updated_at=latest_time,
        status="archived",
    )
    seeded = [
        public_old,
        public_tie_low,
        public_tie_high,
        private_ready,
        public_failed,
        public_archived,
    ]
    try:
        async with session_factory() as session:
            async with session.begin():
                await _cleanup(session)
                session.add_all(seeded)

        async with session_factory() as session:
            repository = KnowledgeManagementRepository(session)
            user = _principal("user")
            admin = _principal("admin")

            assert await repository.get_visible(public_old.id, user) is not None
            for hidden in (private_ready, public_failed, public_archived):
                assert await repository.get_visible(hidden.id, user) is None
                assert await repository.get_visible(hidden.id, admin) is not None
            assert await repository.get_visible(UUID(int=9999), admin) is None

            user_items, user_total = await repository.list_visible(user)
            assert [item.id for item in user_items] == [
                public_tie_high.id,
                public_tie_low.id,
                public_old.id,
            ]
            assert user_total == 3

            page_items, page_total = await repository.list_visible(
                user,
                status=None,
                visibility=None,
                category=None,
                offset=1,
                limit=1,
            )
            assert [item.id for item in page_items] == [public_tie_low.id]
            assert page_total == 3

            category_items, category_total = await repository.list_visible(
                user,
                status="ready",
                visibility="public",
                category="management-algebra",
                offset=0,
                limit=100,
            )
            assert [item.id for item in category_items] == [
                public_tie_low.id,
                public_old.id,
            ]
            assert category_total == 2

            hidden_items, hidden_total = await repository.list_visible(
                user,
                status="failed",
                visibility="private",
                category=None,
                offset=0,
                limit=100,
            )
            assert hidden_items == []
            assert hidden_total == 0

            admin_items, admin_total = await repository.list_visible(
                admin,
                status=None,
                visibility=None,
                category=None,
                offset=0,
                limit=100,
            )
            assert {item.id for item in admin_items} == {item.id for item in seeded}
            assert admin_total == len(seeded)

        uncommitted = _item(1099, title="未提交条目", updated_at=latest_time)
        async with session_factory() as session:
            repository = KnowledgeManagementRepository(session)
            session.add(uncommitted)
            await session.flush()
            items, _ = await repository.list_visible(
                _principal("admin"),
                status=None,
                visibility=None,
                category=None,
                offset=0,
                limit=100,
            )
            assert uncommitted.id in {item.id for item in items}
            async with session_factory() as observer:
                assert await observer.scalar(
                    select(KnowledgeItem.id).where(KnowledgeItem.id == uncommitted.id)
                ) is None

        async with session_factory() as observer:
            assert await observer.scalar(
                select(KnowledgeItem.id).where(KnowledgeItem.id == uncommitted.id)
            ) is None
    finally:
        try:
            async with session_factory() as session:
                async with session.begin():
                    await _cleanup(session)
        finally:
            await engine.dispose()


def test_repository_enforces_visibility_sorting_pagination_and_transaction_boundary() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(_exercise_repository(safe_url))
