"""会话 Repository 的双用户 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.auth.models import UserSession
from app.modules.conversations.models import Conversation, Message
from app.modules.conversations.repository import ConversationRepository
from app.modules.rag.models import RAGReference, RAGRun
from app.modules.users.models import User
from tests.integration.database_safety import require_test_database_url


async def exercise_owner_scope(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    try:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(RAGReference))
                await session.execute(delete(RAGRun))
                await session.execute(delete(Message))
                await session.execute(delete(Conversation))
                await session.execute(delete(UserSession))
                await session.execute(delete(User))
                owner = User(username="owner-a", password_hash="hash-a")
                other = User(username="owner-b", password_hash="hash-b")
                session.add_all([owner, other])
                await session.flush()
                conversation = Conversation(
                    id=uuid4(),
                    user_id=owner.id,
                    title="A 的标题",
                    created_at=now,
                    updated_at=now,
                )
                session.add(conversation)
                await session.flush()
                session.add_all(
                    [
                        Message(
                            id=UUID(int=1),
                            conversation_id=conversation.id,
                            role="user",
                            content="A 的问题",
                            status="completed",
                            created_at=now,
                        ),
                        Message(
                            id=UUID(int=2),
                            conversation_id=conversation.id,
                            role="assistant",
                            content="A 的回答",
                            status="completed",
                            created_at=now,
                        ),
                    ]
                )

        async with session_factory() as session:
            repository = ConversationRepository(session)
            assert await repository.get_owned(conversation.id, owner.id) is not None
            assert await repository.get_owned(conversation.id, other.id) is None
            owner_items, owner_total = await repository.list_owned(
                owner.id,
                status="active",
                offset=0,
                limit=20,
            )
            other_items, other_total = await repository.list_owned(
                other.id,
                status="active",
                offset=0,
                limit=20,
            )
            assert [item.id for item in owner_items] == [conversation.id]
            assert owner_total == 1
            assert other_items == []
            assert other_total == 0
            assert await repository.update_owned(
                conversation.id,
                other.id,
                values={"title": "越权标题", "updated_at": now},
            ) is None
            owner_messages = await repository.list_owned_messages(
                conversation.id,
                owner.id,
                offset=0,
                limit=50,
            )
            assert owner_messages is not None
            assert [message.content for message in owner_messages[0]] == ["A 的问题", "A 的回答"]
            assert await repository.list_owned_messages(
                conversation.id,
                other.id,
                offset=0,
                limit=50,
            ) is None
    finally:
        await engine.dispose()


def test_conversation_repository_enforces_owner_scope_in_sql() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_owner_scope(database_url))
