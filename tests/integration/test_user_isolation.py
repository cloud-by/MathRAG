"""真实 PostgreSQL 上的双用户资源隔离测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.conversations.errors import ConversationNotFoundError
from app.modules.conversations.models import Conversation, Message
from app.modules.conversations.service import ConversationService
from app.modules.rag.execution import RAGExecution
from app.modules.rag.errors import ConversationArchivedError
from app.modules.rag.models import RAGRun
from app.modules.rag.service import ChatPersistenceService
from app.modules.users.models import User
from tests.integration.database_safety import require_test_database_url


class EchoExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def execute(self, *, question, history, top_k) -> RAGExecution:
        self.calls.append(question)
        assert history == []
        return RAGExecution(
            question=question,
            answer=f"回答：{question}",
            steps=(),
            used_knowledge=(),
            related_questions=(),
            hits=(),
            strategy="single",
            retrieval_queries=(question,),
            top_k=top_k,
            llm_model="llm-test",
            embedding_model="embedding-test",
            reasoning_content=None,
            model_metadata={},
        )


def principal(user_id, username: str) -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=user_id,
        session_id=uuid4(),
        username=username,
        role="user",
        session_token_hash=b"x" * 32,
    )


async def exercise_user_isolation(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    user_a_id = uuid4()
    user_b_id = uuid4()
    conversation_a_id = uuid4()
    conversation_b_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with session_factory() as session:
            async with session.begin():
                session.add_all(
                    [
                        User(
                            id=user_a_id,
                            username=f"isolation-a-{user_a_id.hex[:8]}",
                            password_hash="hash-a",
                        ),
                        User(
                            id=user_b_id,
                            username=f"isolation-b-{user_b_id.hex[:8]}",
                            password_hash="hash-b",
                        ),
                    ]
                )
                await session.flush()
                session.add_all(
                    [
                        Conversation(
                            id=conversation_a_id,
                            user_id=user_a_id,
                            title="A 的私有会话",
                            created_at=now,
                            updated_at=now,
                        ),
                        Conversation(
                            id=conversation_b_id,
                            user_id=user_b_id,
                            title="B 的私有会话",
                            created_at=now,
                            updated_at=now,
                        ),
                    ]
                )

        conversation_service = ConversationService(session_factory)
        executor = EchoExecutor()
        chat_service = ChatPersistenceService(
            session_factory,
            executor,
            lambda: datetime.now(UTC),
        )
        principal_a = principal(user_a_id, "user-a")
        principal_b = principal(user_b_id, "user-b")

        page_a = await conversation_service.list(
            user_a_id,
            status="active",
            page=1,
            page_size=20,
        )
        assert [item.id for item in page_a.items] == [conversation_a_id]
        with pytest.raises(ConversationNotFoundError):
            await conversation_service.get(conversation_b_id, user_a_id)
        with pytest.raises(ConversationNotFoundError):
            await conversation_service.update(
                conversation_b_id,
                user_a_id,
                {"title": "越权修改"},
            )
        with pytest.raises(ConversationNotFoundError):
            await conversation_service.list_messages(
                conversation_b_id,
                user_a_id,
                page=1,
                page_size=50,
            )
        with pytest.raises(ConversationNotFoundError) as captured:
            await chat_service.chat(
                principal=principal_a,
                conversation_id=conversation_b_id,
                client_request_id=uuid4(),
                question="不能写入 B",
                top_k=1,
            )
        assert "B 的私有会话" not in captured.value.message
        assert executor.calls == []

        result_a, result_b = await asyncio.gather(
            chat_service.chat(
                principal=principal_a,
                conversation_id=conversation_a_id,
                client_request_id=uuid4(),
                question="A 的问题",
                top_k=1,
            ),
            chat_service.chat(
                principal=principal_b,
                conversation_id=conversation_b_id,
                client_request_id=uuid4(),
                question="B 的问题",
                top_k=1,
            ),
        )
        assert result_a.conversation_id == conversation_a_id
        assert result_b.conversation_id == conversation_b_id
        assert sorted(executor.calls) == ["A 的问题", "B 的问题"]

        await conversation_service.archive(conversation_a_id, user_a_id)
        with pytest.raises(ConversationArchivedError):
            await chat_service.chat(
                principal=principal_a,
                conversation_id=conversation_a_id,
                client_request_id=uuid4(),
                question="归档后问题",
                top_k=1,
            )
        assert sorted(executor.calls) == ["A 的问题", "B 的问题"]

        async with session_factory() as session:
            run_owners = set(
                (
                    await session.execute(
                        select(Conversation.user_id, RAGRun.conversation_id)
                        .join(RAGRun, RAGRun.conversation_id == Conversation.id)
                        .where(Conversation.id.in_((conversation_a_id, conversation_b_id)))
                    )
                ).all()
            )
            assert run_owners == {
                (user_a_id, conversation_a_id),
                (user_b_id, conversation_b_id),
            }
            for conversation_id in (conversation_a_id, conversation_b_id):
                assert await session.scalar(
                    select(func.count()).select_from(Message).where(
                        Message.conversation_id == conversation_id
                    )
                ) == 2
    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    delete(User).where(User.id.in_((user_a_id, user_b_id)))
                )
        await engine.dispose()


def test_two_users_cannot_read_or_write_each_others_resources() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_user_isolation(database_url))
