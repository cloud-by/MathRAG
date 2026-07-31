from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.errors import AppError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.conversations.models import Conversation, Message
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.search import KnowledgeSearchHit
from app.modules.rag.execution import RAGExecution
from app.modules.rag.models import RAGReference, RAGRun
from app.modules.rag.service import ChatPersistenceService
from app.modules.users.models import User
from tests.integration.database_safety import require_test_database_url


class FakeExecutor:
    def __init__(
        self,
        execution: RAGExecution,
        session_factory,
        conversation_id,
        client_request_id,
    ) -> None:
        self.execution = execution
        self.session_factory = session_factory
        self.conversation_id = conversation_id
        self.client_request_id = client_request_id
        self.calls = 0

    async def execute(self, *, question, history, top_k) -> RAGExecution:
        self.calls += 1
        assert question == "持久化问题"
        assert history == [
            {"role": "user", "content": "历史问题"},
            {"role": "assistant", "content": "历史回答"},
        ]
        assert top_k == 2
        async with self.session_factory() as session:
            run = await session.scalar(
                select(RAGRun).where(
                    RAGRun.conversation_id == self.conversation_id,
                    RAGRun.client_request_id == self.client_request_id,
                )
            )
            assert run is not None and run.status == "running"
            question_message = await session.get(Message, run.question_message_id)
            answer_message = await session.get(Message, run.answer_message_id)
            assert question_message is not None
            assert (question_message.role, question_message.status) == (
                "user",
                "completed",
            )
            assert answer_message is not None
            assert (answer_message.role, answer_message.status, answer_message.content) == (
                "assistant",
                "pending",
                "",
            )
        return self.execution


class BlockingExecutor:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def execute(self, *, question, history, top_k) -> RAGExecution:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        return RAGExecution(
            question=question,
            answer="并发回答",
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


async def exercise_persistence(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    user_id = uuid4()
    conversation_id = uuid4()
    item_id = uuid4()
    chunk_id = uuid4()
    client_request_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with session_factory() as session:
            async with session.begin():
                user = User(
                    id=user_id,
                    username=f"rag-{user_id.hex[:12]}",
                    password_hash="argon2-test-hash",
                )
                session.add(user)
                await session.flush()
                session.add(
                    Conversation(
                        id=conversation_id,
                        user_id=user_id,
                        title="新对话",
                        created_at=now,
                        updated_at=now,
                    )
                )
                await session.flush()
                session.add_all(
                    [
                        Message(
                            id=uuid4(),
                            conversation_id=conversation_id,
                            role="user",
                            content="历史问题",
                            status="completed",
                            created_at=now - timedelta(seconds=2),
                        ),
                        Message(
                            id=uuid4(),
                            conversation_id=conversation_id,
                            role="assistant",
                            content="历史回答",
                            status="completed",
                            created_at=now - timedelta(seconds=1),
                        ),
                    ]
                )
                item = KnowledgeItem(
                    id=item_id,
                    legacy_id=f"legacy-{item_id.hex[:12]}",
                    category="algebra",
                    title="快照标题",
                    keywords=["方程"],
                    content="不可变内容",
                    example="示例",
                    steps=["步骤"],
                    difficulty="easy",
                    visibility="public",
                    status="indexing",
                )
                session.add(item)
                await session.flush()
                session.add(
                    KnowledgeChunk(
                        id=chunk_id,
                        knowledge_item_id=item_id,
                        chunk_index=0,
                        retrieval_text="检索文本",
                        answer_context="回答上下文",
                        metadata_={},
                        status="pending",
                    )
                )

        hit = KnowledgeSearchHit(
            database_chunk_id=chunk_id,
            legacy_chunk_id="legacy-chunk-0",
            legacy_source_id="legacy-source",
            category="algebra",
            title="快照标题",
            keywords=("方程",),
            content="不可变内容",
            example="示例",
            steps=("步骤",),
            difficulty="easy",
            answer_context="回答上下文",
            retrieval_text="检索文本",
            source_line=7,
            metadata={"safe": True},
            distance=0.1,
        )
        execution = RAGExecution(
            question="持久化问题",
            answer="持久化回答",
            steps=("第一步",),
            used_knowledge=("快照标题",),
            related_questions=("继续？",),
            hits=(hit,),
            strategy="multi",
            retrieval_queries=("持久化问题", "扩展查询"),
            top_k=2,
            llm_model="llm-test",
            embedding_model="embedding-test",
            reasoning_content="可公开推理",
            model_metadata={"finish_reason": "stop", "total_tokens": 8},
            agentic_plan_queries=("扩展查询",),
        )
        executor = FakeExecutor(
            execution,
            session_factory,
            conversation_id,
            client_request_id,
        )
        principal = AuthenticatedPrincipal(
            user_id=user_id,
            session_id=uuid4(),
            username="rag-user",
            role="user",
            session_token_hash=b"hash",
        )
        service = ChatPersistenceService(session_factory, executor, lambda: datetime.now(UTC))

        first = await service.chat(
            principal=principal,
            conversation_id=conversation_id,
            client_request_id=client_request_id,
            question="持久化问题",
            top_k=2,
        )
        replay = await service.chat(
            principal=principal,
            conversation_id=conversation_id,
            client_request_id=client_request_id,
            question="持久化问题",
            top_k=2,
        )

        assert replay == first
        assert executor.calls == 1
        assert first.response == execution.to_public_response()
        async with session_factory() as session:
            messages = list(
                (
                    await session.scalars(
                        select(Message)
                        .where(Message.conversation_id == conversation_id)
                        .order_by(Message.created_at, Message.id)
                    )
                ).all()
            )
            run = await session.scalar(
                select(RAGRun).where(RAGRun.id == first.rag_run_id)
            )
            reference = await session.scalar(
                select(RAGReference).where(RAGReference.rag_run_id == first.rag_run_id)
            )
            assert [(item.role, item.status) for item in messages] == [
                ("user", "completed"),
                ("assistant", "completed"),
                ("user", "completed"),
                ("assistant", "completed"),
            ]
            assert [item.content for item in messages] == [
                "历史问题",
                "历史回答",
                "持久化问题",
                "持久化回答",
            ]
            conversation = await session.get(Conversation, conversation_id)
            assert conversation is not None and conversation.title == "持久化问题"
            assert run is not None and run.status == "completed"
            assert reference is not None
            assert reference.chunk_id == chunk_id
            assert reference.snapshot["content"] == "不可变内容"

        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id))
        after_delete = await service.chat(
            principal=principal,
            conversation_id=conversation_id,
            client_request_id=client_request_id,
            question="持久化问题",
            top_k=2,
        )
        assert after_delete == first
        async with session_factory() as session:
            reference = await session.scalar(
                select(RAGReference).where(RAGReference.rag_run_id == first.rag_run_id)
            )
            assert reference is not None
            assert reference.chunk_id is None
            assert reference.snapshot["content"] == "不可变内容"
            assert await session.scalar(
                select(func.count()).select_from(RAGRun).where(
                    RAGRun.conversation_id == conversation_id
                )
            ) == 1
    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(Conversation).where(Conversation.id == conversation_id))
                await session.execute(delete(KnowledgeItem).where(KnowledgeItem.id == item_id))
                await session.execute(delete(User).where(User.id == user_id))
        await engine.dispose()


def test_rag_persistence_replays_completed_result_and_keeps_snapshot() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_persistence(database_url))


async def exercise_concurrent_idempotency(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    user_id = uuid4()
    conversation_id = uuid4()
    client_request_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with session_factory() as session:
            async with session.begin():
                session.add(
                    User(
                        id=user_id,
                        username=f"concurrent-{user_id.hex[:10]}",
                        password_hash="hash",
                    )
                )
                await session.flush()
                session.add(
                    Conversation(
                        id=conversation_id,
                        user_id=user_id,
                        title="并发会话",
                        created_at=now,
                        updated_at=now,
                    )
                )

        principal = AuthenticatedPrincipal(
            user_id=user_id,
            session_id=uuid4(),
            username="concurrent-user",
            role="user",
            session_token_hash=b"hash",
        )
        executor = BlockingExecutor()
        service = ChatPersistenceService(
            session_factory,
            executor,
            lambda: datetime.now(UTC),
        )
        first_task = asyncio.create_task(
            service.chat(
                principal=principal,
                conversation_id=conversation_id,
                client_request_id=client_request_id,
                question="并发问题",
                top_k=2,
            )
        )
        await asyncio.wait_for(executor.entered.wait(), timeout=5)
        try:
            with pytest.raises(AppError) as captured:
                await service.chat(
                    principal=principal,
                    conversation_id=conversation_id,
                    client_request_id=client_request_id,
                    question="并发问题",
                    top_k=2,
                )
        finally:
            executor.release.set()
        assert captured.value.code == "RAG_REQUEST_IN_PROGRESS"
        first = await asyncio.wait_for(first_task, timeout=5)
        replay = await service.chat(
            principal=principal,
            conversation_id=conversation_id,
            client_request_id=client_request_id,
            question="并发问题",
            top_k=2,
        )
        assert replay == first
        assert executor.calls == 1

        async with session_factory() as session:
            assert await session.scalar(
                select(func.count()).select_from(RAGRun).where(
                    RAGRun.conversation_id == conversation_id
                )
            ) == 1
            assert await session.scalar(
                select(func.count()).select_from(Message).where(
                    Message.conversation_id == conversation_id
                )
            ) == 2
    finally:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(User).where(User.id == user_id))
        await engine.dispose()


def test_concurrent_same_client_request_executes_only_once() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_concurrent_idempotency(database_url))
