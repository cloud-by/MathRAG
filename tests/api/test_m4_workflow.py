"""M4 登录、会话、持久化问答与退出的完整工作流。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.modules.auth.dependencies import get_auth_service
from app.modules.auth.models import UserSession
from app.modules.auth.router import router as auth_router
from app.modules.auth.security import hash_password
from app.modules.auth.service import AuthService
from app.modules.conversations.models import Conversation, Message
from app.modules.conversations.router import (
    get_conversation_service,
    router as conversation_router,
)
from app.modules.conversations.service import ConversationService
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.search import KnowledgeSearchHit
from app.modules.rag.execution import RAGExecution
from app.modules.rag.models import RAGReference, RAGRun
from app.modules.rag.router import get_chat_persistence_service, router as rag_router
from app.modules.rag.service import ChatPersistenceService
from app.modules.users.models import User
from tests.integration.database_safety import require_test_database_url


class WorkflowExecutor:
    def __init__(self, hit: KnowledgeSearchHit) -> None:
        self.hit = hit
        self.calls = 0

    async def execute(self, *, question, history, top_k) -> RAGExecution:
        self.calls += 1
        assert history == []
        return RAGExecution(
            question=question,
            answer="工作流回答",
            steps=("第一步",),
            used_knowledge=(self.hit.title,),
            related_questions=("继续提问",),
            hits=(self.hit,),
            strategy="single",
            retrieval_queries=(question,),
            top_k=top_k,
            llm_model="llm-test",
            embedding_model="embedding-test",
            reasoning_content=None,
            model_metadata={"finish_reason": "stop"},
            agentic_plan_queries=(question,),
        )


def build_app(
    auth_service: AuthService,
    conversation_service: ConversationService,
    chat_service: ChatPersistenceService,
) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(auth_router)
    application.include_router(conversation_router)
    application.include_router(rag_router)
    application.dependency_overrides[get_auth_service] = lambda: auth_service
    application.dependency_overrides[get_conversation_service] = (
        lambda: conversation_service
    )
    application.dependency_overrides[get_chat_persistence_service] = lambda: chat_service
    return application


async def seed_workflow(session_factory, user_id, item_id, chunk_id) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add(
                User(
                    id=user_id,
                    username=f"workflow-{user_id.hex[:10]}",
                    email=f"workflow-{user_id.hex[:10]}@example.local",
                    password_hash=await hash_password("workflow-password"),
                )
            )
            session.add(
                KnowledgeItem(
                    id=item_id,
                    legacy_id=f"workflow-{item_id.hex[:12]}",
                    category="geometry",
                    title="工作流知识",
                    keywords=["几何"],
                    content="工作流知识内容",
                    example="示例",
                    steps=["步骤"],
                    difficulty="easy",
                    visibility="public",
                    status="indexing",
                )
            )
            await session.flush()
            session.add(
                KnowledgeChunk(
                    id=chunk_id,
                    knowledge_item_id=item_id,
                    chunk_index=0,
                    retrieval_text="工作流检索文本",
                    answer_context="工作流回答上下文",
                    metadata_={},
                    status="pending",
                )
            )


async def verify_workflow_rows(session_factory, conversation_id) -> None:
    async with session_factory() as session:
        assert await session.scalar(
            select(func.count()).select_from(Message).where(
                Message.conversation_id == conversation_id
            )
        ) == 2
        assert await session.scalar(
            select(func.count()).select_from(RAGRun).where(
                RAGRun.conversation_id == conversation_id,
                RAGRun.status == "completed",
            )
        ) == 1
        assert await session.scalar(
            select(func.count())
            .select_from(RAGReference)
            .join(RAGRun, RAGReference.rag_run_id == RAGRun.id)
            .where(RAGRun.conversation_id == conversation_id)
        ) == 1


def test_login_conversation_chat_messages_logout_workflow() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    engine = create_async_engine(database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    user_id = uuid4()
    username = f"workflow-{user_id.hex[:10]}"
    item_id = uuid4()
    chunk_id = uuid4()
    asyncio.run(seed_workflow(session_factory, user_id, item_id, chunk_id))
    hit = KnowledgeSearchHit(
        database_chunk_id=chunk_id,
        legacy_chunk_id="workflow-chunk-0",
        legacy_source_id="workflow-source",
        category="geometry",
        title="工作流知识",
        keywords=("几何",),
        content="工作流知识内容",
        example="示例",
        steps=("步骤",),
        difficulty="easy",
        answer_context="工作流回答上下文",
        retrieval_text="工作流检索文本",
        source_line=1,
        metadata={},
        distance=0.1,
    )
    executor = WorkflowExecutor(hit)
    auth_service = AuthService(
        session_factory,
        session_ttl_seconds=settings.SESSION_TTL_SECONDS,
        csrf_secret=settings.SESSION_SECRET,
    )
    conversation_service = ConversationService(session_factory)
    chat_service = ChatPersistenceService(
        session_factory,
        executor,
        lambda: datetime.now(UTC),
    )
    client = TestClient(build_app(auth_service, conversation_service, chat_service))
    conversation_id = None
    try:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": username, "password": "workflow-password"},
            headers={"Origin": "http://localhost:8000"},
        )
        assert login.status_code == 200
        csrf_token = client.cookies.get(settings.csrf_cookie_name)
        assert csrf_token
        unsafe_headers = {
            "Origin": "http://localhost:8000",
            "X-CSRF-Token": csrf_token,
        }

        created = client.post(
            "/api/v1/conversations",
            json={"title": "新对话"},
            headers=unsafe_headers,
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        client_request_id = str(uuid4())
        answered = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "client_request_id": client_request_id,
                "question": "工作流问题",
                "top_k": 1,
            },
            headers=unsafe_headers,
        )
        assert answered.status_code == 200
        assert answered.json()["client_request_id"] == client_request_id
        assert answered.json()["references"][0]["chunk_id"] == "workflow-chunk-0"

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages"
        )
        assert messages.status_code == 200
        assert [(item["role"], item["status"]) for item in messages.json()["items"]] == [
            ("user", "completed"),
            ("assistant", "completed"),
        ]

        logout = client.post("/api/v1/auth/logout", headers=unsafe_headers)
        assert logout.status_code == 204
        assert client.get("/api/v1/auth/me").status_code == 401
        assert executor.calls == 1
        asyncio.run(verify_workflow_rows(session_factory, conversation_id))
    finally:
        async def cleanup() -> None:
            async with session_factory() as session:
                async with session.begin():
                    await session.execute(delete(UserSession).where(UserSession.user_id == user_id))
                    await session.execute(delete(User).where(User.id == user_id))
                    await session.execute(delete(KnowledgeItem).where(KnowledgeItem.id == item_id))
            await engine.dispose()

        asyncio.run(cleanup())
