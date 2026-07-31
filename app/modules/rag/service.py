"""两段短事务的持久化 RAG 编排服务。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import UUID

from openai import APIError, APITimeoutError, RateLimitError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.errors import AppError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.conversations.errors import ConversationNotFoundError
from app.modules.knowledge.errors import KnowledgeSearchError
from app.modules.rag.errors import (
    ConversationArchivedError,
    RAGRequestInProgressError,
    RAGStateConflictError,
    persisted_rag_error,
)
from app.modules.rag.execution import RAGExecution, RAGExecutor
from app.modules.rag.repository import (
    PendingRun,
    PersistedChatResult,
    PersistedRun,
    RAGRepository,
)


HISTORY_LIMIT = 8


class RAGRepositoryProtocol(Protocol):
    async def get_by_client_request(
        self,
        conversation_id: UUID,
        user_id: UUID,
        client_request_id: UUID,
    ) -> PersistedRun | None: ...

    async def get_owned_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> object | None: ...

    async def create_running(self, **kwargs: object) -> PendingRun: ...

    async def complete(
        self,
        pending: PendingRun,
        execution: RAGExecution,
        latency_ms: int,
    ) -> PersistedChatResult: ...

    async def fail(self, pending: PendingRun, **kwargs: object) -> None: ...


class ChatPersistenceService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        executor: RAGExecutor,
        clock: Callable[[], datetime],
        *,
        repository_factory: Callable[[AsyncSession], RAGRepositoryProtocol] = RAGRepository,
    ) -> None:
        self._session_factory = session_factory
        self._executor = executor
        self._clock = clock
        self._repository_factory = repository_factory

    async def chat(
        self,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: UUID,
        client_request_id: UUID,
        question: str,
        top_k: int | None,
    ) -> PersistedChatResult:
        normalized_question = str(question or "").strip()
        if not normalized_question or len(normalized_question) > 8000:
            raise AppError(
                code="REQUEST_VALIDATION_FAILED",
                message="问题长度必须在 1 到 8000 个字符之间。",
                status_code=422,
            )
        resolved_top_k = settings.TOP_K if top_k is None else top_k
        if type(resolved_top_k) is not int or not 1 <= resolved_top_k <= 10:
            raise AppError(
                code="REQUEST_VALIDATION_FAILED",
                message="top_k 必须是 1 到 10 的整数。",
                status_code=422,
            )

        started_at = self._clock()
        try:
            prepared = await self._prepare(
                principal=principal,
                conversation_id=conversation_id,
                client_request_id=client_request_id,
                question=normalized_question,
                top_k=resolved_top_k,
                now=started_at,
            )
        except IntegrityError:
            return await self._replay_after_unique_conflict(
                principal=principal,
                conversation_id=conversation_id,
                client_request_id=client_request_id,
            )
        except SQLAlchemyError:
            raise _database_unavailable() from None
        if isinstance(prepared, PersistedChatResult):
            return prepared
        pending = prepared

        try:
            execution = await self._executor.execute(
                question=pending.question,
                history=list(pending.history),
                top_k=pending.top_k,
            )
        except asyncio.CancelledError:
            try:
                await self._finalize_failure(
                    pending,
                    run_status="cancelled",
                    error_code="RAG_CANCELLED",
                    public_message="请求已取消。",
                    latency_ms=_latency_ms(started_at, self._clock()),
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            public_error = _map_execution_error(exc)
            try:
                await self._finalize_failure(
                    pending,
                    run_status="failed",
                    error_code=public_error.code,
                    public_message=public_error.message,
                    latency_ms=_latency_ms(started_at, self._clock()),
                )
            except Exception:
                public_error.details["persistence_confirmed"] = False
            raise public_error from None

        latency_ms = _latency_ms(started_at, self._clock())
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    return await self._repository_factory(session).complete(
                        pending,
                        execution,
                        latency_ms,
                    )
        except RAGStateConflictError:
            return await self._replay_after_unique_conflict(
                principal=principal,
                conversation_id=conversation_id,
                client_request_id=client_request_id,
            )
        except SQLAlchemyError:
            database_error = _database_unavailable()
            try:
                await self._finalize_failure(
                    pending,
                    run_status="failed",
                    error_code=database_error.code,
                    public_message=database_error.message,
                    latency_ms=latency_ms,
                )
            except Exception:
                database_error.details["persistence_confirmed"] = False
            raise database_error from None

    async def _prepare(
        self,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: UUID,
        client_request_id: UUID,
        question: str,
        top_k: int,
        now: datetime,
    ) -> PendingRun | PersistedChatResult:
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                existing = await repository.get_by_client_request(
                    conversation_id,
                    principal.user_id,
                    client_request_id,
                )
                if existing is not None:
                    return _replay(existing)
                conversation = await repository.get_owned_conversation(
                    conversation_id,
                    principal.user_id,
                )
                if conversation is None:
                    raise ConversationNotFoundError()
                if getattr(conversation, "status", None) != "active":
                    raise ConversationArchivedError()
                return await repository.create_running(
                    conversation=conversation,
                    client_request_id=client_request_id,
                    question=question,
                    top_k=top_k,
                    history_limit=HISTORY_LIMIT,
                    now=now,
                )

    async def _replay_after_unique_conflict(
        self,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: UUID,
        client_request_id: UUID,
    ) -> PersistedChatResult:
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    existing = await self._repository_factory(
                        session
                    ).get_by_client_request(
                        conversation_id,
                        principal.user_id,
                        client_request_id,
                    )
        except SQLAlchemyError:
            raise _database_unavailable() from None
        if existing is None:
            raise AppError(
                code="RAG_REQUEST_CONFLICT",
                message="幂等请求发生并发冲突，请重试。",
                status_code=409,
            )
        return _replay(existing)

    async def _finalize_failure(
        self,
        pending: PendingRun,
        *,
        run_status: str,
        error_code: str,
        public_message: str,
        latency_ms: int,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                await self._repository_factory(session).fail(
                    pending,
                    run_status=run_status,
                    error_code=error_code,
                    public_message=public_message,
                    latency_ms=latency_ms,
                )


def _replay(run: PersistedRun) -> PersistedChatResult:
    if run.status == "completed":
        return run.to_chat_result()
    if run.status == "running":
        raise RAGRequestInProgressError()
    if run.status in {"failed", "cancelled"}:
        raise persisted_rag_error(
            run.error_code or "INTERNAL_ERROR",
            run.answer or "请求处理失败。",
        )
    raise AppError(
        code="INTERNAL_ERROR",
        message="请求状态无效。",
        status_code=500,
    )


def _map_execution_error(exc: Exception) -> AppError:
    if isinstance(exc, SQLAlchemyError):
        return _database_unavailable()
    if isinstance(exc, KnowledgeSearchError):
        return AppError(
            code="EMBEDDING_UNAVAILABLE",
            message="知识检索服务暂时不可用。",
            status_code=502,
        )
    if isinstance(exc, RateLimitError):
        return AppError(
            code="LLM_RATE_LIMITED",
            message="回答服务请求过于频繁，请稍后重试。",
            status_code=429,
        )
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return AppError(
            code="RAG_UPSTREAM_TIMEOUT",
            message="上游服务响应超时。",
            status_code=504,
        )
    if isinstance(exc, (APIError, ValueError, RuntimeError)):
        return AppError(
            code="LLM_UNAVAILABLE",
            message="回答服务暂时不可用。",
            status_code=502,
        )
    return AppError(
        code="INTERNAL_ERROR",
        message="请求处理失败。",
        status_code=500,
    )


def _database_unavailable() -> AppError:
    return AppError(
        code="DATABASE_UNAVAILABLE",
        message="数据库服务暂时不可用。",
        status_code=503,
    )


def _latency_ms(started_at: datetime, finished_at: datetime) -> int:
    return max(0, int((finished_at - started_at).total_seconds() * 1000))
