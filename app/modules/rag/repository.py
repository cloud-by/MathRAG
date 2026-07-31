"""RAG 运行、消息与引用快照的事务内持久化操作。"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.modules.conversations.models import Conversation, Message
from app.modules.rag.errors import RAGStateConflictError
from app.modules.rag.execution import RAGExecution
from app.modules.rag.models import RAGReference, RAGRun


@dataclass(frozen=True)
class PendingRun:
    conversation_id: UUID
    client_request_id: UUID
    question_message_id: UUID
    answer_message_id: UUID
    rag_run_id: UUID
    question: str
    history: tuple[dict[str, str], ...]
    top_k: int
    started_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "history", tuple(deepcopy(list(self.history))))


@dataclass(frozen=True)
class PersistedReference:
    rank: int
    chunk_id: UUID | None
    score: float
    snapshot: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot", deepcopy(self.snapshot))


@dataclass(frozen=True)
class PersistedChatResult:
    conversation_id: UUID
    question_message_id: UUID
    answer_message_id: UUID
    rag_run_id: UUID
    client_request_id: UUID
    response: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", deepcopy(self.response))

    def to_public_response(self) -> dict[str, object]:
        return {
            **deepcopy(self.response),
            "conversation_id": self.conversation_id,
            "question_message_id": self.question_message_id,
            "answer_message_id": self.answer_message_id,
            "rag_run_id": self.rag_run_id,
            "client_request_id": self.client_request_id,
        }


@dataclass(frozen=True)
class PersistedRun:
    conversation_id: UUID
    client_request_id: UUID
    question_message_id: UUID
    answer_message_id: UUID | None
    rag_run_id: UUID
    status: str
    question: str
    answer: str
    answer_metadata: dict[str, object]
    strategy: str | None
    retrieval_queries: tuple[str, ...]
    top_k: int
    error_code: str | None
    references: tuple[PersistedReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "answer_metadata", deepcopy(self.answer_metadata))

    def to_chat_result(self) -> PersistedChatResult:
        if self.status != "completed" or self.answer_message_id is None:
            raise RAGStateConflictError("只有 completed 运行可以重建回答")
        stored_response = self.answer_metadata.get("response")
        if not isinstance(stored_response, dict):
            raise RAGStateConflictError("completed 运行缺少回答元数据")
        response = {
            "question": self.question,
            "answer": self.answer,
            **deepcopy(stored_response),
            "references": [_public_reference(item) for item in self.references],
        }
        return PersistedChatResult(
            conversation_id=self.conversation_id,
            question_message_id=self.question_message_id,
            answer_message_id=self.answer_message_id,
            rag_run_id=self.rag_run_id,
            client_request_id=self.client_request_id,
            response=response,
        )


class RAGRepository:
    """只执行当前事务内 SQL，不管理 commit、rollback 或 Session 生命周期。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_owned_conversation(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> Conversation | None:
        return await self._session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )

    async def get_by_client_request(
        self,
        conversation_id: UUID,
        user_id: UUID,
        client_request_id: UUID,
    ) -> PersistedRun | None:
        question_message = aliased(Message)
        answer_message = aliased(Message)
        statement = (
            select(RAGRun, question_message, answer_message)
            .join(Conversation, RAGRun.conversation_id == Conversation.id)
            .join(
                question_message,
                question_message.id == RAGRun.question_message_id,
            )
            .outerjoin(
                answer_message,
                answer_message.id == RAGRun.answer_message_id,
            )
            .where(
                RAGRun.conversation_id == conversation_id,
                Conversation.user_id == user_id,
                RAGRun.client_request_id == client_request_id,
            )
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        run, question, answer = row
        references = tuple(
            PersistedReference(
                rank=reference.rank,
                chunk_id=reference.chunk_id,
                score=reference.score,
                snapshot=reference.snapshot,
            )
            for reference in (
                await self._session.scalars(
                    select(RAGReference)
                    .where(RAGReference.rag_run_id == run.id)
                    .order_by(RAGReference.rank.asc())
                )
            ).all()
        )
        return PersistedRun(
            conversation_id=run.conversation_id,
            client_request_id=run.client_request_id,
            question_message_id=run.question_message_id,
            answer_message_id=run.answer_message_id,
            rag_run_id=run.id,
            status=run.status,
            question=question.content,
            answer=answer.content if answer is not None else "",
            answer_metadata=(
                answer.model_metadata if answer is not None else {}
            ),
            strategy=run.strategy,
            retrieval_queries=tuple(run.retrieval_queries),
            top_k=run.top_k,
            error_code=run.error_code,
            references=references,
        )

    async def create_running(
        self,
        *,
        conversation: Conversation,
        client_request_id: UUID,
        question: str,
        top_k: int,
        history_limit: int,
        now: datetime | None = None,
    ) -> PendingRun:
        created_at = now or datetime.now(UTC)
        rows = (
            await self._session.execute(
                select(Message.role, Message.content)
                .where(
                    Message.conversation_id == conversation.id,
                    Message.status == "completed",
                    Message.role.in_(("user", "assistant")),
                )
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(history_limit)
            )
        ).all()
        history = tuple(
            {"role": role, "content": content}
            for role, content in reversed(rows)
        )

        question_message_id = uuid4()
        answer_message_id = uuid4()
        rag_run_id = uuid4()
        question_message = Message(
            id=question_message_id,
            conversation_id=conversation.id,
            role="user",
            content=question,
            status="completed",
            model_metadata={},
            created_at=created_at,
        )
        answer_message = Message(
            id=answer_message_id,
            conversation_id=conversation.id,
            role="assistant",
            content="",
            status="pending",
            model_metadata={},
            created_at=created_at + timedelta(microseconds=1),
        )
        run = RAGRun(
            id=rag_run_id,
            conversation_id=conversation.id,
            question_message_id=question_message_id,
            answer_message_id=answer_message_id,
            client_request_id=client_request_id,
            retrieval_queries=[],
            top_k=top_k,
            status="running",
            created_at=created_at,
        )
        if conversation.title == "新对话":
            conversation.title = question[:40]
        conversation.updated_at = created_at
        self._session.add_all([question_message, answer_message, run])
        return PendingRun(
            conversation_id=conversation.id,
            client_request_id=client_request_id,
            question_message_id=question_message_id,
            answer_message_id=answer_message_id,
            rag_run_id=rag_run_id,
            question=question,
            history=history,
            top_k=top_k,
            started_at=created_at,
        )

    async def complete(
        self,
        pending: PendingRun,
        execution: RAGExecution,
        latency_ms: int,
    ) -> PersistedChatResult:
        if execution.question != pending.question or execution.top_k != pending.top_k:
            raise RAGStateConflictError("RAG 执行结果与 pending 运行不一致")
        if type(latency_ms) is not int or latency_ms < 0:
            raise RAGStateConflictError("latency_ms 必须是非负整数")

        public_response = execution.to_public_response()
        stored_response = {
            key: deepcopy(value)
            for key, value in public_response.items()
            if key not in {"question", "answer", "references"}
        }
        assistant_metadata = {
            "response": stored_response,
            "model": {
                "llm_model": execution.llm_model,
                "embedding_model": execution.embedding_model,
                **deepcopy(execution.model_metadata),
            },
        }
        message_result = await self._session.execute(
            update(Message)
            .where(
                Message.id == pending.answer_message_id,
                Message.conversation_id == pending.conversation_id,
                Message.status == "pending",
            )
            .values(
                content=execution.answer,
                status="completed",
                model_metadata=assistant_metadata,
            )
        )
        _require_single_row(message_result.rowcount, "assistant pending -> completed")

        run_result = await self._session.execute(
            update(RAGRun)
            .where(
                RAGRun.id == pending.rag_run_id,
                RAGRun.conversation_id == pending.conversation_id,
                RAGRun.status == "running",
            )
            .values(
                strategy=execution.strategy,
                retrieval_queries=list(execution.retrieval_queries),
                llm_model=execution.llm_model,
                embedding_model=execution.embedding_model,
                status="completed",
                latency_ms=latency_ms,
                error_code=None,
            )
        )
        _require_single_row(run_result.rowcount, "rag_run running -> completed")

        references: list[RAGReference] = []
        for snapshot in execution.to_reference_snapshots():
            if not math.isfinite(snapshot.score):
                raise RAGStateConflictError("引用 score 必须是有限数值")
            references.append(
                RAGReference(
                    rag_run_id=pending.rag_run_id,
                    rank=snapshot.rank,
                    chunk_id=snapshot.chunk_id,
                    score=snapshot.score,
                    snapshot=snapshot.snapshot,
                )
            )
        self._session.add_all(references)
        return PersistedChatResult(
            conversation_id=pending.conversation_id,
            question_message_id=pending.question_message_id,
            answer_message_id=pending.answer_message_id,
            rag_run_id=pending.rag_run_id,
            client_request_id=pending.client_request_id,
            response=public_response,
        )

    async def fail(
        self,
        pending: PendingRun,
        *,
        run_status: str,
        error_code: str,
        public_message: str,
        latency_ms: int,
    ) -> None:
        if run_status not in {"failed", "cancelled"}:
            raise RAGStateConflictError("RAG 失败终态无效")
        if type(latency_ms) is not int or latency_ms < 0:
            raise RAGStateConflictError("latency_ms 必须是非负整数")
        message_result = await self._session.execute(
            update(Message)
            .where(
                Message.id == pending.answer_message_id,
                Message.conversation_id == pending.conversation_id,
                Message.status == "pending",
            )
            .values(
                content=public_message,
                status="failed",
                model_metadata={"error_code": error_code},
            )
        )
        _require_single_row(message_result.rowcount, "assistant pending -> failed")
        run_result = await self._session.execute(
            update(RAGRun)
            .where(
                RAGRun.id == pending.rag_run_id,
                RAGRun.conversation_id == pending.conversation_id,
                RAGRun.status == "running",
            )
            .values(
                status=run_status,
                latency_ms=latency_ms,
                error_code=error_code,
            )
        )
        _require_single_row(run_result.rowcount, f"rag_run running -> {run_status}")


def _public_reference(reference: PersistedReference) -> dict[str, object]:
    snapshot = deepcopy(reference.snapshot)
    return {
        "rank": reference.rank,
        "score": reference.score,
        "index": None,
        "chunk_id": snapshot.pop("chunk_id", ""),
        "source_id": snapshot.pop("source_id", ""),
        "category": snapshot.pop("category", ""),
        "title": snapshot.pop("title", ""),
        "keywords": snapshot.pop("keywords", []),
        "content": snapshot.pop("content", ""),
        "example": snapshot.pop("example", ""),
        "steps": snapshot.pop("steps", []),
        "difficulty": snapshot.pop("difficulty", ""),
        "answer_context": snapshot.pop("answer_context", ""),
        "retrieval_text": snapshot.pop("retrieval_text", ""),
        "source_line": snapshot.pop("source_line", None),
        "metadata": snapshot.pop("metadata", {}),
    }


def _require_single_row(actual: object, operation: str) -> None:
    if type(actual) is not int or actual != 1:
        raise RAGStateConflictError(f"{operation} CAS 冲突")
