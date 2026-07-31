from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest

from app.modules.rag.execution import RAGExecution
from app.modules.rag.repository import PendingRun, RAGRepository, RAGStateConflictError


class Result:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class FakeSession:
    def __init__(self, rowcounts: list[int]) -> None:
        self.rowcounts = iter(rowcounts)
        self.statements: list[object] = []
        self.added: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return Result(next(self.rowcounts))

    def add_all(self, values) -> None:
        self.added.extend(values)


def make_pending() -> PendingRun:
    return PendingRun(
        conversation_id=UUID(int=1),
        client_request_id=UUID(int=2),
        question_message_id=UUID(int=3),
        answer_message_id=UUID(int=4),
        rag_run_id=UUID(int=5),
        question="问题",
        history=(),
        top_k=3,
        started_at=datetime(2026, 7, 31, tzinfo=UTC),
    )


def make_execution() -> RAGExecution:
    return RAGExecution(
        question="问题",
        answer="回答",
        steps=(),
        used_knowledge=(),
        related_questions=(),
        hits=(),
        strategy="single",
        retrieval_queries=("问题",),
        top_k=3,
        llm_model="llm",
        embedding_model="embedding",
        reasoning_content=None,
        model_metadata={},
    )


def test_complete_uses_pending_and_running_compare_and_swap() -> None:
    session = FakeSession([1, 1])
    repository = RAGRepository(session)  # type: ignore[arg-type]

    asyncio.run(repository.complete(make_pending(), make_execution(), latency_ms=20))

    message_where = str(session.statements[0].whereclause)
    run_where = str(session.statements[1].whereclause)
    assert "messages.status" in message_where
    assert "messages.conversation_id" in message_where
    assert "rag_runs.status" in run_where
    assert "rag_runs.conversation_id" in run_where


def test_complete_rejects_lost_compare_and_swap() -> None:
    session = FakeSession([0])
    repository = RAGRepository(session)  # type: ignore[arg-type]

    with pytest.raises(RAGStateConflictError):
        asyncio.run(repository.complete(make_pending(), make_execution(), latency_ms=20))

