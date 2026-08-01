from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID

import httpx
import pytest
from openai import RateLimitError
from sqlalchemy.exc import OperationalError

from app.core.errors import AppError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import EmbeddingUnavailableError
from app.modules.rag.execution import RAGExecution
from app.modules.rag.repository import PendingRun, PersistedChatResult, PersistedRun
from app.modules.rag.service import ChatPersistenceService


USER_ID = UUID("00000000-0000-0000-0000-000000000101")
CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000102")
CLIENT_REQUEST_ID = UUID("00000000-0000-0000-0000-000000000103")
QUESTION_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000104")
ANSWER_MESSAGE_ID = UUID("00000000-0000-0000-0000-000000000105")
RUN_ID = UUID("00000000-0000-0000-0000-000000000106")


def make_execution() -> RAGExecution:
    return RAGExecution(
        question="当前问题",
        answer="安全回答",
        steps=("第一步",),
        used_knowledge=("知识点",),
        related_questions=("下一问",),
        hits=(),
        strategy="single",
        retrieval_queries=("当前问题",),
        top_k=3,
        llm_model="llm-test",
        embedding_model="embedding-test",
        reasoning_content=None,
        model_metadata={"finish_reason": "stop", "total_tokens": 12},
        agentic_plan_queries=("当前问题",),
    )


def make_pending(now: datetime) -> PendingRun:
    return PendingRun(
        conversation_id=CONVERSATION_ID,
        client_request_id=CLIENT_REQUEST_ID,
        question_message_id=QUESTION_MESSAGE_ID,
        answer_message_id=ANSWER_MESSAGE_ID,
        rag_run_id=RUN_ID,
        question="当前问题",
        history=({"role": "user", "content": "历史问题"},),
        top_k=3,
        started_at=now,
    )


class FakeTransaction:
    def __init__(self, events: list[str], number: int) -> None:
        self.events = events
        self.number = number

    async def __aenter__(self) -> None:
        self.events.append(f"transaction-{self.number}-begin")

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        suffix = "commit" if exc_type is None else "rollback"
        self.events.append(f"transaction-{self.number}-{suffix}")


class FakeSession:
    def __init__(self, events: list[str], number: int) -> None:
        self.events = events
        self.number = number

    async def __aenter__(self) -> FakeSession:
        self.events.append(f"session-{self.number}-open")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.events.append(f"session-{self.number}-close")

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self.events, self.number)


class FakeSessionFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.count = 0

    def __call__(self) -> FakeSession:
        self.count += 1
        return FakeSession(self.events, self.count)


class FakeRepository:
    def __init__(
        self,
        events: list[str],
        now: datetime,
        existing: PersistedRun | None = None,
    ) -> None:
        self.events = events
        self.pending = make_pending(now)
        self.existing = existing
        self.failed: list[tuple[str, str, str]] = []

    async def get_by_client_request(self, *args, **kwargs):
        return self.existing

    async def get_owned_conversation(self, conversation_id, user_id):
        assert (conversation_id, user_id) == (CONVERSATION_ID, USER_ID)
        return SimpleNamespace(id=conversation_id, title="新对话", status="active")

    async def create_running(self, **kwargs) -> PendingRun:
        assert kwargs["history_limit"] == 8
        assert kwargs["question"] == "当前问题"
        self.events.append("owned-conversation-and-history-read")
        self.events.append("question-assistant-run-written")
        return self.pending

    async def complete(self, pending, execution, latency_ms) -> PersistedChatResult:
        assert pending == self.pending
        assert execution.answer == "安全回答"
        assert latency_ms == 250
        self.events.append("answer-references-run-finalized")
        return PersistedChatResult(
            conversation_id=pending.conversation_id,
            question_message_id=pending.question_message_id,
            answer_message_id=pending.answer_message_id,
            rag_run_id=pending.rag_run_id,
            client_request_id=pending.client_request_id,
            response=execution.to_public_response(),
        )

    async def fail(
        self,
        pending,
        *,
        run_status,
        error_code,
        public_message,
        latency_ms,
    ) -> None:
        self.failed.append((run_status, error_code, public_message))


class FakeExecutor:
    def __init__(self, events: list[str], outcome: object) -> None:
        self.events = events
        self.outcome = outcome
        self.calls = 0

    async def execute(self, *, question, history, top_k):
        self.calls += 1
        self.events.append("rag-execute")
        assert question == "当前问题"
        assert history == [{"role": "user", "content": "历史问题"}]
        assert top_k == 3
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def make_principal() -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=USER_ID,
        session_id=UUID(int=999),
        username="tester",
        role="student",
        must_change_password=False,
        session_token_hash=b"hash",
    )


def test_chat_uses_two_short_transactions_around_external_execution() -> None:
    events: list[str] = []
    started_at = datetime(2026, 7, 31, tzinfo=UTC)
    times = iter((started_at, started_at + timedelta(milliseconds=250)))
    repository = FakeRepository(events, started_at)
    executor = FakeExecutor(events, make_execution())
    service = ChatPersistenceService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        executor,
        lambda: next(times),
        repository_factory=lambda session: repository,
    )

    result = asyncio.run(
        service.chat(
            principal=make_principal(),
            conversation_id=CONVERSATION_ID,
            client_request_id=CLIENT_REQUEST_ID,
            question=" 当前问题 ",
            top_k=3,
        )
    )

    assert result.rag_run_id == RUN_ID
    assert events == [
        "session-1-open",
        "transaction-1-begin",
        "owned-conversation-and-history-read",
        "question-assistant-run-written",
        "transaction-1-commit",
        "session-1-close",
        "rag-execute",
        "session-2-open",
        "transaction-2-begin",
        "answer-references-run-finalized",
        "transaction-2-commit",
        "session-2-close",
    ]


def test_embedding_failure_is_persisted_without_sensitive_exception_text() -> None:
    events: list[str] = []
    now = datetime(2026, 7, 31, tzinfo=UTC)
    repository = FakeRepository(events, now)
    executor = FakeExecutor(
        events,
        EmbeddingUnavailableError("provider-secret-marker"),
    )
    service = ChatPersistenceService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        executor,
        lambda: now,
        repository_factory=lambda session: repository,
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            service.chat(
                principal=make_principal(),
                conversation_id=CONVERSATION_ID,
                client_request_id=CLIENT_REQUEST_ID,
                question="当前问题",
                top_k=3,
            )
        )

    assert captured.value.code == "EMBEDDING_UNAVAILABLE"
    assert "provider-secret-marker" not in captured.value.message
    assert repository.failed == [
        ("failed", "EMBEDDING_UNAVAILABLE", "知识检索服务暂时不可用。")
    ]


def test_cancelled_execution_is_persisted_then_re_raised() -> None:
    events: list[str] = []
    now = datetime(2026, 7, 31, tzinfo=UTC)
    repository = FakeRepository(events, now)
    executor = FakeExecutor(events, asyncio.CancelledError())
    service = ChatPersistenceService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        executor,
        lambda: now,
        repository_factory=lambda session: repository,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            service.chat(
                principal=make_principal(),
                conversation_id=CONVERSATION_ID,
                client_request_id=CLIENT_REQUEST_ID,
                question="当前问题",
                top_k=3,
            )
        )

    assert repository.failed == [("cancelled", "RAG_CANCELLED", "请求已取消。")]


@pytest.mark.parametrize(
    ("outcome", "expected_code", "expected_status"),
    [
        (TimeoutError("timeout-marker"), "RAG_UPSTREAM_TIMEOUT", 504),
        (RuntimeError("provider-marker"), "LLM_UNAVAILABLE", 502),
        (
            RateLimitError(
                "rate-limit-marker",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://provider.invalid"),
                ),
                body=None,
            ),
            "LLM_RATE_LIMITED",
            429,
        ),
        (
            OperationalError("select 1", {}, RuntimeError("database-marker")),
            "DATABASE_UNAVAILABLE",
            503,
        ),
    ],
)
def test_execution_failures_use_stable_public_mapping(
    outcome: Exception,
    expected_code: str,
    expected_status: int,
) -> None:
    events: list[str] = []
    now = datetime(2026, 7, 31, tzinfo=UTC)
    repository = FakeRepository(events, now)
    service = ChatPersistenceService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        FakeExecutor(events, outcome),
        lambda: now,
        repository_factory=lambda session: repository,
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            service.chat(
                principal=make_principal(),
                conversation_id=CONVERSATION_ID,
                client_request_id=CLIENT_REQUEST_ID,
                question="当前问题",
                top_k=3,
            )
        )

    assert captured.value.code == expected_code
    assert captured.value.status_code == expected_status
    assert "marker" not in captured.value.message
    assert repository.failed[0][1] == expected_code


@pytest.mark.parametrize(
    ("status", "error_code", "answer", "expected_code"),
    [
        ("running", None, "", "RAG_REQUEST_IN_PROGRESS"),
        (
            "failed",
            "LLM_UNAVAILABLE",
            "回答服务暂时不可用。",
            "LLM_UNAVAILABLE",
        ),
    ],
)
def test_non_completed_replay_never_calls_executor(
    status: str,
    error_code: str | None,
    answer: str,
    expected_code: str,
) -> None:
    events: list[str] = []
    now = datetime(2026, 7, 31, tzinfo=UTC)
    existing = PersistedRun(
        conversation_id=CONVERSATION_ID,
        client_request_id=CLIENT_REQUEST_ID,
        question_message_id=QUESTION_MESSAGE_ID,
        answer_message_id=ANSWER_MESSAGE_ID,
        rag_run_id=RUN_ID,
        status=status,
        question="当前问题",
        answer=answer,
        answer_metadata={},
        strategy=None,
        retrieval_queries=(),
        top_k=3,
        error_code=error_code,
        references=(),
    )
    repository = FakeRepository(events, now, existing)
    executor = FakeExecutor(events, make_execution())
    service = ChatPersistenceService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        executor,
        lambda: now,
        repository_factory=lambda session: repository,
    )

    with pytest.raises(AppError) as captured:
        asyncio.run(
            service.chat(
                principal=make_principal(),
                conversation_id=CONVERSATION_ID,
                client_request_id=CLIENT_REQUEST_ID,
                question="当前问题",
                top_k=3,
            )
        )

    assert captured.value.code == expected_code
    assert executor.calls == 0
