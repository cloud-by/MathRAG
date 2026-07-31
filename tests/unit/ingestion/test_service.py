"""导入服务的事务边界、错误映射和安全 DTO 测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from functools import wraps
from typing import Callable
from uuid import UUID

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.modules.ingestion.errors import DocumentStorageError
from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.repository import JobSnapshot
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import StoredUpload


NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
OWNER_ID = UUID(int=501)
DOCUMENT_ID = UUID(int=601)
JOB_ID = UUID(int=701)


def run_async(test: Callable) -> Callable:
    """沿用项目不依赖 pytest-asyncio 的异步单测执行方式。"""

    @wraps(test)
    def wrapper(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return wrapper


class ConstraintViolation(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__(constraint_name)
        self.constraint_name = constraint_name


class FakeTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    async def __aenter__(self) -> None:
        self._session.events.append("tx.begin")

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            self._session.events.append("tx.rollback")
            return False
        self._session.events.append("tx.commit")
        if self._session.commit_error is not None:
            self._session.events.append("tx.rollback")
            raise self._session.commit_error
        return False


class FakeSession:
    def __init__(
        self,
        events: list[str],
        commit_error: Exception | None,
    ) -> None:
        self.events = events
        self.commit_error = commit_error

    async def __aenter__(self) -> "FakeSession":
        self.events.append("session.open")
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.events.append("session.close")
        return False

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def flush(self) -> None:
        self.events.append("session.flush")


class FakeSessionFactory:
    def __init__(
        self,
        events: list[str],
        *,
        commit_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.commit_error = commit_error
        self.calls = 0

    def __call__(self) -> FakeSession:
        self.calls += 1
        return FakeSession(self.events, self.commit_error)


class FakeStorage:
    def __init__(
        self,
        events: list[str],
        *,
        save_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.save_error = save_error
        self.delete_error = delete_error
        self.deleted: list[str] = []
        self.released: list[str] = []

    async def save_upload(self, upload: object) -> StoredUpload:
        self.events.append("storage.save")
        if self.save_error is not None:
            raise self.save_error
        return StoredUpload(
            relative_path="2026/07/new-upload.pdf",
            size_bytes=2048,
            sha256="a" * 64,
            original_name="lesson.pdf",
            mime_type="application/pdf",
        )

    async def delete_upload(self, relative_path: str) -> None:
        self.events.append("storage.delete")
        self.deleted.append(relative_path)
        if self.delete_error is not None:
            raise self.delete_error

    def release_upload(self, relative_path: str) -> None:
        self.events.append("storage.release")
        self.released.append(relative_path)


class FakeRepository:
    def __init__(
        self,
        session: FakeSession,
        *,
        existing_job: IngestionJob | None = None,
        cancelled_job: IngestionJob | None = None,
        retry_snapshot: JobSnapshot | None = None,
        pending_snapshot: JobSnapshot | None = None,
        documents: list[Document] | None = None,
    ) -> None:
        self.session = session
        self.existing_job = existing_job
        self.cancelled_job = cancelled_job
        self.retry_snapshot = retry_snapshot
        self.pending_snapshot = pending_snapshot
        self.documents = documents or []
        self.added_document: Document | None = None
        self.added_job: IngestionJob | None = None

    def add_document(self, document: Document) -> None:
        self.session.events.append("document.add")
        self.added_document = document

    def add_job(self, job: IngestionJob) -> None:
        self.session.events.append("job.add")
        self.added_job = job

    async def list_documents(self, **kwargs):
        return self.documents, len(self.documents)

    async def get_job(self, job_id: UUID) -> IngestionJob | None:
        return self.existing_job

    async def cancel_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> IngestionJob | None:
        return self.cancelled_job

    async def claim_retry(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None:
        return self.retry_snapshot

    async def claim_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None:
        return self.pending_snapshot


class RepositoryFactory:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.instances: list[FakeRepository] = []

    def __call__(self, session: FakeSession) -> FakeRepository:
        repository = FakeRepository(session, **self.options)
        self.instances.append(repository)
        return repository


def _document(*, status: str = "pending") -> Document:
    return Document(
        id=DOCUMENT_ID,
        owner_id=OWNER_ID,
        original_name="lesson.pdf",
        storage_path="2026/07/private.pdf",
        mime_type="application/pdf",
        size_bytes=2048,
        sha256="b" * 64,
        status=status,
        created_at=NOW,
        updated_at=NOW,
    )


def _job(*, status: str, attempt_count: int = 0) -> IngestionJob:
    return IngestionJob(
        id=JOB_ID,
        requested_by=OWNER_ID,
        document_id=DOCUMENT_ID,
        job_type="pdf",
        status=status,
        progress=0,
        request_payload={"category": "代数", "storage_path": "绝不能公开"},
        attempt_count=attempt_count,
        error_code=None,
        error_message=None,
        started_at=None,
        finished_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


@run_async
async def test_accept_pdf_saves_before_session_and_returns_detached_safe_dto() -> None:
    events: list[str] = []
    sessions = FakeSessionFactory(events)
    storage = FakeStorage(events)
    repositories = RepositoryFactory()
    service = IngestionService(
        sessions,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        repository_factory=repositories,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    accepted = await service.accept_pdf(object(), owner_id=OWNER_ID, category=" 代数 ")

    assert events == [
        "storage.save",
        "session.open",
        "tx.begin",
        "document.add",
        "job.add",
        "session.flush",
        "tx.commit",
        "session.close",
        "storage.release",
    ]
    repository = repositories.instances[0]
    assert repository.added_document is not None
    assert repository.added_document.owner_id == OWNER_ID
    assert repository.added_document.storage_path == "2026/07/new-upload.pdf"
    assert repository.added_job is not None
    assert repository.added_job.requested_by == OWNER_ID
    assert repository.added_job.request_payload == {"category": "代数"}
    assert (accepted.document.status, accepted.job.status) == ("pending", "pending")
    serialized = accepted.model_dump_json()
    assert "storage_path" not in serialized
    assert "request_payload" not in serialized
    assert "new-upload.pdf" not in serialized
    # DTO 已脱离 Session，序列化不触发额外数据库行为。
    json.loads(serialized)
    assert sessions.calls == 1
    assert storage.released == ["2026/07/new-upload.pdf"]


@run_async
@pytest.mark.parametrize(
    "constraint_name",
    [
        "uq_documents_owner_id_sha256",
        "uq_documents_storage_path",
        "uq_ingestion_jobs_document_id_job_type",
    ],
)
async def test_accept_pdf_maps_only_related_unique_conflicts_and_cleans_file(
    constraint_name: str,
) -> None:
    events: list[str] = []
    error = IntegrityError("INSERT", {}, ConstraintViolation(constraint_name))
    sessions = FakeSessionFactory(events, commit_error=error)
    storage = FakeStorage(events)
    service = IngestionService(
        sessions,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(Exception) as captured:
        await service.accept_pdf(object(), owner_id=OWNER_ID, category=None)

    assert getattr(captured.value, "code", None) == "DOCUMENT_DUPLICATE"
    assert getattr(captured.value, "status_code", None) == 409
    assert storage.deleted == ["2026/07/new-upload.pdf"]
    assert events[-2:] == ["session.close", "storage.delete"]


@run_async
@pytest.mark.parametrize(
    "database_error",
    [
        IntegrityError("INSERT", {}, ConstraintViolation("uq_unrelated_table")),
        SQLAlchemyError("database unavailable"),
    ],
)
async def test_accept_pdf_preserves_non_duplicate_meaning_and_cleans_file(
    database_error: Exception,
) -> None:
    events: list[str] = []
    sessions = FakeSessionFactory(events, commit_error=database_error)
    storage = FakeStorage(events)
    service = IngestionService(
        sessions,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(Exception) as captured:
        await service.accept_pdf(object(), owner_id=OWNER_ID, category="代数")

    assert getattr(captured.value, "code", None) == "INGESTION_PERSISTENCE_FAILED"
    assert getattr(captured.value, "status_code", None) == 503
    assert storage.deleted == ["2026/07/new-upload.pdf"]


@run_async
async def test_cleanup_failure_does_not_mask_duplicate_mapping() -> None:
    events: list[str] = []
    duplicate = IntegrityError(
        "INSERT",
        {},
        ConstraintViolation("uq_documents_owner_id_sha256"),
    )
    storage = FakeStorage(events, delete_error=DocumentStorageError())
    service = IngestionService(
        FakeSessionFactory(events, commit_error=duplicate),  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(Exception) as captured:
        await service.accept_pdf(object(), owner_id=OWNER_ID)

    assert getattr(captured.value, "code", None) == "DOCUMENT_DUPLICATE"
    assert storage.deleted == ["2026/07/new-upload.pdf"]


@run_async
async def test_cancellation_waits_for_cleanup_and_remains_cancellation() -> None:
    events: list[str] = []
    storage = FakeStorage(events, delete_error=DocumentStorageError())
    service = IngestionService(
        FakeSessionFactory(  # type: ignore[arg-type]
            events,
            commit_error=asyncio.CancelledError(),
        ),
        storage,  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.accept_pdf(object(), owner_id=OWNER_ID)

    assert events[-2:] == ["session.close", "storage.delete"]


@run_async
async def test_storage_failure_never_opens_session() -> None:
    events: list[str] = []
    sessions = FakeSessionFactory(events)
    storage = FakeStorage(events, save_error=DocumentStorageError())
    service = IngestionService(
        sessions,  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(DocumentStorageError):
        await service.accept_pdf(object(), owner_id=OWNER_ID, category=None)

    assert events == ["storage.save"]
    assert sessions.calls == 0


@run_async
@pytest.mark.parametrize(
    ("action", "status"),
    [
        ("cancel", "running"),
        ("cancel", "completed"),
        ("cancel", "cancelled"),
        ("claim_retry", "pending"),
        ("claim_retry", "running"),
        ("claim_retry", "completed"),
        ("claim_retry", "cancelled"),
        ("claim_pending", "running"),
        ("claim_pending", "completed"),
        ("claim_pending", "cancelled"),
    ],
)
async def test_illegal_service_action_is_stable_conflict(
    action: str,
    status: str,
) -> None:
    events: list[str] = []
    service = IngestionService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        FakeStorage(events),  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(existing_job=_job(status=status)),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(Exception) as captured:
        await getattr(service, action)(JOB_ID)

    assert getattr(captured.value, "code", None) == "INGESTION_JOB_STATE_CONFLICT"
    assert getattr(captured.value, "status_code", None) == 409


@run_async
@pytest.mark.parametrize("action", ["cancel", "claim_retry", "claim_pending"])
async def test_missing_job_is_stable_not_found(action: str) -> None:
    events: list[str] = []
    service = IngestionService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        FakeStorage(events),  # type: ignore[arg-type]
        repository_factory=RepositoryFactory(existing_job=None),  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    with pytest.raises(Exception) as captured:
        await getattr(service, action)(JOB_ID)

    assert getattr(captured.value, "code", None) == "INGESTION_JOB_NOT_FOUND"
    assert getattr(captured.value, "status_code", None) == 404


@run_async
async def test_read_and_cancel_results_never_expose_internal_fields() -> None:
    events: list[str] = []
    cancelled = _job(status="cancelled")
    repositories = RepositoryFactory(
        existing_job=cancelled,
        cancelled_job=cancelled,
        documents=[_document()],
    )
    service = IngestionService(
        FakeSessionFactory(events),  # type: ignore[arg-type]
        FakeStorage(events),  # type: ignore[arg-type]
        repository_factory=repositories,  # type: ignore[arg-type]
        now=lambda: NOW,
    )

    page = await service.list_documents(status="pending", page=1, page_size=20)
    job = await service.get_job(JOB_ID)
    cancelled_dto = await service.cancel(JOB_ID)

    serialized = page.model_dump_json() + job.model_dump_json() + cancelled_dto.model_dump_json()
    assert "storage_path" not in serialized
    assert "request_payload" not in serialized
    assert "private.pdf" not in serialized
    assert "绝不能公开" not in serialized
