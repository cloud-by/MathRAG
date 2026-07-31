"""文档接收、查询与导入任务状态转换用例。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.modules.ingestion.errors import (
    DocumentDuplicateError,
    IngestionJobNotFoundError,
    IngestionJobStateConflictError,
    IngestionPersistenceError,
)
from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.repository import IngestionRepository, JobSnapshot
from app.modules.ingestion.schemas import (
    DocumentAccepted,
    DocumentPage,
    DocumentRead,
    IngestionJobRead,
)
from app.modules.ingestion.storage import StoredUpload, UploadReadable


DOCUMENT_DUPLICATE_CONSTRAINTS = frozenset(
    {
        "uq_documents_owner_id_sha256",
        "uq_documents_storage_path",
        "uq_ingestion_jobs_document_id_job_type",
    }
)
DOCUMENT_STATUSES = frozenset(
    {"pending", "processing", "ready", "failed", "archived"}
)


class UploadStorageProtocol(Protocol):
    async def save_upload(self, upload: UploadReadable) -> StoredUpload: ...

    async def delete_upload(self, relative_path: str) -> None: ...

    def release_upload(self, relative_path: str) -> None: ...


class IngestionRepositoryProtocol(Protocol):
    def add_document(self, document: Document) -> None: ...

    def add_job(self, job: IngestionJob) -> None: ...

    async def list_documents(
        self,
        *,
        offset: int,
        limit: int,
        status: str | None,
    ) -> tuple[list[Document], int]: ...

    async def get_job(self, job_id: UUID) -> IngestionJob | None: ...

    async def claim_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None: ...

    async def claim_retry(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None: ...

    async def cancel_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> IngestionJob | None: ...


class IngestionService:
    """用短事务协调受控文件与导入任务元数据。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        storage: UploadStorageProtocol,
        *,
        repository_factory: Callable[
            [AsyncSession], IngestionRepositoryProtocol
        ] = IngestionRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._repository_factory = repository_factory
        self._now = now or (lambda: datetime.now(UTC))

    async def accept_pdf(
        self,
        upload: UploadReadable,
        *,
        owner_id: UUID,
        category: str | None = None,
    ) -> DocumentAccepted:
        """先完整保存并校验 PDF，再以单个短事务创建 document 与 job。"""
        request_payload = _pdf_request_payload(category)
        stored = await self._storage.save_upload(upload)
        timestamp = self._now()
        document = Document(
            id=uuid4(),
            owner_id=owner_id,
            original_name=stored.original_name,
            storage_path=stored.relative_path,
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
            sha256=stored.sha256,
            status="pending",
            created_at=timestamp,
            updated_at=timestamp,
        )
        job = IngestionJob(
            id=uuid4(),
            requested_by=owner_id,
            document_id=document.id,
            job_type="pdf",
            status="pending",
            progress=0,
            request_payload=request_payload,
            attempt_count=0,
            error_code=None,
            error_message=None,
            started_at=None,
            finished_at=None,
            created_at=timestamp,
            updated_at=timestamp,
        )

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    repository = self._repository_factory(session)
                    repository.add_document(document)
                    repository.add_job(job)
                    await session.flush()
                    accepted = DocumentAccepted(
                        document=DocumentRead.model_validate(document),
                        job=IngestionJobRead.model_validate(job),
                    )
        except BaseException as error:
            try:
                await self._cleanup_saved_upload(stored.relative_path)
            except asyncio.CancelledError:
                raise
            except BaseException:
                # 清理已尽力执行；不能用次生文件错误遮蔽原始数据库语义。
                pass
            if isinstance(error, IntegrityError):
                if _constraint_name(error) in DOCUMENT_DUPLICATE_CONSTRAINTS:
                    raise DocumentDuplicateError() from None
                raise IngestionPersistenceError() from None
            if isinstance(error, SQLAlchemyError):
                raise IngestionPersistenceError() from None
            raise
        self._storage.release_upload(stored.relative_path)
        return accepted

    async def list_documents(
        self,
        *,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> DocumentPage:
        _validate_document_page(status=status, page=page, page_size=page_size)
        async with self._session_factory() as session:
            documents, total = await self._repository_factory(session).list_documents(
                offset=(page - 1) * page_size,
                limit=page_size,
                status=status,
            )
            items = [DocumentRead.model_validate(document) for document in documents]
        return DocumentPage(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_job(self, job_id: UUID) -> IngestionJobRead:
        async with self._session_factory() as session:
            job = await self._repository_factory(session).get_job(job_id)
            if job is None:
                raise IngestionJobNotFoundError()
            return IngestionJobRead.model_validate(job)

    async def cancel(self, job_id: UUID) -> IngestionJobRead:
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                cancelled = await repository.cancel_pending(job_id, self._now())
                if cancelled is None:
                    await _raise_job_action_error(repository, job_id)
                return IngestionJobRead.model_validate(cancelled)

    async def claim_pending(self, job_id: UUID) -> JobSnapshot:
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                snapshot = await repository.claim_pending(job_id, self._now())
                if snapshot is None:
                    await _raise_job_action_error(repository, job_id)
                return snapshot

    async def claim_retry(self, job_id: UUID) -> JobSnapshot:
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                snapshot = await repository.claim_retry(job_id, self._now())
                if snapshot is None:
                    await _raise_job_action_error(repository, job_id)
                return snapshot

    async def retry(self, job_id: UUID) -> JobSnapshot:
        """API/CLI 兼容入口；成功后由调用方安排 resume_retry。"""
        return await self.claim_retry(job_id)

    async def _cleanup_saved_upload(self, relative_path: str) -> None:
        """即使外层任务收到取消，也尽力等待受控清理结束。"""
        cleanup = asyncio.create_task(self._storage.delete_upload(relative_path))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            try:
                await cleanup
            except BaseException:
                pass
            raise


async def _raise_job_action_error(
    repository: IngestionRepositoryProtocol,
    job_id: UUID,
) -> None:
    existing = await repository.get_job(job_id)
    if existing is None:
        raise IngestionJobNotFoundError()
    raise IngestionJobStateConflictError()


def _pdf_request_payload(category: str | None) -> dict[str, object]:
    if category is None:
        return {}
    if not isinstance(category, str):
        raise AppError(
            code="REQUEST_VALIDATION_FAILED",
            message="文档分类无效。",
            status_code=422,
        )
    normalized = category.strip()
    if not normalized:
        return {}
    if len(normalized) > 100:
        raise AppError(
            code="REQUEST_VALIDATION_FAILED",
            message="文档分类无效。",
            status_code=422,
        )
    return {"category": normalized}


def _validate_document_page(
    *,
    status: str | None,
    page: int,
    page_size: int,
) -> None:
    if (
        (status is not None and status not in DOCUMENT_STATUSES)
        or type(page) is not int
        or page < 1
        or type(page_size) is not int
        or not 1 <= page_size <= 100
    ):
        raise AppError(
            code="REQUEST_VALIDATION_FAILED",
            message="文档筛选或分页参数无效。",
            status_code=422,
        )


def _constraint_name(error: IntegrityError) -> str | None:
    """仅读取驱动提供的约束名，不依赖可能含 SQL/值的异常文本。"""
    current: BaseException | None = error.orig
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = getattr(current, "constraint_name", None)
        if isinstance(name, str):
            return name
        current = current.__cause__ or current.__context__
    return None
