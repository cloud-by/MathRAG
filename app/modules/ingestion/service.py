"""文档接收、查询与导入任务状态转换用例。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID, uuid4

from openai import APIError, APITimeoutError, RateLimitError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.core.errors import AppError
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    get_embedding_provider,
    validate_and_normalize_vector,
)
from app.modules.ingestion.errors import (
    DocumentDuplicateError,
    IngestionJobNotFoundError,
    IngestionJobStateConflictError,
    IngestionPersistenceError,
)
from app.modules.ingestion.extractors import ExtractedPDF, extract_pdf_text
from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.repository import (
    DocumentSnapshot,
    IngestionRepository,
    JobSnapshot,
    PipelineChunkSnapshot,
)
from app.modules.ingestion.schemas import (
    DocumentAccepted,
    DocumentPage,
    DocumentRead,
    IngestionJobPage,
    IngestionJobRead,
)
from app.modules.ingestion.storage import (
    StoredUpload,
    UploadReadable,
    resolve_stored_path,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    KnowledgeSearchError,
)
from app.services.knowledge_extractor import (
    KnowledgeDraft,
    extract_knowledge_drafts,
)
from app.services.math_knowledge_importer import (
    SOURCE_REGISTRY,
    chunk_text,
    discover_documents,
)


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
INGESTION_JOB_STATUSES = frozenset(
    {"pending", "running", "completed", "failed", "cancelled"}
)
INGESTION_JOB_TYPES = frozenset({"text", "pdf", "web", "reindex"})
PDF_MIME_TYPE = "application/pdf"


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

    async def list_jobs(
        self,
        *,
        offset: int,
        limit: int,
        status: str | None,
        job_type: str | None,
        document_id: UUID | None,
    ) -> tuple[list[IngestionJob], int]: ...

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

    async def get_document_snapshot(
        self,
        document_id: UUID,
    ) -> DocumentSnapshot | None: ...

    async def list_pipeline_chunks(
        self,
        job_id: UUID,
    ) -> list[PipelineChunkSnapshot]: ...

    async def create_pipeline_items(
        self,
        snapshot: JobSnapshot,
        drafts: list[dict[str, object]],
    ) -> list[PipelineChunkSnapshot]: ...

    async def finalize_pipeline(
        self,
        *,
        snapshot: JobSnapshot,
        chunks: list[PipelineChunkSnapshot],
        vectors: list[list[float]],
        model: str,
        now: datetime,
    ) -> bool: ...

    async def fail_pipeline(
        self,
        *,
        snapshot: JobSnapshot,
        code: str,
        message: str,
        now: datetime,
    ) -> bool: ...

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
        embedding_provider: EmbeddingProvider | None = None,
        draft_extractor: Callable[
            [str, str | None], list[KnowledgeDraft]
        ] = extract_knowledge_drafts,
        pdf_extractor: Callable[..., ExtractedPDF] = extract_pdf_text,
        web_source_loader: Callable[[Mapping[str, object]], str] | None = None,
        upload_root: Path = settings.UPLOAD_DIR,
        max_pdf_pages: int = settings.MAX_PDF_PAGES,
        max_ingestion_text_chars: int = settings.MAX_INGESTION_TEXT_CHARS,
        embedding_batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if max_pdf_pages <= 0:
            raise ValueError("max_pdf_pages 必须大于 0")
        if max_ingestion_text_chars <= 0:
            raise ValueError("max_ingestion_text_chars 必须大于 0")
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size 必须大于 0")
        self._session_factory = session_factory
        self._storage = storage
        self._repository_factory = repository_factory
        self._embedding_provider = embedding_provider
        self._draft_extractor = draft_extractor
        self._pdf_extractor = pdf_extractor
        self._web_source_loader = web_source_loader or _load_web_source_text
        self._upload_root = upload_root
        self._max_pdf_pages = max_pdf_pages
        self._max_ingestion_text_chars = max_ingestion_text_chars
        self._embedding_batch_size = embedding_batch_size
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

    async def accept_local_pdf(
        self,
        path: Path,
        *,
        owner_id: UUID,
        category: str | None = None,
    ) -> DocumentAccepted:
        """让本地 CLI 复用在线上传的校验、摘要和短事务路径。"""
        upload = _LocalPathUpload(Path(path))
        try:
            return await self.accept_pdf(
                upload,
                owner_id=owner_id,
                category=category,
            )
        finally:
            await upload.close()

    async def accept_web(
        self,
        *,
        requested_by: UUID,
        sources: Sequence[str],
        keywords: Sequence[str],
        limit_per_source: int = 3,
        category: str | None = None,
        delay_seconds: float = 1.0,
        max_chunk_chars: int = 6000,
    ) -> IngestionJobRead:
        """以短事务创建一个不含凭据的网页导入任务。"""
        payload = _web_request_payload(
            sources=sources,
            keywords=keywords,
            limit_per_source=limit_per_source,
            category=category,
            delay_seconds=delay_seconds,
            max_chunk_chars=max_chunk_chars,
        )
        timestamp = self._now()
        job = IngestionJob(
            id=uuid4(),
            requested_by=requested_by,
            document_id=None,
            job_type="web",
            status="pending",
            progress=0,
            request_payload=payload,
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
                    self._repository_factory(session).add_job(job)
                    await session.flush()
                    result = IngestionJobRead.model_validate(job)
        except SQLAlchemyError:
            raise IngestionPersistenceError() from None
        return result

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

    async def list_jobs(
        self,
        *,
        status: str | None = None,
        job_type: str | None = None,
        document_id: UUID | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> IngestionJobPage:
        _validate_job_page(
            status=status,
            job_type=job_type,
            document_id=document_id,
            offset=offset,
            limit=limit,
        )
        async with self._session_factory() as session:
            jobs, total = await self._repository_factory(session).list_jobs(
                status=status,
                job_type=job_type,
                document_id=document_id,
                offset=offset,
                limit=limit,
            )
            items = [IngestionJobRead.model_validate(job) for job in jobs]
        return IngestionJobPage(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
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

    async def run_pending(self, job_id: UUID) -> None:
        """认领 pending 任务，并在短事务之间执行所有外部调用。"""
        snapshot = await self.claim_pending(job_id)
        await self._execute_pipeline(snapshot)

    async def resume_retry(self, snapshot: JobSnapshot) -> None:
        """继续已经由 API/CLI 原子认领的重试 attempt。"""
        if not isinstance(snapshot, JobSnapshot):
            raise TypeError("snapshot 必须是 JobSnapshot")
        await self._execute_pipeline(snapshot)

    async def _execute_pipeline(self, snapshot: JobSnapshot) -> None:
        cancelled = False
        try:
            document, chunks = await self._load_pipeline_state(snapshot)
            if not chunks:
                text = await self._read_source(snapshot, document)
                category = _category_from_payload(snapshot.request_payload)
                drafts = await self._extract_drafts(
                    snapshot,
                    text,
                    category,
                )
                _validate_drafts(drafts)
                chunks = await self._persist_drafts(snapshot, drafts)
            vectors, model = await self._embed_chunks(chunks)
            await self._finalize(snapshot, chunks, vectors, model)
        except asyncio.CancelledError as exc:
            cancelled = True
            await self._record_pipeline_failure(snapshot, exc)
        except Exception as exc:
            await self._record_pipeline_failure(snapshot, exc)
        if cancelled:
            raise asyncio.CancelledError

    async def _load_pipeline_state(
        self,
        snapshot: JobSnapshot,
    ) -> tuple[DocumentSnapshot | None, list[PipelineChunkSnapshot]]:
        async with self._session_factory() as session:
            repository = self._repository_factory(session)
            document = None
            if snapshot.document_id is not None:
                document = await repository.get_document_snapshot(
                    snapshot.document_id
                )
            chunks = await repository.list_pipeline_chunks(snapshot.job_id)
        if snapshot.job_type == "pdf" and (
            snapshot.document_id is None or document is None
        ):
            raise _PipelineContractError()
        if snapshot.job_type == "web" and snapshot.document_id is not None:
            raise _PipelineContractError()
        if snapshot.job_type not in {"pdf", "web"}:
            raise _PipelineContractError()
        return document, chunks

    async def _read_source(
        self,
        snapshot: JobSnapshot,
        document: DocumentSnapshot | None,
    ) -> str:
        if snapshot.job_type == "web":
            text = await asyncio.to_thread(
                self._web_source_loader,
                snapshot.request_payload,
            )
            normalized = text.strip()
            if not normalized or len(normalized) > self._max_ingestion_text_chars:
                raise _PipelineContractError()
            return normalized
        if document is None or snapshot.document_id != document.document_id:
            raise _PipelineContractError()
        source_path = resolve_stored_path(
            self._upload_root,
            document.storage_path,
        )
        extracted = await asyncio.to_thread(
            self._pdf_extractor,
            source_path,
            max_pages=self._max_pdf_pages,
        )
        text = extracted.text.strip()
        if not text or len(text) > self._max_ingestion_text_chars:
            raise _PipelinePdfError()
        return text

    async def _extract_drafts(
        self,
        snapshot: JobSnapshot,
        text: str,
        category: str | None,
    ) -> list[KnowledgeDraft]:
        segments = [text]
        if snapshot.job_type == "web":
            options = _web_options_from_payload(snapshot.request_payload)
            segments = chunk_text(text, max_chars=options.max_chunk_chars)
        drafts: list[KnowledgeDraft] = []
        for segment in segments:
            extracted = await asyncio.to_thread(
                self._draft_extractor,
                segment,
                category,
            )
            _validate_drafts(extracted)
            drafts.extend(extracted)
        return drafts

    async def _persist_drafts(
        self,
        snapshot: JobSnapshot,
        drafts: list[KnowledgeDraft],
    ) -> list[PipelineChunkSnapshot]:
        values = [draft.to_values() for draft in drafts]
        async with self._session_factory() as session:
            async with session.begin():
                chunks = await self._repository_factory(
                    session
                ).create_pipeline_items(snapshot, values)
                if not chunks:
                    raise _PipelineContractError()
                return chunks

    async def _embed_chunks(
        self,
        chunks: list[PipelineChunkSnapshot],
    ) -> tuple[list[list[float]], str]:
        provider = self._embedding_provider or get_embedding_provider()
        try:
            dimensions = provider.dimensions
            raw_model = provider.model
        except Exception:
            raise EmbeddingResponseError("Embedding Provider 契约无效") from None
        if dimensions != 1024:
            raise EmbeddingResponseError("Embedding 维度与导入契约不一致")
        if not isinstance(raw_model, str):
            raise EmbeddingResponseError("Embedding 模型标识无效")
        model = raw_model.strip()
        if not model or len(model) > 128:
            raise EmbeddingResponseError("Embedding 模型标识无效")

        vectors: list[list[float]] = []
        for offset in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[offset : offset + self._embedding_batch_size]
            returned = await provider.embed_texts(
                [chunk.retrieval_text for chunk in batch]
            )
            if not isinstance(returned, list) or len(returned) != len(batch):
                raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
            for vector in returned:
                if not isinstance(vector, list):
                    raise EmbeddingResponseError("Embedding 返回向量无效")
                try:
                    vectors.append(validate_and_normalize_vector(vector, 1024))
                except EmbeddingResponseError:
                    raise
                except Exception:
                    raise EmbeddingResponseError("Embedding 返回向量无效") from None
        if len(vectors) != len(chunks):
            raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
        return vectors, model

    async def _finalize(
        self,
        snapshot: JobSnapshot,
        chunks: list[PipelineChunkSnapshot],
        vectors: list[list[float]],
        model: str,
    ) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                completed = await self._repository_factory(
                    session
                ).finalize_pipeline(
                    snapshot=snapshot,
                    chunks=chunks,
                    vectors=vectors,
                    model=model,
                    now=self._now(),
                )
                if not completed:
                    raise _PipelineCASConflict()

    async def _record_pipeline_failure(
        self,
        snapshot: JobSnapshot,
        error: BaseException,
    ) -> None:
        code, message = _map_pipeline_error(error)

        async def write_failure() -> None:
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        await self._repository_factory(session).fail_pipeline(
                            snapshot=snapshot,
                            code=code,
                            message=message,
                            now=self._now(),
                        )
            except BaseException:
                # 数据库不可用时可能无法收口；不能用第二个异常泄露原始细节。
                return

        task = asyncio.create_task(write_failure())
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task

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


def _validate_job_page(
    *,
    status: str | None,
    job_type: str | None,
    document_id: UUID | None,
    offset: int,
    limit: int,
) -> None:
    if (
        (status is not None and status not in INGESTION_JOB_STATUSES)
        or (job_type is not None and job_type not in INGESTION_JOB_TYPES)
        or (document_id is not None and not isinstance(document_id, UUID))
        or type(offset) is not int
        or offset < 0
        or type(limit) is not int
        or not 1 <= limit <= 100
    ):
        raise AppError(
            code="REQUEST_VALIDATION_FAILED",
            message="导入任务筛选或分页参数无效。",
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


class _PipelineContractError(Exception):
    """持久化快照与 pipeline 前置条件不一致。"""


class _PipelineCASConflict(Exception):
    """当前 attempt 已失效或写回快照发生变化。"""


class _PipelinePdfError(Exception):
    """PDF 抽取结果不满足在线导入长度契约。"""


class _LocalPathUpload:
    """把本地 PDF 适配为 UploadStorage 的异步读取协议。"""

    content_type = PDF_MIME_TYPE

    def __init__(self, path: Path) -> None:
        self._path = path
        self.filename = path.name
        self._file: BinaryIO | None = None

    async def read(self, size: int = -1) -> bytes:
        if self._file is None:
            self._file = await asyncio.to_thread(self._path.open, "rb")
        return await asyncio.to_thread(self._file.read, size)

    async def close(self) -> None:
        file = self._file
        self._file = None
        if file is not None:
            await asyncio.to_thread(file.close)


class _WebOptions:
    """已校验的网页任务执行参数。"""

    def __init__(
        self,
        *,
        sources: list[str],
        keywords: list[str],
        limit_per_source: int,
        delay_seconds: float,
        max_chunk_chars: int,
    ) -> None:
        self.sources = sources
        self.keywords = keywords
        self.limit_per_source = limit_per_source
        self.delay_seconds = delay_seconds
        self.max_chunk_chars = max_chunk_chars


def _category_from_payload(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("category")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _web_request_payload(
    *,
    sources: Sequence[str],
    keywords: Sequence[str],
    limit_per_source: int,
    category: str | None,
    delay_seconds: float,
    max_chunk_chars: int,
) -> dict[str, object]:
    raw: dict[str, object] = {
        "sources": list(sources),
        "keywords": list(keywords),
        "limit_per_source": limit_per_source,
        "delay_seconds": delay_seconds,
        "max_chunk_chars": max_chunk_chars,
    }
    normalized_category = _pdf_request_payload(category).get("category")
    if normalized_category is not None:
        raw["category"] = normalized_category
    options = _web_options_from_payload(raw)
    payload: dict[str, object] = {
        "sources": options.sources,
        "keywords": options.keywords,
        "limit_per_source": options.limit_per_source,
        "delay_seconds": options.delay_seconds,
        "max_chunk_chars": options.max_chunk_chars,
    }
    if normalized_category is not None:
        payload["category"] = normalized_category
    return payload


def _web_options_from_payload(payload: object) -> _WebOptions:
    if not isinstance(payload, Mapping):
        raise _PipelineContractError()
    sources_value = payload.get("sources")
    keywords_value = payload.get("keywords")
    if not isinstance(sources_value, list) or not isinstance(keywords_value, list):
        raise _PipelineContractError()
    sources = _normalized_string_list(sources_value, max_items=10, max_length=64)
    keywords = _normalized_string_list(keywords_value, max_items=20, max_length=200)
    if any(source not in SOURCE_REGISTRY for source in sources):
        raise _PipelineContractError()

    limit = payload.get("limit_per_source")
    max_chars = payload.get("max_chunk_chars")
    delay = payload.get("delay_seconds")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise _PipelineContractError()
    if type(max_chars) is not int or not 200 <= max_chars <= 20_000:
        raise _PipelineContractError()
    if isinstance(delay, bool) or not isinstance(delay, (int, float)):
        raise _PipelineContractError()
    normalized_delay = float(delay)
    if not math.isfinite(normalized_delay) or not 0 <= normalized_delay <= 60:
        raise _PipelineContractError()
    return _WebOptions(
        sources=sources,
        keywords=keywords,
        limit_per_source=limit,
        delay_seconds=normalized_delay,
        max_chunk_chars=max_chars,
    )


def _normalized_string_list(
    values: list[object],
    *,
    max_items: int,
    max_length: int,
) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise _PipelineContractError()
        text = value.strip()
        if not text or len(text) > max_length or text in normalized:
            if text in normalized:
                continue
            raise _PipelineContractError()
        normalized.append(text)
    if not normalized or len(normalized) > max_items:
        raise _PipelineContractError()
    return normalized


def _load_web_source_text(payload: Mapping[str, object]) -> str:
    """抓取网页来源并返回纯文本，不写 seed/error JSONL。"""
    options = _web_options_from_payload(payload)
    documents = discover_documents(
        sources=options.sources,
        keywords=options.keywords,
        limit_per_source=options.limit_per_source,
        delay_seconds=options.delay_seconds,
        error_path=None,
    )
    return "\n\n".join(document.text for document in documents if document.text)


def _validate_drafts(drafts: object) -> None:
    if (
        not isinstance(drafts, list)
        or not drafts
        or any(not isinstance(draft, KnowledgeDraft) for draft in drafts)
    ):
        raise _PipelineContractError()


def _map_pipeline_error(error: BaseException) -> tuple[str, str]:
    """只返回稳定常量，不复制上游异常文本、SQL 或路径。"""
    if isinstance(error, SQLAlchemyError):
        return (
            "INGESTION_DATABASE_UNAVAILABLE",
            "数据库服务暂时不可用。",
        )
    if isinstance(error, RateLimitError):
        return (
            "INGESTION_LLM_RATE_LIMITED",
            "知识抽取服务请求过于频繁，请稍后重试。",
        )
    if isinstance(error, (APITimeoutError, TimeoutError)):
        return (
            "INGESTION_UPSTREAM_TIMEOUT",
            "上游服务响应超时。",
        )
    if isinstance(
        error,
        (
            EmbeddingInputError,
            EmbeddingResponseError,
            EmbeddingUnavailableError,
            KnowledgeSearchError,
        ),
    ):
        return (
            "INGESTION_EMBEDDING_UNAVAILABLE",
            "知识向量化服务暂时不可用。",
        )
    if isinstance(error, AppError) and error.code.startswith("DOCUMENT_PDF_"):
        return (
            "INGESTION_PDF_INVALID",
            "PDF 文档无法读取或解析。",
        )
    if isinstance(error, (OSError, _PipelinePdfError)):
        return (
            "INGESTION_PDF_INVALID",
            "PDF 文档无法读取或解析。",
        )
    if isinstance(error, (APIError, ValueError, RuntimeError)):
        return (
            "INGESTION_LLM_UNAVAILABLE",
            "知识抽取服务暂时不可用。",
        )
    return (
        "INGESTION_INTERNAL_ERROR",
        "导入任务处理失败。",
    )
