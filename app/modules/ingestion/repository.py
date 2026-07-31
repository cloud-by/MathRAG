"""导入文档读取和任务 CAS 状态机仓储。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ingestion.models import Document, IngestionJob
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.rendering import build_answer_context, build_retrieval_text


@dataclass(frozen=True)
class JobSnapshot:
    """可安全跨事务传递的纯数据任务快照。"""

    job_id: UUID
    document_id: UUID | None
    requested_by: UUID | None
    job_type: str
    attempt_count: int
    request_payload: Mapping[str, object]


@dataclass(frozen=True)
class DocumentSnapshot:
    """受控文档源的跨事务只读快照。"""

    document_id: UUID
    storage_path: str
    status: str


@dataclass(frozen=True)
class PipelineChunkSnapshot:
    """一次向量化写回所需的不可变分块快照。"""

    chunk_id: UUID
    item_id: UUID
    retrieval_text: str


class IngestionRepository:
    """仅操作调用方会话，不提交、回滚或关闭事务。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add_document(self, document: Document) -> None:
        self._session.add(document)

    def add_job(self, job: IngestionJob) -> None:
        self._session.add(job)

    async def list_documents(
        self,
        *,
        offset: int,
        limit: int,
        status: str | None,
    ) -> tuple[list[Document], int]:
        filters = [] if status is None else [Document.status == status]
        documents = list(
            (
                await self._session.scalars(
                    select(Document)
                    .where(*filters)
                    .order_by(Document.created_at.desc(), Document.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        )
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(Document).where(*filters)
            )
            or 0
        )
        return documents, total

    async def list_jobs(
        self,
        *,
        offset: int,
        limit: int,
        status: str | None,
        job_type: str | None,
        document_id: UUID | None,
    ) -> tuple[list[IngestionJob], int]:
        filters = []
        if status is not None:
            filters.append(IngestionJob.status == status)
        if job_type is not None:
            filters.append(IngestionJob.job_type == job_type)
        if document_id is not None:
            filters.append(IngestionJob.document_id == document_id)

        jobs = list(
            (
                await self._session.scalars(
                    select(IngestionJob)
                    .where(*filters)
                    .order_by(IngestionJob.created_at.desc(), IngestionJob.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        )
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(IngestionJob).where(*filters)
            )
            or 0
        )
        return jobs, total

    async def get_job(self, job_id: UUID) -> IngestionJob | None:
        return await self._session.scalar(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )

    async def claim_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None:
        """以单 SQL 认领 pending；首次认领 attempt_count 从 0 变为 1。"""
        job = await self._session.scalar(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "pending",
            )
            .values(
                status="running",
                progress=0,
                attempt_count=IngestionJob.attempt_count + 1,
                error_code=None,
                error_message=None,
                started_at=now,
                finished_at=None,
                updated_at=now,
            )
            .returning(IngestionJob)
        )
        if job is None:
            return None
        await self._mark_document_processing(job.document_id, now)
        return _snapshot(job)

    async def claim_retry(
        self,
        job_id: UUID,
        now: datetime,
    ) -> JobSnapshot | None:
        """以单 SQL 将 failed 任务切回 running 并推进 attempt。"""
        job = await self._session.scalar(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "failed",
            )
            .values(
                status="running",
                progress=0,
                attempt_count=IngestionJob.attempt_count + 1,
                error_code=None,
                error_message=None,
                started_at=now,
                finished_at=None,
                updated_at=now,
            )
            .returning(IngestionJob)
        )
        if job is None:
            return None
        await self._mark_document_processing(job.document_id, now)
        return _snapshot(job)

    async def get_document_snapshot(
        self,
        document_id: UUID,
    ) -> DocumentSnapshot | None:
        row = (
            await self._session.execute(
                select(Document.id, Document.storage_path, Document.status).where(
                    Document.id == document_id
                )
            )
        ).one_or_none()
        if row is None:
            return None
        return DocumentSnapshot(
            document_id=row.id,
            storage_path=row.storage_path,
            status=row.status,
        )

    async def list_pipeline_chunks(
        self,
        job_id: UUID,
    ) -> list[PipelineChunkSnapshot]:
        rows = (
            await self._session.execute(
                select(
                    KnowledgeChunk.id,
                    KnowledgeChunk.knowledge_item_id,
                    KnowledgeChunk.retrieval_text,
                )
                .join(
                    KnowledgeItem,
                    KnowledgeItem.id == KnowledgeChunk.knowledge_item_id,
                )
                .where(
                    KnowledgeItem.ingestion_job_id == job_id,
                    KnowledgeItem.status.in_(("indexing", "failed")),
                    KnowledgeChunk.status.in_(("pending", "failed")),
                )
                .order_by(KnowledgeChunk.chunk_index, KnowledgeChunk.id)
            )
        ).all()
        return [
            PipelineChunkSnapshot(
                chunk_id=row.id,
                item_id=row.knowledge_item_id,
                retrieval_text=row.retrieval_text,
            )
            for row in rows
        ]

    async def create_pipeline_items(
        self,
        snapshot: JobSnapshot,
        drafts: Sequence[Mapping[str, object]],
    ) -> list[PipelineChunkSnapshot]:
        """为首次执行批量创建关联条目；调用方拥有提交边界。"""
        existing = await self._session.scalar(
            select(KnowledgeItem.id)
            .where(KnowledgeItem.ingestion_job_id == snapshot.job_id)
            .limit(1)
        )
        if existing is not None:
            raise ValueError("导入任务已经存在知识条目")

        highest_index = None
        if snapshot.document_id is not None:
            highest_index = await self._session.scalar(
                select(func.max(KnowledgeChunk.chunk_index)).where(
                    KnowledgeChunk.document_id == snapshot.document_id
                )
            )
        next_index = int(highest_index) + 1 if highest_index is not None else 0
        chunks: list[PipelineChunkSnapshot] = []
        for offset, draft in enumerate(drafts):
            values = dict(draft)
            item = KnowledgeItem(
                owner_id=snapshot.requested_by,
                ingestion_job_id=snapshot.job_id,
                visibility="public",
                status="indexing",
                revision=1,
                **values,
            )
            self._session.add(item)
            await self._session.flush()

            retrieval_text = build_retrieval_text(values)
            chunk = KnowledgeChunk(
                knowledge_item_id=item.id,
                document_id=snapshot.document_id,
                chunk_index=next_index + offset,
                retrieval_text=retrieval_text,
                answer_context=build_answer_context(values),
                embedding=None,
                embedding_model=None,
                metadata_={"ingestion_job_id": str(snapshot.job_id)},
                status="pending",
            )
            self._session.add(chunk)
            await self._session.flush()
            chunks.append(
                PipelineChunkSnapshot(
                    chunk_id=chunk.id,
                    item_id=item.id,
                    retrieval_text=retrieval_text,
                )
            )
        return chunks

    async def finalize_pipeline(
        self,
        *,
        snapshot: JobSnapshot,
        chunks: Sequence[PipelineChunkSnapshot],
        vectors: Sequence[Sequence[float]],
        model: str,
        now: datetime,
    ) -> bool:
        """以 job attempt 与分块正文双重 CAS 原子完成导入。"""
        if len(chunks) != len(vectors) or not chunks:
            return False
        job = await self._session.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.id == snapshot.job_id,
                IngestionJob.status == "running",
                IngestionJob.attempt_count == snapshot.attempt_count,
            )
            .with_for_update()
        )
        if job is None:
            return False

        item_ids = sorted(
            {chunk.item_id for chunk in chunks},
            key=lambda value: value.int,
        )
        locked_item_ids = list(
            (
                await self._session.scalars(
                    select(KnowledgeItem.id)
                    .where(
                        KnowledgeItem.id.in_(item_ids),
                        KnowledgeItem.ingestion_job_id == snapshot.job_id,
                        KnowledgeItem.status.in_(("indexing", "failed")),
                    )
                    .order_by(KnowledgeItem.id)
                    .with_for_update()
                )
            ).all()
        )
        if set(locked_item_ids) != set(item_ids):
            return False

        for chunk, vector in zip(chunks, vectors, strict=True):
            result = await self._session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.id == chunk.chunk_id,
                    KnowledgeChunk.knowledge_item_id == chunk.item_id,
                    KnowledgeChunk.document_id == snapshot.document_id,
                    KnowledgeChunk.retrieval_text == chunk.retrieval_text,
                    KnowledgeChunk.status.in_(("pending", "failed")),
                )
                .values(
                    embedding=list(vector),
                    embedding_model=model,
                    status="ready",
                )
            )
            if result.rowcount != 1:
                return False

        await self._session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id.in_(item_ids),
                KnowledgeItem.ingestion_job_id == snapshot.job_id,
                KnowledgeItem.status.in_(("indexing", "failed")),
            )
            .values(status="ready", updated_at=now)
        )
        remaining = int(
            await self._session.scalar(
                select(func.count())
                .select_from(KnowledgeChunk)
                .join(
                    KnowledgeItem,
                    KnowledgeItem.id == KnowledgeChunk.knowledge_item_id,
                )
                .where(
                    KnowledgeItem.ingestion_job_id == snapshot.job_id,
                    KnowledgeChunk.status != "ready",
                )
            )
            or 0
        )
        if remaining:
            return False
        if snapshot.document_id is not None:
            document_result = await self._session.execute(
                update(Document)
                .where(
                    Document.id == snapshot.document_id,
                    Document.status == "processing",
                )
                .values(status="ready", updated_at=now)
            )
            if document_result.rowcount != 1:
                return False
        return await self.complete(
            snapshot.job_id,
            snapshot.attempt_count,
            now,
        )

    async def fail_pipeline(
        self,
        *,
        snapshot: JobSnapshot,
        code: str,
        message: str,
        now: datetime,
    ) -> bool:
        """仅让当前 attempt 收口失败，不覆盖 ready/completed 数据。"""
        job = await self._session.scalar(
            select(IngestionJob)
            .where(
                IngestionJob.id == snapshot.job_id,
                IngestionJob.status == "running",
                IngestionJob.attempt_count == snapshot.attempt_count,
            )
            .with_for_update()
        )
        if job is None:
            return False
        await self._session.execute(
            update(KnowledgeChunk)
            .where(
                KnowledgeChunk.knowledge_item_id.in_(
                    select(KnowledgeItem.id).where(
                        KnowledgeItem.ingestion_job_id == snapshot.job_id,
                        KnowledgeItem.status.in_(("indexing", "failed")),
                    )
                ),
                KnowledgeChunk.status.in_(("pending", "failed")),
            )
            .values(status="failed", embedding=None, embedding_model=None)
        )
        await self._session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.ingestion_job_id == snapshot.job_id,
                KnowledgeItem.status.in_(("indexing", "failed")),
            )
            .values(status="failed", updated_at=now)
        )
        if snapshot.document_id is not None:
            await self._session.execute(
                update(Document)
                .where(
                    Document.id == snapshot.document_id,
                    Document.status == "processing",
                )
                .values(status="failed", updated_at=now)
            )
        return await self.fail(
            snapshot.job_id,
            snapshot.attempt_count,
            code,
            message,
            now,
        )

    async def set_progress(
        self,
        job_id: UUID,
        expected_attempt: int,
        progress: int,
    ) -> bool:
        if type(progress) is not int or not 0 <= progress <= 100:
            raise ValueError("progress 必须是 0..100 的整数")
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "running",
                IngestionJob.attempt_count == expected_attempt,
            )
            .values(progress=progress, updated_at=func.now())
        )
        return result.rowcount == 1

    async def complete(
        self,
        job_id: UUID,
        expected_attempt: int,
        now: datetime,
    ) -> bool:
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "running",
                IngestionJob.attempt_count == expected_attempt,
            )
            .values(
                status="completed",
                progress=100,
                error_code=None,
                error_message=None,
                finished_at=now,
                updated_at=now,
            )
        )
        return result.rowcount == 1

    async def fail(
        self,
        job_id: UUID,
        expected_attempt: int,
        code: str,
        message: str,
        now: datetime,
    ) -> bool:
        """写入调用方已脱敏的稳定错误，并兜底限制数据库字段长度。"""
        if not isinstance(code, str) or not isinstance(message, str):
            raise TypeError("code 和 message 必须是字符串")
        result = await self._session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "running",
                IngestionJob.attempt_count == expected_attempt,
            )
            .values(
                status="failed",
                error_code=code[:64],
                error_message=message[:500],
                finished_at=now,
                updated_at=now,
            )
        )
        return result.rowcount == 1

    async def cancel_pending(
        self,
        job_id: UUID,
        now: datetime,
    ) -> IngestionJob | None:
        return await self._session.scalar(
            update(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.status == "pending",
            )
            .values(
                status="cancelled",
                finished_at=now,
                updated_at=now,
            )
            .returning(IngestionJob)
        )

    async def _mark_document_processing(
        self,
        document_id: UUID | None,
        now: datetime,
    ) -> None:
        if document_id is None:
            return
        await self._session.execute(
            update(Document)
            .where(
                Document.id == document_id,
                Document.status.in_(("pending", "failed")),
            )
            .values(status="processing", updated_at=now)
        )


def _snapshot(job: IngestionJob) -> JobSnapshot:
    """复制 JSON 映射，避免快照继续引用 ORM 的可变载荷。"""
    return JobSnapshot(
        job_id=job.id,
        document_id=job.document_id,
        requested_by=job.requested_by,
        job_type=job.job_type,
        attempt_count=job.attempt_count,
        request_payload=dict(job.request_payload),
    )
