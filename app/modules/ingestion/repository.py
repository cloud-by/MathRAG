"""导入文档读取和任务 CAS 状态机仓储。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.ingestion.models import Document, IngestionJob


@dataclass(frozen=True)
class JobSnapshot:
    """可安全跨事务传递的纯数据任务快照。"""

    job_id: UUID
    document_id: UUID | None
    requested_by: UUID | None
    job_type: str
    attempt_count: int
    request_payload: Mapping[str, object]


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
        return None if job is None else _snapshot(job)

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
        return None if job is None else _snapshot(job)

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
