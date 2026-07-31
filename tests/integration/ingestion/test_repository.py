"""导入任务仓储的 PostgreSQL 状态机集成测试。"""

from __future__ import annotations

import asyncio
import os
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.repository import IngestionRepository, JobSnapshot
from tests.integration.database_safety import require_test_database_url


NOW = datetime(2026, 7, 31, 10, 0, tzinfo=UTC)


def _document(
    identity: int,
    *,
    created_at: datetime = NOW,
    status: str = "pending",
) -> Document:
    return Document(
        id=UUID(int=identity),
        owner_id=None,
        original_name=f"lesson-{identity}.pdf",
        storage_path=f"2026/07/{UUID(int=identity)}.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
        sha256=f"{identity:064x}",
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


def _job(
    identity: int,
    *,
    document_id: UUID,
    status: str = "pending",
    attempt_count: int = 0,
    progress: int = 0,
    created_at: datetime = NOW,
) -> IngestionJob:
    return IngestionJob(
        id=UUID(int=identity),
        requested_by=None,
        document_id=document_id,
        job_type="pdf",
        status=status,
        progress=progress,
        request_payload={"category": "代数"},
        attempt_count=attempt_count,
        error_code="OLD_ERROR" if status == "failed" else None,
        error_message="旧错误摘要" if status == "failed" else None,
        started_at=NOW - timedelta(minutes=1) if attempt_count else None,
        finished_at=NOW - timedelta(seconds=1) if status == "failed" else None,
        created_at=created_at,
        updated_at=created_at,
    )


async def _cleanup(session: AsyncSession) -> None:
    await session.execute(delete(IngestionJob))
    await session.execute(delete(Document))


async def _seed(
    session_factory: async_sessionmaker[AsyncSession],
    identity: int,
    *,
    status: str = "pending",
    attempt_count: int = 0,
    progress: int = 0,
    created_at: datetime = NOW,
    document_status: str = "pending",
) -> tuple[Document, IngestionJob]:
    document = _document(
        identity,
        created_at=created_at,
        status=document_status,
    )
    job = _job(
        identity + 10_000,
        document_id=document.id,
        status=status,
        attempt_count=attempt_count,
        progress=progress,
        created_at=created_at,
    )
    async with session_factory() as session:
        async with session.begin():
            repository = IngestionRepository(session)
            repository.add_document(document)
            repository.add_job(job)
            await session.flush()
    return document, job


async def _claim(
    session_factory: async_sessionmaker[AsyncSession],
    job_id: UUID,
) -> JobSnapshot | None:
    async with session_factory() as session:
        async with session.begin():
            return await IngestionRepository(session).claim_pending(job_id, NOW)


async def _exercise_repository(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as session:
            async with session.begin():
                await _cleanup(session)

        # pending -> running，首次执行把 attempt 从 0 推进到 1。
        _, pending = await _seed(session_factory, 1)
        async with session_factory() as session:
            async with session.begin():
                snapshot = await IngestionRepository(session).claim_pending(
                    pending.id,
                    NOW,
                )
                assert snapshot == JobSnapshot(
                    job_id=pending.id,
                    document_id=pending.document_id,
                    requested_by=None,
                    job_type="pdf",
                    attempt_count=1,
                    request_payload={"category": "代数"},
                )
                with pytest.raises(FrozenInstanceError):
                    snapshot.attempt_count = 9  # type: ignore[misc]

        # running 的进度及两个终态都必须匹配当前 attempt。
        async with session_factory() as session:
            async with session.begin():
                repository = IngestionRepository(session)
                assert await repository.set_progress(pending.id, 1, 45)
                assert not await repository.set_progress(pending.id, 0, 50)
                assert not await repository.complete(pending.id, 0, NOW)
                assert await repository.fail(
                    pending.id,
                    1,
                    "PDF_READ_FAILED" * 8,
                    "安全摘要" * 200,
                    NOW,
                )

        async with session_factory() as observer:
            failed = await observer.get(IngestionJob, pending.id)
            assert failed is not None
            assert (failed.status, failed.progress, failed.attempt_count) == (
                "failed",
                45,
                1,
            )
            assert len(failed.error_code or "") == 64
            assert len(failed.error_message or "") == 500

        # failed -> running 清理旧错误；旧 worker 不能覆盖 attempt=2。
        async with session_factory() as session:
            async with session.begin():
                repository = IngestionRepository(session)
                retry = await repository.claim_retry(pending.id, NOW + timedelta(minutes=1))
                assert retry is not None and retry.attempt_count == 2
                assert not await repository.set_progress(pending.id, 1, 80)
                assert not await repository.fail(
                    pending.id,
                    1,
                    "STALE_WORKER",
                    "旧 worker",
                    NOW,
                )
                assert not await repository.complete(pending.id, 1, NOW)
                assert await repository.set_progress(pending.id, 2, 90)
                assert await repository.complete(
                    pending.id,
                    2,
                    NOW + timedelta(minutes=2),
                )

        async with session_factory() as observer:
            completed = await observer.get(IngestionJob, pending.id)
            assert completed is not None
            assert (
                completed.status,
                completed.progress,
                completed.attempt_count,
                completed.error_code,
                completed.error_message,
            ) == ("completed", 100, 2, None, None)

        # completed 的所有写动作均失败，不返回旧 ORM 假装成功。
        async with session_factory() as session:
            async with session.begin():
                repository = IngestionRepository(session)
                assert await repository.claim_pending(completed.id, NOW) is None
                assert await repository.claim_retry(completed.id, NOW) is None
                assert await repository.cancel_pending(completed.id, NOW) is None
                assert not await repository.set_progress(completed.id, 2, 99)
                assert not await repository.complete(completed.id, 2, NOW)
                assert not await repository.fail(
                    completed.id,
                    2,
                    "LATE",
                    "迟到写回",
                    NOW,
                )

        # pending -> cancelled；cancelled 和 running 的非法动作均失败。
        _, cancellable = await _seed(session_factory, 2)
        async with session_factory() as session:
            async with session.begin():
                repository = IngestionRepository(session)
                cancelled = await repository.cancel_pending(cancellable.id, NOW)
                assert cancelled is not None and cancelled.status == "cancelled"
                assert await repository.claim_pending(cancellable.id, NOW) is None
                assert await repository.claim_retry(cancellable.id, NOW) is None
                assert await repository.cancel_pending(cancellable.id, NOW) is None
                assert not await repository.complete(cancellable.id, 0, NOW)
                assert not await repository.fail(
                    cancellable.id,
                    0,
                    "LATE",
                    "迟到写回",
                    NOW,
                )

        _, running = await _seed(
            session_factory,
            3,
            status="running",
            attempt_count=1,
        )
        async with session_factory() as session:
            async with session.begin():
                repository = IngestionRepository(session)
                assert await repository.claim_pending(running.id, NOW) is None
                assert await repository.claim_retry(running.id, NOW) is None
                assert await repository.cancel_pending(running.id, NOW) is None

        # 两个独立事务并发 claim，只有一个能得到快照。
        _, concurrent = await _seed(session_factory, 4)
        claims = await asyncio.gather(
            _claim(session_factory, concurrent.id),
            _claim(session_factory, concurrent.id),
        )
        assert sum(result is not None for result in claims) == 1

        # list 与 count 使用同一筛选，并按 created_at DESC、id DESC 稳定排序。
        latest = NOW + timedelta(hours=1)
        newest_low, _ = await _seed(
            session_factory,
            10,
            created_at=latest,
            document_status="archived",
        )
        newest_high, _ = await _seed(
            session_factory,
            11,
            created_at=latest,
            document_status="archived",
        )
        older, _ = await _seed(
            session_factory,
            12,
            created_at=NOW,
            document_status="archived",
        )
        async with session_factory() as session:
            repository = IngestionRepository(session)
            first_page, total = await repository.list_documents(
                offset=0,
                limit=2,
                status="archived",
            )
            assert [item.id for item in first_page] == [newest_high.id, newest_low.id]
            assert total == 3
            second_page, second_total = await repository.list_documents(
                offset=2,
                limit=2,
                status="archived",
            )
            assert [item.id for item in second_page] == [older.id]
            assert second_total == total
            assert await repository.get_job(UUID(int=999_999)) is None
            assert await repository.get_job(concurrent.id) is not None

        # Repository 不提交：外部 observer 看不到未提交新增，会话关闭后行消失。
        uncommitted_document = _document(99)
        uncommitted_job = _job(
            10_099,
            document_id=uncommitted_document.id,
        )
        async with session_factory() as session:
            repository = IngestionRepository(session)
            repository.add_document(uncommitted_document)
            repository.add_job(uncommitted_job)
            await session.flush()
            async with session_factory() as observer:
                assert await observer.get(Document, uncommitted_document.id) is None
                assert await observer.get(IngestionJob, uncommitted_job.id) is None
        async with session_factory() as observer:
            assert await observer.scalar(
                select(Document.id).where(Document.id == uncommitted_document.id)
            ) is None
    finally:
        try:
            async with session_factory() as session:
                async with session.begin():
                    await _cleanup(session)
        finally:
            await engine.dispose()


def test_repository_enforces_cas_state_machine_and_transaction_boundary() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    asyncio.run(_exercise_repository(safe_url))
