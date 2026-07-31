"""导入失败重试不重复创建知识数据的集成测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.service import IngestionService
from app.modules.knowledge.errors import EmbeddingUnavailableError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.services.knowledge_extractor import KnowledgeDraft
from tests.integration.database_safety import require_test_database_url


NOW = datetime(2026, 7, 31, 17, 0, tzinfo=UTC)
DOCUMENT_ID = UUID(int=8_201)
JOB_ID = UUID(int=8_202)


async def _cleanup(session) -> None:
    item_ids = select(KnowledgeItem.id).where(
        KnowledgeItem.ingestion_job_id == JOB_ID
    )
    await session.execute(
        delete(KnowledgeChunk).where(KnowledgeChunk.knowledge_item_id.in_(item_ids))
    )
    await session.execute(
        delete(KnowledgeItem).where(KnowledgeItem.ingestion_job_id == JOB_ID)
    )
    await session.execute(delete(IngestionJob).where(IngestionJob.id == JOB_ID))
    await session.execute(delete(Document).where(Document.id == DOCUMENT_ID))


async def _exercise(database_url: str, upload_root: Path) -> None:
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    source = upload_root / "retry.pdf"
    source.write_bytes(b"%PDF-retry")

    class Extractor:
        calls = 0

        def __call__(self, text: str, category: str | None):
            self.calls += 1
            return [
                KnowledgeDraft(
                    category="几何",
                    title="三角形内角和",
                    keywords=("三角形",),
                    content="三角形内角和为 180 度。",
                    example="",
                    steps=("延长一边", "使用平行线性质"),
                    difficulty="easy",
                )
            ]

    class Provider:
        model = "retry-model"
        dimensions = 1024
        calls = 0

        async def embed_texts(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise EmbeddingUnavailableError("secret-provider-token")
            return [[1.0, *([0.0] * 1023)] for _text in texts]

    extractor = Extractor()
    provider = Provider()
    try:
        async with sessions() as session:
            async with session.begin():
                await _cleanup(session)
                session.add(
                    Document(
                        id=DOCUMENT_ID,
                        owner_id=None,
                        original_name="retry.pdf",
                        storage_path="retry.pdf",
                        mime_type="application/pdf",
                        size_bytes=1024,
                        sha256="9" * 64,
                        status="pending",
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )
                session.add(
                    IngestionJob(
                        id=JOB_ID,
                        requested_by=None,
                        document_id=DOCUMENT_ID,
                        job_type="pdf",
                        status="pending",
                        progress=0,
                        request_payload={},
                        attempt_count=0,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )

        service = IngestionService(
            sessions,
            SimpleNamespace(),
            embedding_provider=provider,  # type: ignore[arg-type]
            draft_extractor=extractor,
            pdf_extractor=lambda _path, *, max_pages: SimpleNamespace(
                text="三角形教材", page_count=1, title=None
            ),
            upload_root=upload_root,
            now=lambda: NOW,
        )
        await service.run_pending(JOB_ID)

        async with sessions() as observer:
            first_document = await observer.get(Document, DOCUMENT_ID)
            first_job = await observer.get(IngestionJob, JOB_ID)
            first_items = list(
                (
                    await observer.scalars(
                        select(KnowledgeItem).where(
                            KnowledgeItem.ingestion_job_id == JOB_ID
                        )
                    )
                ).all()
            )
            first_chunks = list(
                (
                    await observer.scalars(
                        select(KnowledgeChunk).where(
                            KnowledgeChunk.document_id == DOCUMENT_ID
                        )
                    )
                ).all()
            )
            assert first_job is not None
            assert first_document is not None and first_document.status == "failed"
            assert (
                first_job.status,
                first_job.error_code,
                first_job.error_message,
            ) == (
                "failed",
                "INGESTION_EMBEDDING_UNAVAILABLE",
                "知识向量化服务暂时不可用。",
            )
            first_ids = (
                {item.id for item in first_items},
                {chunk.id for chunk in first_chunks},
            )
            counts_after_failure = (len(first_items), len(first_chunks))
            assert {item.status for item in first_items} == {"failed"}
            assert {chunk.status for chunk in first_chunks} == {"failed"}

        results = await asyncio.gather(
            service.claim_retry(JOB_ID),
            service.claim_retry(JOB_ID),
            return_exceptions=True,
        )
        snapshots = [result for result in results if not isinstance(result, BaseException)]
        assert len(snapshots) == 1
        await service.resume_retry(snapshots[0])

        async with sessions() as observer:
            ready_document = await observer.get(Document, DOCUMENT_ID)
            completed_job = await observer.get(IngestionJob, JOB_ID)
            retry_items = list(
                (
                    await observer.scalars(
                        select(KnowledgeItem).where(
                            KnowledgeItem.ingestion_job_id == JOB_ID
                        )
                    )
                ).all()
            )
            retry_chunks = list(
                (
                    await observer.scalars(
                        select(KnowledgeChunk).where(
                            KnowledgeChunk.document_id == DOCUMENT_ID
                        )
                    )
                ).all()
            )
            counts_after_retry = (len(retry_items), len(retry_chunks))
            assert counts_after_retry == counts_after_failure
            assert (
                {item.id for item in retry_items},
                {chunk.id for chunk in retry_chunks},
            ) == first_ids
            assert extractor.calls == 1
            assert provider.calls == 2
            assert completed_job is not None
            assert ready_document is not None and ready_document.status == "ready"
            assert (completed_job.status, completed_job.attempt_count) == (
                "completed",
                2,
            )
            assert {item.status for item in retry_items} == {"ready"}
            assert {chunk.status for chunk in retry_chunks} == {"ready"}
    finally:
        try:
            async with sessions() as session:
                async with session.begin():
                    await _cleanup(session)
        finally:
            await engine.dispose()


def test_retry_reuses_existing_items_and_concurrent_claim_has_one_winner(
    tmp_path: Path,
) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    asyncio.run(_exercise(safe_url, tmp_path))
