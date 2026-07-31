"""导入 pipeline 从任务认领到 pgvector 写回的集成测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.errors import DocumentPdfInvalidError
from app.modules.ingestion.repository import IngestionRepository
from app.modules.ingestion.service import IngestionService
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.services.knowledge_extractor import KnowledgeDraft
from tests.integration.database_safety import require_test_database_url


NOW = datetime(2026, 7, 31, 16, 0, tzinfo=UTC)
DOCUMENT_ID = UUID(int=8_101)
JOB_ID = UUID(int=8_102)


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
    source = upload_root / "2026" / "07" / "pipeline.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-pipeline")
    extractor_calls = 0

    def pdf_extractor(path: Path, *, max_pages: int):
        assert path == source
        return SimpleNamespace(text="一次函数与方程", page_count=1, title=None)

    def draft_extractor(text: str, category: str | None):
        nonlocal extractor_calls
        extractor_calls += 1
        return [
            KnowledgeDraft(
                category=category or "代数",
                title="一次函数",
                keywords=("函数", "方程"),
                content="一次函数可以用线性方程表示。",
                example="y=2x+1",
                steps=("确定斜率", "确定截距"),
                difficulty="easy",
            ),
            KnowledgeDraft(
                category=category or "代数",
                title="一元一次方程",
                keywords=("方程",),
                content="通过移项求未知数。",
                example="x+1=2",
                steps=("移项", "求解"),
                difficulty="medium",
            ),
        ]

    class Provider:
        model = "pipeline-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            return [
                [float(index + 1), *([0.0] * 1023)]
                for index, _text in enumerate(texts)
            ]

    try:
        async with sessions() as session:
            async with session.begin():
                await _cleanup(session)
                session.add(
                    Document(
                        id=DOCUMENT_ID,
                        owner_id=None,
                        original_name="pipeline.pdf",
                        storage_path="2026/07/pipeline.pdf",
                        mime_type="application/pdf",
                        size_bytes=1024,
                        sha256="8" * 64,
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
                        request_payload={"category": "代数"},
                        attempt_count=0,
                        created_at=NOW,
                        updated_at=NOW,
                    )
                )

        service = IngestionService(
            sessions,
            SimpleNamespace(),
            embedding_provider=Provider(),  # type: ignore[arg-type]
            draft_extractor=draft_extractor,
            pdf_extractor=pdf_extractor,
            upload_root=upload_root,
            now=lambda: NOW,
        )
        await service.run_pending(JOB_ID)

        async with sessions() as observer:
            document = await observer.get(Document, DOCUMENT_ID)
            job = await observer.get(IngestionJob, JOB_ID)
            items = list(
                (
                    await observer.scalars(
                        select(KnowledgeItem)
                        .where(KnowledgeItem.ingestion_job_id == JOB_ID)
                        .order_by(KnowledgeItem.title)
                    )
                ).all()
            )
            chunks = list(
                (
                    await observer.scalars(
                        select(KnowledgeChunk)
                        .where(KnowledgeChunk.document_id == DOCUMENT_ID)
                        .order_by(KnowledgeChunk.chunk_index)
                    )
                ).all()
            )
            assert document is not None and document.status == "ready"
            assert job is not None
            assert (job.status, job.progress, job.attempt_count) == (
                "completed",
                100,
                1,
            )
            assert extractor_calls == 1
            assert len(items) == len(chunks) == 2
            assert {item.status for item in items} == {"ready"}
            assert [chunk.chunk_index for chunk in chunks] == [0, 1]
            assert {chunk.status for chunk in chunks} == {"ready"}
            assert {chunk.embedding_model for chunk in chunks} == {"pipeline-model"}
    finally:
        try:
            async with sessions() as session:
                async with session.begin():
                    await _cleanup(session)
        finally:
            await engine.dispose()


def test_pipeline_persists_linked_items_and_global_document_chunk_indexes(
    tmp_path: Path,
) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    asyncio.run(_exercise(safe_url, tmp_path))


async def _exercise_pre_embedding_failures(
    database_url: str,
    upload_root: Path,
) -> None:
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    cases = [
        ("pdf", UUID(int=8_111), UUID(int=8_112), "a" * 64),
        ("llm", UUID(int=8_121), UUID(int=8_122), "b" * 64),
        ("database", UUID(int=8_131), UUID(int=8_132), "c" * 64),
    ]

    class Provider:
        model = "unused-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            return [[1.0, *([0.0] * 1023)] for _text in texts]

    try:
        for stage, document_id, job_id, sha256 in cases:
            relative_path = f"{stage}.pdf"
            (upload_root / relative_path).write_bytes(b"%PDF-failure")
            async with sessions() as session:
                async with session.begin():
                    session.add(
                        Document(
                            id=document_id,
                            owner_id=None,
                            original_name=relative_path,
                            storage_path=relative_path,
                            mime_type="application/pdf",
                            size_bytes=1024,
                            sha256=sha256,
                            status="pending",
                            created_at=NOW,
                            updated_at=NOW,
                        )
                    )
                    session.add(
                        IngestionJob(
                            id=job_id,
                            requested_by=None,
                            document_id=document_id,
                            job_type="pdf",
                            status="pending",
                            progress=0,
                            request_payload={},
                            attempt_count=0,
                            created_at=NOW,
                            updated_at=NOW,
                        )
                    )

            class DatabaseFailingRepository(IngestionRepository):
                async def create_pipeline_items(self, snapshot, drafts):
                    if stage == "database":
                        raise SQLAlchemyError("password=database-secret")
                    return await super().create_pipeline_items(snapshot, drafts)

            def pdf_extractor(_path: Path, *, max_pages: int):
                if stage == "pdf":
                    raise DocumentPdfInvalidError()
                return SimpleNamespace(text="教材", page_count=1, title=None)

            def draft_extractor(text: str, category: str | None):
                if stage == "llm":
                    raise RuntimeError("Authorization: Bearer llm-secret")
                return [
                    KnowledgeDraft(
                        category="代数",
                        title="失败回滚测试",
                        keywords=("事务",),
                        content="数据库失败时不得保留半成品。",
                        example="",
                        steps=("回滚",),
                        difficulty="easy",
                    )
                ]

            service = IngestionService(
                sessions,
                SimpleNamespace(),
                repository_factory=DatabaseFailingRepository,
                embedding_provider=Provider(),  # type: ignore[arg-type]
                draft_extractor=draft_extractor,
                pdf_extractor=pdf_extractor,
                upload_root=upload_root,
                now=lambda: NOW,
            )
            await service.run_pending(job_id)

            async with sessions() as observer:
                document = await observer.get(Document, document_id)
                job = await observer.get(IngestionJob, job_id)
                item_count = int(
                    await observer.scalar(
                        select(func.count())
                        .select_from(KnowledgeItem)
                        .where(KnowledgeItem.ingestion_job_id == job_id)
                    )
                    or 0
                )
                chunk_count = int(
                    await observer.scalar(
                        select(func.count())
                        .select_from(KnowledgeChunk)
                        .join(
                            KnowledgeItem,
                            KnowledgeItem.id == KnowledgeChunk.knowledge_item_id,
                        )
                        .where(KnowledgeItem.ingestion_job_id == job_id)
                    )
                    or 0
                )
                assert document is not None and document.status == "failed"
                assert job is not None and job.status == "failed"
                assert job.error_code == {
                    "pdf": "INGESTION_PDF_INVALID",
                    "llm": "INGESTION_LLM_UNAVAILABLE",
                    "database": "INGESTION_DATABASE_UNAVAILABLE",
                }[stage]
                assert "secret" not in (job.error_message or "").lower()
                assert (item_count, chunk_count) == (0, 0)

            async with sessions() as session:
                async with session.begin():
                    await session.execute(
                        delete(IngestionJob).where(IngestionJob.id == job_id)
                    )
                    await session.execute(
                        delete(Document).where(Document.id == document_id)
                    )
    finally:
        await engine.dispose()


def test_pdf_llm_and_database_failures_close_real_rows_without_secrets(
    tmp_path: Path,
) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    asyncio.run(_exercise_pre_embedding_failures(safe_url, tmp_path))
