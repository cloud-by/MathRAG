"""统一 ingestion CLI 的参数传递与依赖边界测试。"""

from __future__ import annotations

import argparse
import ast
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.ingestion import factory as ingestion_factory
from app.modules.ingestion.models import IngestionJob
from app.modules.ingestion.service import IngestionService
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.users.models import User
from app.services.knowledge_extractor import KnowledgeDraft
from scripts import import_math_knowledge, import_pdf_knowledge
from tests.integration.database_safety import require_test_database_url


ADMIN_ID = UUID("11111111-1111-4111-8111-111111111111")


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.next_job = 0

    async def accept_local_pdf(
        self,
        path: Path,
        *,
        owner_id: UUID,
        category: str | None,
    ):
        self.next_job += 1
        job_id = UUID(int=100 + self.next_job)
        self.calls.append(
            (
                "accept_local_pdf",
                {"path": path, "owner_id": owner_id, "category": category},
            )
        )
        return SimpleNamespace(job=SimpleNamespace(id=job_id))

    async def accept_web(
        self,
        *,
        requested_by: UUID,
        sources: list[str],
        keywords: list[str],
        limit_per_source: int,
        category: str | None,
        delay_seconds: float,
        max_chunk_chars: int,
    ):
        job_id = UUID(int=201)
        self.calls.append(
            (
                "accept_web",
                {
                    "requested_by": requested_by,
                    "sources": sources,
                    "keywords": keywords,
                    "limit_per_source": limit_per_source,
                    "category": category,
                    "delay_seconds": delay_seconds,
                    "max_chunk_chars": max_chunk_chars,
                },
            )
        )
        return SimpleNamespace(id=job_id)

    async def run_pending(self, job_id: UUID) -> None:
        self.calls.append(("run_pending", job_id))

    async def get_job(self, job_id: UUID):
        self.calls.append(("get_job", job_id))
        return SimpleNamespace(id=job_id, status="completed")


async def _resolve_admin(username: str) -> UUID:
    assert username == "admin"
    return ADMIN_ID


def test_pdf_cli_routes_files_through_service_without_jsonl(tmp_path, capsys) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "a.pdf"
    second = nested / "b.pdf"
    first.write_bytes(b"%PDF-a")
    second.write_bytes(b"%PDF-b")
    service = FakeService()
    args = argparse.Namespace(
        data_dir=tmp_path,
        no_recursive=False,
        max_chunks=1,
        category="代数",
        requested_by="admin",
    )

    completed = asyncio.run(
        import_pdf_knowledge.run_import(
            args,
            service=service,
            resolve_admin=_resolve_admin,
        )
    )

    assert completed == 1
    assert service.calls[0] == (
        "accept_local_pdf",
        {"path": first, "owner_id": ADMIN_ID, "category": "代数"},
    )
    assert service.calls[1] == ("run_pending", UUID(int=101))
    assert service.calls[2] == ("get_job", UUID(int=101))
    assert "Completed jobs: 1" in capsys.readouterr().out
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_web_cli_passes_frozen_payload_and_runs_job(capsys) -> None:
    service = FakeService()
    args = argparse.Namespace(
        sources=["wikipedia"],
        keywords=["derivative", "integral"],
        limit_per_source=2,
        category="微积分",
        max_chunk_chars=5000,
        delay_seconds=0.25,
        requested_by="admin",
    )

    completed = asyncio.run(
        import_math_knowledge.run_import(
            args,
            service=service,
            resolve_admin=_resolve_admin,
        )
    )

    assert completed == 1
    assert service.calls == [
        (
            "accept_web",
            {
                "requested_by": ADMIN_ID,
                "sources": ["wikipedia"],
                "keywords": ["derivative", "integral"],
                "limit_per_source": 2,
                "category": "微积分",
                "delay_seconds": 0.25,
                "max_chunk_chars": 5000,
            },
        ),
        ("run_pending", UUID(int=201)),
        ("get_job", UUID(int=201)),
    ]
    assert "Completed jobs: 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "module",
    [import_math_knowledge, import_pdf_knowledge],
)
def test_cli_source_has_no_second_database_or_jsonl_write_path(module) -> None:
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8")
    lowered = source.lower()
    tree = ast.parse(source, filename=str(path))
    imported = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any("sqlalchemy" in name for name in imported)
    assert not any("models" in name or "repository" in name for name in imported)
    assert "session.execute" not in lowered
    assert "jsonl" not in lowered
    assert "append_records" not in lowered
    assert "--output" not in source
    assert "--text-output" not in source
    assert "--append-text-output" not in source
    assert "build_ingestion_service" in source


@pytest.mark.parametrize(
    "builder, required_args",
    [
        (
            import_pdf_knowledge.build_parser,
            ["--data-dir", "."],
        ),
        (
            import_math_knowledge.build_parser,
            ["--keywords", "derivative"],
        ),
    ],
)
def test_cli_requires_requested_by(builder, required_args) -> None:
    with pytest.raises(SystemExit):
        builder().parse_args(required_args)


async def _exercise_web_pipeline(database_url: str, monkeypatch) -> None:
    engine = create_async_engine(database_url)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    admin_id = UUID(int=9_301)

    async def cleanup() -> None:
        async with sessions() as session:
            async with session.begin():
                job_ids = select(IngestionJob.id).where(
                    IngestionJob.requested_by == admin_id
                )
                item_ids = select(KnowledgeItem.id).where(
                    KnowledgeItem.ingestion_job_id.in_(job_ids)
                )
                await session.execute(
                    delete(KnowledgeChunk).where(
                        KnowledgeChunk.knowledge_item_id.in_(item_ids)
                    )
                )
                await session.execute(
                    delete(KnowledgeItem).where(
                        KnowledgeItem.ingestion_job_id.in_(job_ids)
                    )
                )
                await session.execute(
                    delete(IngestionJob).where(IngestionJob.requested_by == admin_id)
                )
                await session.execute(delete(User).where(User.id == admin_id))

    class Provider:
        model = "web-cli-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            return [[1.0, *([0.0] * 1023)] for _text in texts]

    try:
        await cleanup()
        async with sessions() as session:
            async with session.begin():
                session.add(
                    User(
                        id=admin_id,
                        username="cli-admin",
                        email=None,
                        password_hash="test-only-hash",
                        role="admin",
                        status="active",
                    )
                )
        monkeypatch.setattr(
            ingestion_factory,
            "get_session_factory",
            lambda: sessions,
        )
        assert await ingestion_factory.resolve_active_admin(" CLI-ADMIN ") == admin_id

        service = IngestionService(
            sessions,
            SimpleNamespace(),
            embedding_provider=Provider(),  # type: ignore[arg-type]
            web_source_loader=lambda payload: "函数描述变量之间的对应关系。",
            draft_extractor=lambda text, category: [
                KnowledgeDraft(
                    category=category or "函数",
                    title="函数概念",
                    keywords=("函数", "对应关系"),
                    content=text,
                    example="y=2x+1",
                    steps=("确定定义域", "确定对应关系"),
                    difficulty="easy",
                )
            ],
        )
        accepted = await service.accept_web(
            requested_by=admin_id,
            sources=["wikipedia"],
            keywords=["function"],
            limit_per_source=1,
            category="函数",
            delay_seconds=0,
            max_chunk_chars=200,
        )
        await service.run_pending(accepted.id)

        async with sessions() as observer:
            job = await observer.get(IngestionJob, accepted.id)
            item = await observer.scalar(
                select(KnowledgeItem).where(
                    KnowledgeItem.ingestion_job_id == accepted.id
                )
            )
            assert item is not None
            chunk = await observer.scalar(
                select(KnowledgeChunk).where(
                    KnowledgeChunk.knowledge_item_id == item.id
                )
            )
            assert job is not None
            assert (job.status, job.attempt_count, job.document_id) == (
                "completed",
                1,
                None,
            )
            assert set(job.request_payload) == {
                "sources",
                "keywords",
                "limit_per_source",
                "category",
                "delay_seconds",
                "max_chunk_chars",
            }
            assert item.status == "ready"
            assert chunk is not None
            assert (chunk.status, chunk.document_id) == ("ready", None)
    finally:
        try:
            await cleanup()
        finally:
            await engine.dispose()


def test_web_cli_service_uses_active_admin_and_real_pipeline(monkeypatch) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    asyncio.run(_exercise_web_pipeline(safe_url, monkeypatch))
