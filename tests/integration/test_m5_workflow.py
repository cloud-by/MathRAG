"""M5 登录、上传、摄取、隔离、revision 与归档检索工作流。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.modules.auth.dependencies import get_auth_service
from app.modules.auth.models import UserSession
from app.modules.auth.router import router as auth_router
from app.modules.auth.security import hash_password
from app.modules.auth.service import AuthService
from app.modules.ingestion.models import Document, IngestionJob
from app.modules.ingestion.router import get_ingestion_service, router as ingestion_router
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import UploadStorage
from app.modules.knowledge.management_service import KnowledgeManagementService
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.router import (
    get_knowledge_management_service,
    get_knowledge_read_service,
    router as knowledge_router,
)
from app.modules.users.models import User
from app.services.knowledge_extractor import KnowledgeDraft
from tests.integration.database_safety import require_test_database_url


class WorkflowProvider:
    model = "m5-workflow-model"
    dimensions = 1024

    async def embed_texts(self, texts):
        return [[1.0, *([0.0] * 1023)] for _text in texts]


class FlakyWorkflowProvider:
    model = "m5-retry-model"
    dimensions = 1024

    def __init__(self) -> None:
        self.calls = 0

    async def embed_texts(self, texts):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("secret-provider-response")
        return [[1.0, *([0.0] * 1023)] for _text in texts]


class CountingDraftExtractor:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, category: str | None) -> list[KnowledgeDraft]:
        self.calls += 1
        return [
            KnowledgeDraft(
                category=category or "几何",
                title="三角形内角和",
                keywords=("三角形", "内角和"),
                content="三角形内角和为 180 度。",
                example="等边三角形每个内角为 60 度。",
                steps=("作辅助线", "利用平行线性质"),
                difficulty="easy",
            )
        ]


def _build_app(
    auth_service: AuthService,
    ingestion_service: IngestionService,
    knowledge_service: KnowledgeManagementService,
) -> FastAPI:
    application = FastAPI()
    install_exception_handlers(application)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(auth_router)
    application.include_router(knowledge_router)
    application.include_router(ingestion_router)
    application.dependency_overrides[get_auth_service] = lambda: auth_service
    application.dependency_overrides[get_ingestion_service] = (
        lambda: ingestion_service
    )
    application.dependency_overrides[get_knowledge_read_service] = (
        lambda: knowledge_service
    )
    application.dependency_overrides[get_knowledge_management_service] = (
        lambda: knowledge_service
    )
    return application


def _csrf_headers(client: TestClient) -> dict[str, str]:
    token = client.cookies.get(settings.csrf_cookie_name)
    assert token
    return {
        "Origin": settings.ALLOWED_ORIGINS[0],
        "X-CSRF-Token": token,
    }


async def _seed_users(session_factory, admin_id: UUID, user_id: UUID) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add_all(
                [
                    User(
                        id=admin_id,
                        username=f"m5-admin-{admin_id.hex[:8]}",
                        email=None,
                        password_hash=await hash_password("admin-password"),
                        role="admin",
                        status="active",
                    ),
                    User(
                        id=user_id,
                        username=f"m5-user-{user_id.hex[:8]}",
                        email=None,
                        password_hash=await hash_password("user-password"),
                        role="student",
                        status="active",
                    ),
                ]
            )


async def _cleanup(session_factory, user_ids: tuple[UUID, UUID]) -> None:
    async with session_factory() as session:
        async with session.begin():
            job_ids = select(IngestionJob.id).where(
                IngestionJob.requested_by.in_(user_ids)
            )
            item_ids = select(KnowledgeItem.id).where(
                (KnowledgeItem.owner_id.in_(user_ids))
                | (KnowledgeItem.ingestion_job_id.in_(job_ids))
            )
            await session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.knowledge_item_id.in_(item_ids)
                )
            )
            await session.execute(delete(KnowledgeItem).where(KnowledgeItem.id.in_(item_ids)))
            await session.execute(delete(IngestionJob).where(IngestionJob.id.in_(job_ids)))
            await session.execute(delete(Document).where(Document.owner_id.in_(user_ids)))
            await session.execute(delete(UserSession).where(UserSession.user_id.in_(user_ids)))
            await session.execute(delete(User).where(User.id.in_(user_ids)))


async def _search_archived(session_factory) -> list[object]:
    async with session_factory() as session:
        return await KnowledgeRepository(session).search_ready_chunks(
            query_vector=[1.0, *([0.0] * 1023)],
            embedding_model="m5-workflow-model",
            limit=3,
        )


async def _ingestion_resource_snapshot(
    session_factory,
    job_id: UUID,
    document_id: UUID,
) -> dict[str, object]:
    async with session_factory() as session:
        job = await session.get(IngestionJob, job_id)
        document = await session.get(Document, document_id)
        items = list(
            (
                await session.scalars(
                    select(KnowledgeItem).where(
                        KnowledgeItem.ingestion_job_id == job_id
                    )
                )
            ).all()
        )
        chunks = list(
            (
                await session.scalars(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.document_id == document_id
                    )
                )
            ).all()
        )
        assert job is not None
        assert document is not None
        return {
            "job_status": job.status,
            "attempt_count": job.attempt_count,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "document_status": document.status,
            "item_ids": {item.id for item in items},
            "item_statuses": {item.status for item in items},
            "chunk_ids": {chunk.id for chunk in chunks},
            "chunk_statuses": {chunk.status for chunk in chunks},
        }


def test_m5_admin_ingestion_user_isolation_revision_and_archive(
    tmp_path: Path,
) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    engine = create_async_engine(safe_url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    admin_id = uuid4()
    user_id = uuid4()
    asyncio.run(_seed_users(sessions, admin_id, user_id))

    storage = UploadStorage(
        root=tmp_path / "uploads",
        max_bytes=1024 * 1024,
        max_pages=10,
    )

    async def accept_test_pdf(_path: Path) -> None:
        return None

    storage._validate_pdf = accept_test_pdf  # type: ignore[method-assign]
    provider = WorkflowProvider()
    ingestion_service = IngestionService(
        sessions,
        storage,
        embedding_provider=provider,  # type: ignore[arg-type]
        draft_extractor=lambda text, category: [
            KnowledgeDraft(
                category=category or "代数",
                title="一元一次方程",
                keywords=("方程", "移项"),
                content="通过移项和合并同类项求解未知数。",
                example="x+1=2",
                steps=("移项", "合并同类项"),
                difficulty="easy",
            )
        ],
        pdf_extractor=lambda _path, *, max_pages: SimpleNamespace(
            text="一元一次方程教材正文",
            page_count=1,
            title="方程教材",
        ),
        upload_root=tmp_path / "uploads",
    )
    knowledge_service = KnowledgeManagementService(
        sessions,
        provider,  # type: ignore[arg-type]
    )
    auth_service = AuthService(
        sessions,
        session_ttl_seconds=settings.SESSION_TTL_SECONDS,
        csrf_secret=settings.SESSION_SECRET,
    )
    application = _build_app(auth_service, ingestion_service, knowledge_service)
    admin = TestClient(application)
    ordinary = TestClient(application)
    item_id: str | None = None
    try:
        admin_login = admin.post(
            "/api/v1/auth/login",
            json={
                "username": f"m5-admin-{admin_id.hex[:8]}",
                "password": "admin-password",
            },
            headers={"Origin": settings.ALLOWED_ORIGINS[0]},
        )
        assert admin_login.status_code == 200
        accepted = admin.post(
            "/api/v1/documents",
            files={
                "file": (
                    "lesson.pdf",
                    b"%PDF-1.4 workflow",
                    "application/pdf",
                )
            },
            data={"category": "代数"},
            headers=_csrf_headers(admin),
        )
        assert accepted.status_code == 202
        assert "storage_path" not in accepted.text
        job_id = accepted.json()["job"]["id"]
        document_id = accepted.json()["document"]["id"]

        job = admin.get(f"/api/v1/ingestion-jobs/{job_id}")
        documents = admin.get("/api/v1/documents?status=ready")
        knowledge = admin.get(
            "/api/v1/knowledge-items?status=ready&visibility=public"
        )
        assert job.status_code == 200
        assert (job.json()["status"], job.json()["attempt_count"]) == (
            "completed",
            1,
        )
        assert documents.status_code == 200
        assert documents.json()["items"][0]["id"] == document_id
        assert knowledge.status_code == 200
        item = knowledge.json()["items"][0]
        item_id = item["id"]
        assert item["revision"] == 1

        user_login = ordinary.post(
            "/api/v1/auth/login",
            json={
                "username": f"m5-user-{user_id.hex[:8]}",
                "password": "user-password",
            },
            headers={"Origin": settings.ALLOWED_ORIGINS[0]},
        )
        assert user_login.status_code == 200
        assert ordinary.get(f"/api/v1/knowledge-items/{item_id}").status_code == 200

        private = admin.patch(
            f"/api/v1/knowledge-items/{item_id}",
            json={"revision": 1, "visibility": "private"},
            headers=_csrf_headers(admin),
        )
        stale = admin.patch(
            f"/api/v1/knowledge-items/{item_id}",
            json={"revision": 1, "title": "过期写入"},
            headers=_csrf_headers(admin),
        )
        forbidden = ordinary.patch(
            f"/api/v1/knowledge-items/{item_id}",
            json={"revision": 2, "title": "越权写入"},
            headers=_csrf_headers(ordinary),
        )
        assert private.status_code == 200
        assert private.json()["revision"] == 2
        assert stale.status_code == 409
        assert stale.json()["error"]["code"] == "KNOWLEDGE_REVISION_CONFLICT"
        assert ordinary.get(f"/api/v1/knowledge-items/{item_id}").status_code == 404
        assert forbidden.status_code == 403

        archived = admin.delete(
            f"/api/v1/knowledge-items/{item_id}?revision=2",
            headers=_csrf_headers(admin),
        )
        assert archived.status_code == 204
        assert asyncio.run(_search_archived(sessions)) == []
    finally:
        asyncio.run(_cleanup(sessions, (admin_id, user_id)))
        asyncio.run(engine.dispose())


def test_m5_failed_ingestion_retry_reuses_all_resources(tmp_path: Path) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    safe_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    engine = create_async_engine(safe_url, poolclass=NullPool)
    sessions = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    admin_id = uuid4()
    user_id = uuid4()
    asyncio.run(_seed_users(sessions, admin_id, user_id))

    storage = UploadStorage(
        root=tmp_path / "uploads",
        max_bytes=1024 * 1024,
        max_pages=10,
    )

    async def accept_test_pdf(_path: Path) -> None:
        return None

    storage._validate_pdf = accept_test_pdf  # type: ignore[method-assign]
    provider = FlakyWorkflowProvider()
    extractor = CountingDraftExtractor()
    ingestion_service = IngestionService(
        sessions,
        storage,
        embedding_provider=provider,  # type: ignore[arg-type]
        draft_extractor=extractor,
        pdf_extractor=lambda _path, *, max_pages: SimpleNamespace(
            text="三角形教材正文",
            page_count=1,
            title="几何教材",
        ),
        upload_root=tmp_path / "uploads",
    )
    knowledge_service = KnowledgeManagementService(
        sessions,
        provider,  # type: ignore[arg-type]
    )
    auth_service = AuthService(
        sessions,
        session_ttl_seconds=settings.SESSION_TTL_SECONDS,
        csrf_secret=settings.SESSION_SECRET,
    )
    admin = TestClient(
        _build_app(auth_service, ingestion_service, knowledge_service)
    )
    try:
        login = admin.post(
            "/api/v1/auth/login",
            json={
                "username": f"m5-admin-{admin_id.hex[:8]}",
                "password": "admin-password",
            },
            headers={"Origin": settings.ALLOWED_ORIGINS[0]},
        )
        assert login.status_code == 200
        accepted = admin.post(
            "/api/v1/documents",
            files={
                "file": (
                    "retry.pdf",
                    b"%PDF-1.4 retry workflow",
                    "application/pdf",
                )
            },
            data={"category": "几何"},
            headers=_csrf_headers(admin),
        )
        assert accepted.status_code == 202
        job_id = UUID(accepted.json()["job"]["id"])
        document_id = UUID(accepted.json()["document"]["id"])

        failed_job = admin.get(f"/api/v1/ingestion-jobs/{job_id}")
        assert failed_job.status_code == 200
        assert failed_job.json()["status"] == "failed"
        assert failed_job.json()["error_code"] == "INGESTION_UPSTREAM_TIMEOUT"
        assert failed_job.json()["error_message"] == "上游服务响应超时。"
        assert "secret-provider-response" not in failed_job.text
        failed = asyncio.run(
            _ingestion_resource_snapshot(sessions, job_id, document_id)
        )
        assert failed["document_status"] == "failed"
        assert failed["error_message"] == "上游服务响应超时。"
        assert failed["item_statuses"] == {"failed"}
        assert failed["chunk_statuses"] == {"failed"}

        retried = admin.post(
            f"/api/v1/ingestion-jobs/{job_id}/retry",
            headers=_csrf_headers(admin),
        )
        assert retried.status_code == 202
        completed_job = admin.get(f"/api/v1/ingestion-jobs/{job_id}")
        assert completed_job.status_code == 200
        assert (
            completed_job.json()["status"],
            completed_job.json()["attempt_count"],
        ) == ("completed", 2)

        completed = asyncio.run(
            _ingestion_resource_snapshot(sessions, job_id, document_id)
        )
        assert completed["document_status"] == "ready"
        assert completed["item_statuses"] == {"ready"}
        assert completed["chunk_statuses"] == {"ready"}
        assert completed["item_ids"] == failed["item_ids"]
        assert completed["chunk_ids"] == failed["chunk_ids"]
        assert extractor.calls == 1
        assert provider.calls == 2
    finally:
        asyncio.run(_cleanup(sessions, (admin_id, user_id)))
        asyncio.run(engine.dispose())
