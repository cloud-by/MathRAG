"""文档上传与列表 API 的权限、调度和公开响应测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import AppError
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.modules.auth.dependencies import get_current_principal
from app.modules.auth.security import issue_csrf_token
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.ingestion.router import get_ingestion_service, router
from app.modules.ingestion.schemas import (
    DocumentAccepted,
    DocumentPage,
    DocumentRead,
    IngestionJobRead,
)


ADMIN_ID = UUID("11111111-1111-4111-8111-111111111111")
USER_ID = UUID("22222222-2222-4222-8222-222222222222")
DOCUMENT_ID = UUID("33333333-3333-4333-8333-333333333333")
JOB_ID = UUID("44444444-4444-4444-8444-444444444444")
ADMIN_SESSION_HASH = b"a" * 32
USER_SESSION_HASH = b"u" * 32
TRUSTED_ORIGIN = settings.ALLOWED_ORIGINS[0]
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _principal(role: str) -> AuthenticatedPrincipal:
    is_admin = role == "admin"
    return AuthenticatedPrincipal(
        user_id=ADMIN_ID if is_admin else USER_ID,
        session_id=uuid4(),
        username=role,
        role="admin" if is_admin else "student",
        must_change_password=False,
        session_token_hash=ADMIN_SESSION_HASH if is_admin else USER_SESSION_HASH,
    )


def _document() -> DocumentRead:
    return DocumentRead(
        id=DOCUMENT_ID,
        owner_id=ADMIN_ID,
        original_name="lesson.pdf",
        mime_type="application/pdf",
        size_bytes=2048,
        sha256="a" * 64,
        status="pending",
        created_at=NOW,
        updated_at=NOW,
    )


def _job(status: str = "pending", attempt_count: int = 0) -> IngestionJobRead:
    return IngestionJobRead(
        id=JOB_ID,
        requested_by=ADMIN_ID,
        document_id=DOCUMENT_ID,
        job_type="pdf",
        status=status,
        progress=0,
        attempt_count=attempt_count,
        error_code=None,
        error_message=None,
        started_at=None,
        finished_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def accept_pdf(self, upload, *, owner_id: UUID, category: str | None):
        body = await upload.read()
        self.calls.append(
            (
                "accept_pdf",
                {
                    "filename": upload.filename,
                    "content_type": upload.content_type,
                    "body": body,
                    "owner_id": owner_id,
                    "category": category,
                },
            )
        )
        return DocumentAccepted(document=_document(), job=_job())

    async def run_pending(self, job_id: UUID) -> None:
        self.calls.append(("run_pending", job_id))

    async def list_documents(
        self,
        *,
        status: str | None,
        page: int,
        page_size: int,
    ) -> DocumentPage:
        self.calls.append(
            (
                "list_documents",
                {"status": status, "page": page, "page_size": page_size},
            )
        )
        return DocumentPage(
            items=[_document()],
            page=page,
            page_size=page_size,
            total=1,
        )


def _build_client() -> tuple[TestClient, FakeIngestionService]:
    service = FakeIngestionService()
    application = FastAPI()
    install_exception_handlers(application)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(router)
    application.dependency_overrides[get_ingestion_service] = lambda: service

    async def principal_from_header(request: Request) -> AuthenticatedPrincipal:
        role = request.headers.get("X-Test-Role")
        if role not in {"admin", "student"}:
            raise AppError(
                code="AUTH_SESSION_INVALID",
                message="登录状态无效或已过期。",
                status_code=401,
            )
        return _principal(role)

    application.dependency_overrides[get_current_principal] = principal_from_header
    return TestClient(application), service


def _safe_headers(client: TestClient, role: str) -> dict[str, str]:
    session_hash = ADMIN_SESSION_HASH if role == "admin" else USER_SESSION_HASH
    token = issue_csrf_token(session_hash, settings.SESSION_SECRET)
    client.cookies.set(settings.csrf_cookie_name, token)
    return {
        "X-Test-Role": role,
        "X-CSRF-Token": token,
        "Origin": TRUSTED_ORIGIN,
    }


def test_admin_upload_returns_202_and_schedules_after_acceptance() -> None:
    client, service = _build_client()

    response = client.post(
        "/api/v1/documents",
        files={"file": ("lesson.pdf", b"%PDF-1.4 test", "application/pdf")},
        data={"category": "代数"},
        headers=_safe_headers(client, "admin"),
    )

    assert response.status_code == 202
    assert response.json()["job"]["status"] == "pending"
    assert "storage_path" not in response.text
    assert "request_payload" not in response.text
    assert [name for name, _value in service.calls] == [
        "accept_pdf",
        "run_pending",
    ]
    arguments = service.calls[0][1]
    assert arguments == {
        "filename": "lesson.pdf",
        "content_type": "application/pdf",
        "body": b"%PDF-1.4 test",
        "owner_id": ADMIN_ID,
        "category": "代数",
    }
    assert service.calls[1] == ("run_pending", JOB_ID)


def test_document_routes_require_admin_and_upload_requires_csrf() -> None:
    client, service = _build_client()
    upload = {"file": ("lesson.pdf", b"%PDF-1.4", "application/pdf")}

    anonymous = client.get("/api/v1/documents")
    ordinary_read = client.get(
        "/api/v1/documents",
        headers={"X-Test-Role": "student"},
    )
    ordinary_upload = client.post(
        "/api/v1/documents",
        files=upload,
        headers=_safe_headers(client, "student"),
    )
    missing_csrf = client.post(
        "/api/v1/documents",
        files=upload,
        headers={"X-Test-Role": "admin", "Origin": TRUSTED_ORIGIN},
    )

    assert anonymous.status_code == 401
    assert ordinary_read.status_code == 403
    assert ordinary_upload.status_code == 403
    assert missing_csrf.status_code == 403
    assert service.calls == []


def test_admin_lists_documents_with_exact_filters() -> None:
    client, service = _build_client()

    response = client.get(
        "/api/v1/documents?status=failed&page=2&page_size=5",
        headers={"X-Test-Role": "admin"},
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert service.calls == [
        (
            "list_documents",
            {"status": "failed", "page": 2, "page_size": 5},
        )
    ]


def test_openapi_exposes_safe_document_contracts() -> None:
    client, _service = _build_client()

    schema = client.get("/openapi.json").json()
    operations = schema["paths"]["/api/v1/documents"]

    assert set(operations) == {"get", "post"}
    assert operations["post"]["responses"]["202"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/DocumentAccepted")
    assert operations["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/DocumentPage")


def test_main_app_registers_ingestion_router_once() -> None:
    from app.main import create_app

    application = create_app()
    application.dependency_overrides[get_ingestion_service] = (
        lambda: FakeIngestionService()
    )
    schema = TestClient(application).get("/openapi.json").json()

    assert "/api/v1/documents" in schema["paths"]
    assert "/api/v1/ingestion-jobs/{job_id}" in schema["paths"]
    assert "/api/v1/ingestion-jobs/{job_id}/cancel" in schema["paths"]
    assert "/api/v1/ingestion-jobs/{job_id}/retry" in schema["paths"]
    assert sum(
        getattr(included, "original_router", None) is router
        for included in application.routes
    ) == 1
