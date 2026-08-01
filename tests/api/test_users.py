"""用户管理 HTTP API 契约测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import AppError
from app.core.exception_handlers import install_exception_handlers
from app.modules.auth.dependencies import get_current_principal
from app.modules.auth.security import issue_csrf_token
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.users.dependencies import get_user_service
from app.modules.users.router import router
from app.modules.users.schemas import ManagedUserRead, UserPage


ADMIN_ID = UUID("00000000-0000-0000-0000-000000000801")
TEACHER_ID = UUID("00000000-0000-0000-0000-000000000802")
STUDENT_ID = UUID("00000000-0000-0000-0000-000000000803")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000804")
SESSION_HASH = b"u" * 32
NOW = datetime(2026, 8, 1, tzinfo=UTC)


def managed_user(
    *,
    user_id: UUID = STUDENT_ID,
    username: str = "student-a",
    role: str = "student",
    created_by_user_id: UUID | None = TEACHER_ID,
) -> ManagedUserRead:
    return ManagedUserRead(
        id=user_id,
        username=username,
        email=f"{username}@example.local",
        role=role,
        status="active",
        created_by_user_id=created_by_user_id,
        created_by_username="teacher-a" if created_by_user_id else None,
        must_change_password=True,
        created_at=NOW,
        updated_at=NOW,
    )


class FakeUserService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def list_managed_users(self, actor, **filters):
        self.calls.append(("list", {"actor": actor, **filters}))
        return UserPage(items=[managed_user()], page=filters["page"], page_size=filters["page_size"], total=1)

    async def create_managed_user(self, actor, request):
        self.calls.append(("create", {"actor": actor, "request": request}))
        return managed_user(
            username=request.username,
            role=request.role,
            created_by_user_id=actor.user_id,
        )

    async def get_managed_user(self, actor, user_id):
        self.calls.append(("get", {"actor": actor, "user_id": user_id}))
        if user_id == OTHER_ID:
            raise AppError(
                code="USER_NOT_FOUND",
                message="用户不存在。",
                status_code=404,
            )
        return managed_user(user_id=user_id)

    async def update_managed_user(self, actor, user_id, request, now):
        self.calls.append(
            (
                "update",
                {"actor": actor, "user_id": user_id, "request": request, "now": now},
            )
        )
        return managed_user(user_id=user_id, username=request.username or "student-a")

    async def reset_managed_password(self, actor, user_id, password, now):
        self.calls.append(
            (
                "reset",
                {"actor": actor, "user_id": user_id, "password": password, "now": now},
            )
        )


def build_client() -> tuple[TestClient, FakeUserService]:
    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    service = FakeUserService()
    app.dependency_overrides[get_user_service] = lambda: service

    async def principal_from_header(request: Request) -> AuthenticatedPrincipal:
        role = request.headers.get("X-Test-Role")
        if role not in {"student", "teacher", "admin"}:
            raise AppError(
                code="AUTH_SESSION_INVALID",
                message="登录状态无效或已过期。",
                status_code=401,
            )
        user_id = {
            "student": STUDENT_ID,
            "teacher": TEACHER_ID,
            "admin": ADMIN_ID,
        }[role]
        return AuthenticatedPrincipal(
            user_id=user_id,
            session_id=UUID(int=99),
            username=f"{role}-actor",
            role=role,  # type: ignore[arg-type]
            must_change_password=False,
            session_token_hash=SESSION_HASH,
        )

    app.dependency_overrides[get_current_principal] = principal_from_header
    return TestClient(app), service


def safe_headers(client: TestClient, role: str) -> dict[str, str]:
    token = issue_csrf_token(SESSION_HASH, settings.SESSION_SECRET)
    client.cookies.set(settings.csrf_cookie_name, token)
    return {
        "X-Test-Role": role,
        "Origin": settings.ALLOWED_ORIGINS[0],
        "X-CSRF-Token": token,
    }


def test_admin_and_teacher_routes_pass_actor_and_payload() -> None:
    client, service = build_client()

    created = client.post(
        "/api/v1/users",
        json={
            "username": "student-a",
            "email": "student-a@example.local",
            "password": "temporary-123",
            "role": "student",
        },
        headers=safe_headers(client, "teacher"),
    )
    listing = client.get(
        "/api/v1/users?q=student&role=student&status=active&page=2&page_size=10",
        headers={"X-Test-Role": "admin"},
    )

    assert created.status_code == 201
    assert created.json()["created_by_user_id"] == str(TEACHER_ID)
    assert listing.status_code == 200
    assert service.calls[0][0] == "create"
    assert service.calls[1][0] == "list"
    assert service.calls[1][1]["page"] == 2
    assert service.calls[1][1]["query"] == "student"


def test_openapi_exposes_five_operations_without_delete() -> None:
    client, _ = build_client()

    schema = client.get("/openapi.json").json()
    collection = schema["paths"]["/api/v1/users"]
    detail = schema["paths"]["/api/v1/users/{user_id}"]
    reset = schema["paths"]["/api/v1/users/{user_id}/reset-password"]

    assert set(collection) == {"get", "post"}
    assert set(detail) == {"get", "patch"}
    assert set(reset) == {"post"}
    assert "delete" not in detail


def test_student_is_forbidden_and_teacher_scope_miss_is_404() -> None:
    client, _ = build_client()

    forbidden = client.get("/api/v1/users", headers={"X-Test-Role": "student"})
    hidden = client.get(
        f"/api/v1/users/{OTHER_ID}",
        headers={"X-Test-Role": "teacher"},
    )

    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "AUTH_FORBIDDEN"
    assert hidden.status_code == 404
    assert hidden.json()["error"]["code"] == "USER_NOT_FOUND"


def test_empty_patch_and_invalid_pagination_are_rejected() -> None:
    client, service = build_client()

    empty = client.patch(
        f"/api/v1/users/{STUDENT_ID}",
        json={},
        headers=safe_headers(client, "admin"),
    )
    invalid_page = client.get(
        "/api/v1/users?page=0&page_size=101",
        headers={"X-Test-Role": "admin"},
    )

    assert empty.status_code == 422
    assert invalid_page.status_code == 422
    assert service.calls == []


def test_mutations_require_origin_and_csrf() -> None:
    client, service = build_client()
    payload = {
        "username": "student-a",
        "password": "temporary-123",
        "role": "student",
    }

    missing_csrf = client.post(
        "/api/v1/users",
        json=payload,
        headers={"X-Test-Role": "admin", "Origin": settings.ALLOWED_ORIGINS[0]},
    )
    untrusted = client.post(
        "/api/v1/users",
        json=payload,
        headers={
            **safe_headers(client, "admin"),
            "Origin": "https://untrusted.example",
        },
    )

    assert missing_csrf.status_code == 403
    assert untrusted.status_code == 403
    assert service.calls == []


def test_update_and_password_reset_return_only_public_contracts() -> None:
    client, service = build_client()

    updated = client.patch(
        f"/api/v1/users/{STUDENT_ID}",
        json={"username": "renamed-student"},
        headers=safe_headers(client, "admin"),
    )
    reset = client.post(
        f"/api/v1/users/{STUDENT_ID}/reset-password",
        json={"password": "replacement-123"},
        headers=safe_headers(client, "admin"),
    )

    assert updated.status_code == 200
    assert "password" not in updated.json()
    assert "password_hash" not in updated.json()
    assert reset.status_code == 204
    assert reset.content == b""
    assert service.calls[-1][0] == "reset"
