"""知识管理 API 的路由、权限与错误契约测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.errors import AppError
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.modules.auth.dependencies import get_current_principal, require_admin_csrf
from app.modules.auth.security import issue_csrf_token
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import (
    KnowledgeNotFoundError,
    KnowledgeRevisionConflictError,
)
from app.modules.knowledge.management_schemas import (
    KnowledgeItemCreate,
    KnowledgeItemPage,
    KnowledgeItemRead,
    KnowledgeItemUpdate,
)
from app.modules.knowledge.router import (
    get_knowledge_management_service,
    get_knowledge_read_service,
    router,
)


USER_ID = UUID("11111111-1111-4111-8111-111111111111")
ADMIN_ID = UUID("22222222-2222-4222-8222-222222222222")
USER_SESSION_HASH = b"u" * 32
ADMIN_SESSION_HASH = b"a" * 32
TRUSTED_ORIGIN = settings.ALLOWED_ORIGINS[0]


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


def _item(**changes: object) -> KnowledgeItemRead:
    now = datetime(2026, 7, 31, tzinfo=UTC)
    values: dict[str, object] = {
        "id": UUID("33333333-3333-4333-8333-333333333333"),
        "legacy_id": None,
        "owner_id": ADMIN_ID,
        "category": "algebra",
        "title": "一元二次方程",
        "keywords": ["方程"],
        "content": "使用求根公式。",
        "example": "x² - 1 = 0",
        "steps": ["整理", "求根"],
        "difficulty": "medium",
        "visibility": "public",
        "status": "ready",
        "revision": 7,
        "created_at": now,
        "updated_at": now,
    }
    values.update(changes)
    return KnowledgeItemRead.model_validate(values)


def _create_payload() -> dict[str, object]:
    return {
        "category": "algebra",
        "title": "一元二次方程",
        "keywords": ["方程"],
        "content": "使用求根公式。",
        "example": "x² - 1 = 0",
        "steps": ["整理", "求根"],
        "difficulty": "medium",
        "visibility": "public",
    }


class FakeKnowledgeManagementService:
    def __init__(self) -> None:
        self.item = _item()
        self.calls: list[tuple[str, object]] = []
        self.get_error: AppError | None = None
        self.update_error: AppError | None = None

    async def get(
        self,
        item_id: UUID,
        principal: AuthenticatedPrincipal,
    ) -> KnowledgeItemRead:
        self.calls.append(("get", (item_id, principal)))
        if self.get_error is not None:
            raise self.get_error
        return self.item

    async def list(
        self,
        principal: AuthenticatedPrincipal,
        *,
        status: str | None,
        visibility: str | None,
        category: str | None,
        page: int,
        page_size: int,
    ) -> KnowledgeItemPage:
        self.calls.append(
            (
                "list",
                {
                    "principal": principal,
                    "status": status,
                    "visibility": visibility,
                    "category": category,
                    "page": page,
                    "page_size": page_size,
                },
            )
        )
        return KnowledgeItemPage(
            items=[self.item],
            page=page,
            page_size=page_size,
            total=1,
        )

    async def create(
        self,
        owner_id: UUID,
        payload: KnowledgeItemCreate,
    ) -> KnowledgeItemRead:
        self.calls.append(("create", (owner_id, payload)))
        return self.item

    async def update(
        self,
        item_id: UUID,
        payload: KnowledgeItemUpdate,
    ) -> KnowledgeItemRead:
        self.calls.append(("update", (item_id, payload)))
        if self.update_error is not None:
            raise self.update_error
        return self.item

    async def archive(self, item_id: UUID, expected_revision: int) -> None:
        self.calls.append(("archive", (item_id, expected_revision)))


def _build_client(
    service: FakeKnowledgeManagementService | None = None,
) -> tuple[TestClient, FakeKnowledgeManagementService]:
    fake = service or FakeKnowledgeManagementService()
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_knowledge_read_service] = lambda: fake
    app.dependency_overrides[get_knowledge_management_service] = lambda: fake

    async def principal_from_header(request: Request) -> AuthenticatedPrincipal:
        role = request.headers.get("X-Test-Role")
        if role not in {"student", "admin"}:
            raise AppError(
                code="AUTH_SESSION_INVALID",
                message="登录状态无效或已过期。",
                status_code=401,
            )
        return _principal(role)

    app.dependency_overrides[get_current_principal] = principal_from_header
    return TestClient(app), fake


def _safe_headers(client: TestClient, role: str) -> dict[str, str]:
    session_hash = ADMIN_SESSION_HASH if role == "admin" else USER_SESSION_HASH
    token = issue_csrf_token(session_hash, settings.SESSION_SECRET)
    client.cookies.set(settings.csrf_cookie_name, token)
    return {
        "X-Test-Role": role,
        "X-CSRF-Token": token,
        "Origin": TRUSTED_ORIGIN,
    }


def _request_mutation(
    client: TestClient,
    method: str,
    item_id: UUID,
    headers: dict[str, str] | None = None,
):
    if method == "post":
        return client.post(
            "/api/v1/knowledge-items",
            json=_create_payload(),
            headers=headers,
        )
    if method == "patch":
        return client.patch(
            f"/api/v1/knowledge-items/{item_id}",
            json={"revision": 7, "title": "更新标题"},
            headers=headers,
        )
    return client.delete(
        f"/api/v1/knowledge-items/{item_id}?revision=7",
        headers=headers,
    )


def test_openapi_exposes_all_knowledge_item_operations_and_response_models() -> None:
    client, _service = _build_client()

    schema = client.get("/openapi.json").json()
    collection = schema["paths"]["/api/v1/knowledge-items"]
    detail = schema["paths"]["/api/v1/knowledge-items/{item_id}"]

    assert set(collection) == {"get", "post"}
    assert set(detail) == {"get", "patch", "delete"}
    assert collection["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/KnowledgeItemPage")
    assert collection["post"]["responses"]["201"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/KnowledgeItemRead")
    assert detail["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]["$ref"].endswith("/KnowledgeItemRead")


def test_get_routes_require_authentication_and_pass_exact_arguments() -> None:
    client, service = _build_client()
    item_path = f"/api/v1/knowledge-items/{service.item.id}"

    anonymous = client.get("/api/v1/knowledge-items")
    detail = client.get(item_path, headers={"X-Test-Role": "student"})
    listing = client.get(
        "/api/v1/knowledge-items",
        params={
            "status": "draft",
            "visibility": "private",
            "category": "algebra",
            "page": 2,
            "page_size": 5,
        },
        headers={"X-Test-Role": "student"},
    )

    assert anonymous.status_code == 401
    assert anonymous.json()["error"]["code"] == "AUTH_SESSION_INVALID"
    assert detail.status_code == 200
    assert detail.json()["id"] == str(service.item.id)
    assert listing.status_code == 200
    assert listing.json()["page"] == 2
    assert service.calls[0][0] == "get"
    detail_id, detail_principal = service.calls[0][1]
    assert detail_id == service.item.id
    assert detail_principal.user_id == USER_ID
    call_name, list_arguments = service.calls[1]
    assert call_name == "list"
    assert list_arguments["principal"].user_id == USER_ID
    assert list_arguments["principal"].role == "student"
    assert {key: value for key, value in list_arguments.items() if key != "principal"} == {
        "status": "draft",
        "visibility": "private",
        "category": "algebra",
        "page": 2,
        "page_size": 5,
    }


@pytest.mark.parametrize(
    "query",
    [
        "status=unknown",
        "visibility=internal",
        "page=0",
        "page_size=0",
        "page_size=101",
    ],
)
def test_collection_rejects_invalid_frozen_queries(query: str) -> None:
    client, service = _build_client()

    response = client.get(
        f"/api/v1/knowledge-items?{query}",
        headers={"X-Test-Role": "student"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert service.calls == []


def test_admin_mutations_return_exact_status_and_service_arguments() -> None:
    client, service = _build_client()
    headers = _safe_headers(client, "admin")
    item_path = f"/api/v1/knowledge-items/{service.item.id}"

    created = client.post(
        "/api/v1/knowledge-items",
        json=_create_payload(),
        headers=headers,
    )
    updated = client.patch(
        item_path,
        json={"revision": 7, "content": "更新后的内容"},
        headers=headers,
    )
    deleted = client.delete(f"{item_path}?revision=7", headers=headers)

    assert created.status_code == 201
    assert created.json()["id"] == str(service.item.id)
    assert updated.status_code == 200
    assert deleted.status_code == 204
    assert deleted.content == b""
    create_owner, create_request = service.calls[0][1]
    assert create_owner == ADMIN_ID
    assert create_request == KnowledgeItemCreate.model_validate(_create_payload())
    update_id, update_request = service.calls[1][1]
    assert update_id == service.item.id
    assert update_request == KnowledgeItemUpdate(revision=7, content="更新后的内容")
    assert service.calls[2] == ("archive", (service.item.id, 7))


@pytest.mark.parametrize("method", ["post", "patch", "delete"])
def test_each_mutation_rejects_anonymous_and_ordinary_user(method: str) -> None:
    client, service = _build_client()

    anonymous = _request_mutation(client, method, service.item.id)
    user = _request_mutation(
        client,
        method,
        service.item.id,
        _safe_headers(client, "student"),
    )

    assert anonymous.status_code == 401
    assert user.status_code == 403
    assert user.json()["error"]["code"] == "AUTH_FORBIDDEN"
    assert service.calls == []


@pytest.mark.parametrize("method", ["post", "patch", "delete"])
def test_each_mutation_rejects_invalid_admin_csrf_or_origin(method: str) -> None:
    service = FakeKnowledgeManagementService()
    missing_csrf_client, _ = _build_client(service)
    missing_csrf = _request_mutation(
        missing_csrf_client,
        method,
        service.item.id,
        {"X-Test-Role": "admin", "Origin": TRUSTED_ORIGIN},
    )
    wrong_csrf_client, _ = _build_client(service)
    wrong_csrf_client.cookies.set(settings.csrf_cookie_name, "wrong")
    wrong_csrf = _request_mutation(
        wrong_csrf_client,
        method,
        service.item.id,
        {
            "X-Test-Role": "admin",
            "X-CSRF-Token": "wrong",
            "Origin": TRUSTED_ORIGIN,
        },
    )
    wrong_origin_client, _ = _build_client(service)
    wrong_origin_headers = _safe_headers(wrong_origin_client, "admin")
    wrong_origin_headers["Origin"] = "https://untrusted.example"
    wrong_origin = _request_mutation(
        wrong_origin_client,
        method,
        service.item.id,
        wrong_origin_headers,
    )

    assert missing_csrf.status_code == 403
    assert missing_csrf.json()["error"]["code"] == "AUTH_CSRF_INVALID"
    assert wrong_csrf.status_code == 403
    assert wrong_csrf.json()["error"]["code"] == "AUTH_CSRF_INVALID"
    assert wrong_origin.status_code == 403
    assert wrong_origin.json()["error"]["code"] == "AUTH_ORIGIN_INVALID"
    assert service.calls == []


def test_invalid_mutation_body_and_revision_are_redacted_validation_errors() -> None:
    client, service = _build_client()
    headers = _safe_headers(client, "admin")
    secret = "private-content-that-must-not-leak"

    invalid_body = client.post(
        "/api/v1/knowledge-items",
        json={**_create_payload(), "content": secret, "unexpected": secret},
        headers=headers,
    )
    invalid_revision = client.delete(
        f"/api/v1/knowledge-items/{service.item.id}?revision=0",
        headers=headers,
    )

    assert invalid_body.status_code == 422
    assert invalid_body.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert secret not in invalid_body.text
    assert all(
        "input" not in detail
        for detail in invalid_body.json()["error"]["details"]
    )
    assert invalid_revision.status_code == 422
    assert service.calls == []


def test_domain_404_and_revision_conflict_use_v1_envelope_with_request_id() -> None:
    service = FakeKnowledgeManagementService()
    service.get_error = KnowledgeNotFoundError()
    client, _ = _build_client(service)
    missing = client.get(
        f"/api/v1/knowledge-items/{uuid4()}",
        headers={"X-Test-Role": "student", "X-Request-ID": "knowledge-missing"},
    )

    service.get_error = None
    service.update_error = KnowledgeRevisionConflictError()
    conflict = client.patch(
        f"/api/v1/knowledge-items/{service.item.id}",
        json={"revision": 7, "content": "并发后的旧请求"},
        headers={
            **_safe_headers(client, "admin"),
            "X-Request-ID": "knowledge-conflict",
        },
    )

    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "KNOWLEDGE_NOT_FOUND"
    assert missing.json()["error"]["request_id"] == "knowledge-missing"
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "KNOWLEDGE_REVISION_CONFLICT"
    assert conflict.json()["error"]["request_id"] == "knowledge-conflict"


def test_read_and_write_service_dependencies_initialize_only_required_resources(
    monkeypatch,
) -> None:
    from app.modules.knowledge import router as router_module

    calls: list[str] = []
    session_factory = object()
    provider = object()

    monkeypatch.setattr(
        router_module,
        "get_session_factory",
        lambda: calls.append("session") or session_factory,
    )
    monkeypatch.setattr(
        router_module,
        "get_embedding_provider",
        lambda: calls.append("embedding") or provider,
    )

    assert calls == []
    read_service = get_knowledge_read_service()
    assert calls == ["session"]
    assert read_service._session_factory is session_factory
    assert read_service._provider is None

    write_service = get_knowledge_management_service()
    assert calls == ["session", "session", "embedding"]
    assert write_service._session_factory is session_factory
    assert write_service._provider is provider


def test_main_app_assembles_routes_once_and_allows_overridden_dependencies(
    monkeypatch,
) -> None:
    from app.main import create_app
    from app.modules.knowledge import router as router_module

    service = FakeKnowledgeManagementService()
    provider_calls: list[str] = []
    monkeypatch.setattr(
        router_module,
        "get_embedding_provider",
        lambda: provider_calls.append("embedding") or object(),
    )
    app = create_app()
    app.dependency_overrides[get_current_principal] = lambda: _principal("student")
    app.dependency_overrides[require_admin_csrf] = lambda: _principal("admin")
    app.dependency_overrides[get_knowledge_read_service] = lambda: service
    app.dependency_overrides[get_knowledge_management_service] = lambda: service
    client = TestClient(app)

    document = client.get("/openapi.json").json()
    collection_path = "/api/v1/knowledge-items"
    detail_path = "/api/v1/knowledge-items/{item_id}"
    assert set(document["paths"][collection_path]) == {"get", "post"}
    assert set(document["paths"][detail_path]) == {"get", "patch", "delete"}
    assert sum(
        getattr(included, "original_router", None) is router
        for included in app.routes
    ) == 1

    listing = client.get(collection_path)
    detail = client.get(f"{collection_path}/{service.item.id}")
    deleted = client.delete(f"{collection_path}/{service.item.id}?revision=7")

    assert listing.status_code == 200
    assert detail.status_code == 200
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert provider_calls == []
