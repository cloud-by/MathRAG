"""会话 REST API 契约测试。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.exception_handlers import install_exception_handlers
from app.core.errors import AppError
from app.core.middleware import RequestIdMiddleware
from app.modules.auth.dependencies import AuthenticatedPrincipal, get_current_principal, require_csrf
from app.modules.conversations.errors import ConversationNotFoundError
from app.modules.conversations.router import get_conversation_service, router
from app.modules.conversations.schemas import ConversationRead, MessagePage


class FakeConversationService:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.owner_id = uuid4()
        self.conversation = ConversationRead(
            id=uuid4(),
            title="私有会话",
            status="active",
            created_at=now,
            updated_at=now,
        )

    def _require_owner(self, user_id: UUID) -> None:
        if user_id != self.owner_id:
            raise ConversationNotFoundError()

    async def get(self, conversation_id: UUID, user_id: UUID) -> ConversationRead:
        self._require_owner(user_id)
        if conversation_id != self.conversation.id:
            raise ConversationNotFoundError()
        return self.conversation

    async def update(self, conversation_id: UUID, user_id: UUID, values):
        if not values:
            raise AppError(
                code="REQUEST_VALIDATION_FAILED",
                message="至少提供一个需要更新的字段。",
                status_code=422,
            )
        return await self.get(conversation_id, user_id)

    async def archive(self, conversation_id: UUID, user_id: UUID) -> None:
        await self.get(conversation_id, user_id)

    async def list_messages(self, conversation_id: UUID, user_id: UUID, *, page, page_size):
        await self.get(conversation_id, user_id)
        return MessagePage(items=[], page=page, page_size=page_size, total=0)


def build_client(service: FakeConversationService) -> TestClient:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = lambda: service

    async def principal_from_header(request: Request) -> AuthenticatedPrincipal:
        user_id = UUID(request.headers["X-Test-User"])
        return AuthenticatedPrincipal(
            user_id=user_id,
            session_id=uuid4(),
            username="test",
            role="student",
            must_change_password=False,
            session_token_hash=b"x" * 32,
        )

    app.dependency_overrides[get_current_principal] = principal_from_header
    app.dependency_overrides[require_csrf] = principal_from_header
    return TestClient(app)


def test_cross_user_resource_operations_share_same_404_envelope() -> None:
    service = FakeConversationService()
    client = build_client(service)
    other_headers = {"X-Test-User": str(uuid4()), "X-Request-ID": "cross-owner"}
    path = f"/api/v1/conversations/{service.conversation.id}"

    responses = [
        client.get(path, headers=other_headers),
        client.patch(path, json={"title": "不可见"}, headers=other_headers),
        client.delete(path, headers=other_headers),
        client.get(f"{path}/messages", headers=other_headers),
    ]

    assert {response.status_code for response in responses} == {404}
    assert {
        response.json()["error"]["code"] for response in responses
    } == {"CONVERSATION_NOT_FOUND"}
    assert all("私有会话" not in response.text for response in responses)


def test_pagination_and_empty_patch_use_v1_validation_envelope() -> None:
    service = FakeConversationService()
    client = build_client(service)
    headers = {"X-Test-User": str(service.owner_id)}
    path = f"/api/v1/conversations/{service.conversation.id}"

    page_response = client.get(f"{path}/messages?page_size=101", headers=headers)
    patch_response = client.patch(path, json={}, headers=headers)

    assert page_response.status_code == 422
    assert page_response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert patch_response.status_code == 422
    assert patch_response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
