"""受认证保护的持久化聊天 API 契约。"""

from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.main import app as main_app
from app.modules.auth.dependencies import (
    get_current_principal,
    require_csrf,
)
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.conversations.errors import ConversationNotFoundError
from app.modules.rag.errors import ConversationArchivedError
from app.modules.rag.repository import PersistedChatResult
from app.modules.rag.router import get_chat_persistence_service, router


USER_ID = UUID("00000000-0000-0000-0000-000000000201")
CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000202")
CLIENT_REQUEST_ID = UUID("00000000-0000-0000-0000-000000000203")
DATABASE_CHUNK_ID = UUID("00000000-0000-0000-0000-000000000204")


def make_principal() -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=USER_ID,
        session_id=uuid4(),
        username="v1-user",
        role="student",
        must_change_password=False,
        session_token_hash=b"x" * 32,
    )


def make_result() -> PersistedChatResult:
    return PersistedChatResult(
        conversation_id=CONVERSATION_ID,
        question_message_id=UUID(int=301),
        answer_message_id=UUID(int=302),
        rag_run_id=UUID(int=303),
        client_request_id=CLIENT_REQUEST_ID,
        response={
            "question": "导数是什么？",
            "answer": "导数描述瞬时变化率。",
            "steps": ["理解极限"],
            "used_knowledge": ["导数定义"],
            "related_questions": ["几何意义是什么？"],
            "references": [
                {
                    "rank": 1,
                    "score": 0.9,
                    "index": None,
                    "chunk_id": "legacy-chunk-1",
                    "source_id": "legacy-source-1",
                    "category": "calculus",
                    "title": "导数定义",
                    "keywords": ["导数"],
                    "content": "定义内容",
                    "example": "例子",
                    "steps": ["求极限"],
                    "difficulty": "medium",
                    "answer_context": "上下文",
                    "retrieval_text": "检索文本",
                    "source_line": 1,
                    "metadata": {},
                }
            ],
            "agentic_plan": {
                "strategy": "single",
                "retrieval_queries": ["导数是什么？"],
            },
            "reasoning_content": None,
        },
    )


class FakeChatService:
    def __init__(self) -> None:
        self.calls = 0
        self.error: Exception | None = None

    async def chat(self, **kwargs) -> PersistedChatResult:
        self.calls += 1
        assert kwargs["principal"].user_id == USER_ID
        assert kwargs["conversation_id"] == CONVERSATION_ID
        assert kwargs["client_request_id"] == CLIENT_REQUEST_ID
        assert kwargs["question"] == "导数是什么？"
        assert kwargs["top_k"] == 3
        if self.error is not None:
            raise self.error
        return make_result()


def build_client(
    service: FakeChatService,
    *,
    override_csrf: bool,
    override_principal: bool = False,
) -> TestClient:
    application = FastAPI()
    install_exception_handlers(application)
    application.add_middleware(RequestIdMiddleware)
    application.include_router(router)
    application.dependency_overrides[get_chat_persistence_service] = lambda: service
    if override_csrf:
        application.dependency_overrides[require_csrf] = make_principal
    if override_principal:
        application.dependency_overrides[get_current_principal] = make_principal
    return TestClient(application)


def valid_payload() -> dict[str, object]:
    return {
        "conversation_id": str(CONVERSATION_ID),
        "client_request_id": str(CLIENT_REQUEST_ID),
        "question": " 导数是什么？ ",
        "top_k": 3,
    }


def test_v1_chat_requires_session_and_csrf() -> None:
    service = FakeChatService()
    anonymous = build_client(service, override_csrf=False)
    no_session = anonymous.post(
        "/api/v1/chat",
        json=valid_payload(),
        headers={"Origin": "http://localhost:8000"},
    )
    assert no_session.status_code == 401
    assert no_session.json()["error"]["code"] == "AUTH_SESSION_INVALID"

    missing_csrf = build_client(
        service,
        override_csrf=False,
        override_principal=True,
    ).post(
        "/api/v1/chat",
        json=valid_payload(),
        headers={"Origin": "http://localhost:8000"},
    )
    assert missing_csrf.status_code == 403
    assert missing_csrf.json()["error"]["code"] == "AUTH_CSRF_INVALID"
    assert service.calls == 0


def test_v1_chat_returns_persistence_ids_without_database_chunk_uuid() -> None:
    service = FakeChatService()
    response = build_client(service, override_csrf=True).post(
        "/api/v1/chat",
        json=valid_payload(),
        headers={"X-Request-ID": "v1-chat-success"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] == str(CONVERSATION_ID)
    assert data["client_request_id"] == str(CLIENT_REQUEST_ID)
    assert data["references"][0]["chunk_id"] == "legacy-chunk-1"
    assert str(DATABASE_CHUNK_ID) not in response.text
    assert service.calls == 1


def test_v1_chat_validation_and_domain_errors_use_v1_envelope() -> None:
    service = FakeChatService()
    client = build_client(service, override_csrf=True)
    invalid = client.post(
        "/api/v1/chat",
        json={**valid_payload(), "question": " ", "history": []},
        headers={"X-Request-ID": "v1-invalid"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert invalid.json()["error"]["request_id"] == "v1-invalid"

    service.error = ConversationNotFoundError()
    not_found = client.post("/api/v1/chat", json=valid_payload())
    assert not_found.status_code == 404
    assert not_found.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"

    service.error = ConversationArchivedError()
    archived = client.post("/api/v1/chat", json=valid_payload())
    assert archived.status_code == 409
    assert archived.json()["error"]["code"] == "CONVERSATION_ARCHIVED"


def test_openapi_declares_v1_chat_cookie_security_and_error_responses() -> None:
    document = TestClient(main_app).get("/openapi.json").json()

    assert "/api/v1/auth/login" in document["paths"]
    assert "/api/v1/conversations" in document["paths"]
    assert "/api/v1/chat" in document["paths"]
    operation = document["paths"]["/api/v1/chat"]["post"]
    assert {"401", "403", "404", "409", "422"} <= set(operation["responses"])
    assert operation["security"] == [{"SessionCookie": []}]
    schemes = document["components"]["securitySchemes"]
    assert any(
        scheme.get("type") == "apiKey"
        and scheme.get("in") == "cookie"
        and scheme.get("name") == "mathrag_session"
        for scheme in schemes.values()
    )
    serialized = str(document)
    assert "password_hash" not in serialized
    assert "token_hash" not in serialized
