from __future__ import annotations

from uuid import UUID

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app.core.errors import AppError
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.modules.knowledge.management_schemas import KnowledgeItemUpdate


class ValidationPayload(BaseModel):
    password: str = Field(min_length=12)


def _build_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    install_exception_handlers(app)

    @app.post("/api/v1/probe")
    @app.post("/api/probe")
    async def validate_payload(payload: ValidationPayload) -> dict[str, str]:
        return {"password": payload.password}

    @app.get("/api/v1/app-error")
    async def raise_app_error() -> None:
        raise AppError(
            code="PROBE_REJECTED",
            message="探针请求被拒绝。",
            status_code=409,
            details={"reason": "conflict"},
        )

    @app.get("/api/v1/http-error")
    async def raise_v1_http_error() -> None:
        raise HTTPException(status_code=404, detail="探针不存在。")

    @app.get("/api/http-error")
    async def raise_legacy_http_error() -> None:
        raise HTTPException(status_code=404, detail="探针不存在。")

    @app.get("/api/v1/internal-error")
    async def raise_internal_error() -> None:
        raise RuntimeError("database password=do-not-leak")

    @app.patch("/api/v1/knowledge-items/{item_id}")
    async def validate_knowledge_update(
        item_id: UUID,
        payload: KnowledgeItemUpdate,
    ) -> dict[str, str]:
        return {"item_id": str(item_id), "title": payload.title or ""}

    return app


def test_app_error_string_contains_only_code_and_details_default_is_isolated() -> None:
    first = AppError(code="FIRST", message="敏感消息", status_code=400)
    second = AppError(code="SECOND", message="另一条消息", status_code=409)
    first.details["changed"] = True

    assert str(first) == "FIRST"
    assert second.details == {}


def test_v1_validation_error_uses_stable_envelope_and_request_id() -> None:
    client = TestClient(_build_test_app())

    response = client.post(
        "/api/v1/probe",
        json={"password": "private-123"},
        headers={"X-Request-ID": "m4-validation-001"},
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "m4-validation-001"
    assert response.json()["error"] == {
        "code": "REQUEST_VALIDATION_FAILED",
        "message": "请求参数校验失败。",
        "request_id": "m4-validation-001",
        "details": [
            {
                "loc": ["body", "password"],
                "type": "string_too_short",
                "msg": "String should have at least 12 characters",
            }
        ],
    }
    assert "input" not in response.text
    assert "private-123" not in response.text


def test_v1_knowledge_validation_does_not_echo_rejected_content() -> None:
    secret = "private-knowledge-content"
    response = TestClient(_build_test_app()).patch(
        "/api/v1/knowledge-items/33333333-3333-4333-8333-333333333333",
        json={"revision": 0, "content": secret, "status": secret},
        headers={"X-Request-ID": "m5-knowledge-validation"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert response.json()["error"]["request_id"] == "m5-knowledge-validation"
    assert secret not in response.text
    assert all(
        "input" not in detail
        for detail in response.json()["error"]["details"]
    )


def test_v1_app_error_uses_its_public_fields() -> None:
    response = TestClient(_build_test_app()).get(
        "/api/v1/app-error",
        headers={"X-Request-ID": "m4-app-error-001"},
    )

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "PROBE_REJECTED",
            "message": "探针请求被拒绝。",
            "request_id": "m4-app-error-001",
            "details": {"reason": "conflict"},
        }
    }


def test_v1_http_exception_uses_error_envelope() -> None:
    response = TestClient(_build_test_app()).get(
        "/api/v1/http-error",
        headers={"X-Request-ID": "m4-http-error-001"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "HTTP_ERROR",
            "message": "探针不存在。",
            "request_id": "m4-http-error-001",
            "details": {},
        }
    }


def test_v1_unknown_exception_is_fixed_and_does_not_leak_text() -> None:
    client = TestClient(_build_test_app(), raise_server_exceptions=False)

    response = client.get(
        "/api/v1/internal-error",
        headers={"X-Request-ID": "m4-internal-error-001"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "m4-internal-error-001"
    assert response.json() == {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误。",
            "request_id": "m4-internal-error-001",
            "details": {},
        }
    }
    assert "database" not in response.text
    assert "do-not-leak" not in response.text


def test_legacy_api_keeps_detail_responses() -> None:
    client = TestClient(_build_test_app())

    validation_response = client.post("/api/probe", json={"password": "short"})
    http_response = client.get("/api/http-error")

    assert validation_response.status_code == 422
    assert set(validation_response.json()) == {"detail"}
    assert http_response.status_code == 404
    assert http_response.json() == {"detail": "探针不存在。"}
