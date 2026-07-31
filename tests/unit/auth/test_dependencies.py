"""Origin、CSRF 和角色依赖测试。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from starlette.requests import Request

from app.core.config import Settings
from app.core.errors import AppError
from app.modules.auth.dependencies import (
    AuthenticatedPrincipal,
    require_admin,
    validate_csrf_request,
    validate_request_origin,
)
from app.modules.auth.security import issue_csrf_token


def make_request(
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> Request:
    encoded_headers = [
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    ]
    if cookies:
        encoded_headers.append(
            (b"cookie", "; ".join(f"{key}={value}" for key, value in cookies.items()).encode())
        )
    return Request(
        {
            "type": "http",
            "method": method,
            "scheme": "https",
            "path": "/api/v1/probe",
            "query_string": b"",
            "headers": encoded_headers,
            "server": ("mathrag.example", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


def principal(session_hash: bytes = b"x" * 32, role: str = "user") -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=uuid4(),
        session_id=uuid4(),
        username="alice",
        role=role,  # type: ignore[arg-type]
        session_token_hash=session_hash,
    )


def test_origin_uses_explicit_allowlist_and_referer_fallback() -> None:
    configured = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql+asyncpg://user:password@localhost/mathrag",
        SESSION_SECRET="s" * 32,
        ALLOWED_ORIGINS=("https://mathrag.example",),
    )

    validate_request_origin(
        make_request(headers={"Origin": "https://mathrag.example"}),
        configured,
    )
    validate_request_origin(
        make_request(headers={"Referer": "https://mathrag.example/chat/1"}),
        configured,
    )
    with pytest.raises(AppError) as exc_info:
        validate_request_origin(
            make_request(headers={"Origin": "https://attacker.example"}),
            configured,
        )
    assert exc_info.value.code == "AUTH_ORIGIN_INVALID"


def test_csrf_requires_matching_cookie_header_signature_and_session_binding() -> None:
    configured = Settings()
    session_hash = b"a" * 32
    csrf_token = issue_csrf_token(session_hash, configured.SESSION_SECRET)
    request = make_request(
        headers={
            "Origin": "http://localhost:8000",
            "X-CSRF-Token": csrf_token,
        },
        cookies={configured.csrf_cookie_name: csrf_token},
    )

    validate_csrf_request(request, principal(session_hash), configured)
    with pytest.raises(AppError) as exc_info:
        validate_csrf_request(request, principal(b"b" * 32), configured)
    assert exc_info.value.code == "AUTH_CSRF_INVALID"


def test_require_admin_rejects_regular_user() -> None:
    assert asyncio.run(require_admin(principal(role="admin"))).role == "admin"
    with pytest.raises(AppError) as exc_info:
        asyncio.run(require_admin(principal(role="user")))
    assert exc_info.value.code == "AUTH_FORBIDDEN"
    assert exc_info.value.status_code == 403
