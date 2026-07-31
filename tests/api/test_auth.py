"""认证 API 与 Cookie 生命周期测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings
from app.core.exception_handlers import install_exception_handlers
from app.core.middleware import RequestIdMiddleware
from app.api.knowledge import router as knowledge_router
from app.modules.auth.router import get_auth_service, router, set_auth_cookies
from app.modules.auth.service import AuthService, IssuedSession
from app.modules.auth.security import hash_password
from app.modules.auth.models import UserSession
from app.modules.users.models import User
from app.modules.users.schemas import UserRead
from tests.integration.database_safety import require_test_database_url


def build_app(service: AuthService) -> FastAPI:
    app = FastAPI()
    install_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)
    app.include_router(knowledge_router)
    app.dependency_overrides[get_auth_service] = lambda: service
    return app


async def seed_user(session_factory: async_sessionmaker, *, role: str = "user") -> None:
    async with session_factory() as session:
        async with session.begin():
            await session.execute(delete(UserSession))
            await session.execute(delete(User).where(User.username == "auth-api-user"))
            session.add(
                User(
                    username="auth-api-user",
                    email="auth-api@example.local",
                    password_hash=await hash_password("auth-api-password"),
                    role=role,
                    status="active",
                )
            )


def test_login_me_logout_enforces_cookie_and_csrf_contract() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    engine = create_async_engine(database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    asyncio.run(seed_user(session_factory))
    service = AuthService(
        session_factory,
        session_ttl_seconds=3600,
        csrf_secret=Settings().SESSION_SECRET,
    )
    client = TestClient(build_app(service))
    try:
        anonymous_knowledge = client.post(
            "/api/knowledge/extract",
            json={"text": "预览文本", "save": False},
            headers={"Origin": "http://localhost:8000"},
        )
        assert anonymous_knowledge.status_code == 401

        invalid = client.post(
            "/api/v1/auth/login",
            json={"username": "unknown", "password": "wrong"},
            headers={"Origin": "http://localhost:8000"},
        )
        assert invalid.status_code == 401
        assert invalid.json()["error"]["code"] == "AUTH_INVALID_CREDENTIALS"

        login = client.post(
            "/api/v1/auth/login",
            json={"username": "AUTH-API-USER", "password": "auth-api-password"},
            headers={"Origin": "http://localhost:8000"},
        )
        assert login.status_code == 200
        assert login.json()["username"] == "auth-api-user"
        assert login.json()["role"] == "user"
        assert "password" not in login.text
        session_token = client.cookies.get("mathrag_session")
        csrf_token = client.cookies.get("mathrag_csrf")
        assert session_token
        assert csrf_token

        forbidden_knowledge = client.post(
            "/api/knowledge/extract",
            json={"text": "预览文本", "save": False},
            headers={
                "Origin": "http://localhost:8000",
                "X-CSRF-Token": csrf_token,
            },
        )
        assert forbidden_knowledge.status_code == 403

        me = client.get("/api/v1/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "auth-api-user"

        missing_csrf = client.post(
            "/api/v1/auth/logout",
            headers={"Origin": "http://localhost:8000"},
        )
        assert missing_csrf.status_code == 403
        assert missing_csrf.json()["error"]["code"] == "AUTH_CSRF_INVALID"

        logout = client.post(
            "/api/v1/auth/logout",
            headers={
                "Origin": "http://localhost:8000",
                "X-CSRF-Token": csrf_token,
            },
        )
        assert logout.status_code == 204
        assert client.cookies.get("mathrag_session") is None
        assert client.cookies.get("mathrag_csrf") is None
        assert client.get("/api/v1/auth/me").status_code == 401

        client.cookies.set("mathrag_session", session_token)
        client.cookies.set("mathrag_csrf", csrf_token)
        repeated_logout = client.post(
            "/api/v1/auth/logout",
            headers={
                "Origin": "http://localhost:8000",
                "X-CSRF-Token": csrf_token,
            },
        )
        assert repeated_logout.status_code == 204
    finally:
        asyncio.run(engine.dispose())


def test_production_cookie_attributes_are_exact() -> None:
    configured = Settings(
        APP_ENV="production",
        DATABASE_URL="postgresql+asyncpg://user:password@localhost/mathrag",
        SESSION_SECRET="s" * 32,
        ALLOWED_ORIGINS=("https://mathrag.example",),
    )
    now = datetime(2026, 7, 31, tzinfo=UTC)
    user = UserRead(
        id=__import__("uuid").uuid4(),
        username="alice",
        email=None,
        role="user",
        status="active",
        created_at=now,
        updated_at=now,
    )
    issued = IssuedSession(
        session_id=__import__("uuid").uuid4(),
        user=user,
        raw_token="raw-token",
        csrf_token="csrf-token",
        expires_at=now,
    )
    from starlette.responses import Response

    response = Response()
    set_auth_cookies(response, issued, configured)
    cookies = [value.decode("latin-1") for key, value in response.raw_headers if key == b"set-cookie"]

    session_cookie = next(value for value in cookies if value.startswith("__Host-mathrag_session="))
    csrf_cookie = next(value for value in cookies if value.startswith("__Host-mathrag_csrf="))
    assert "HttpOnly" in session_cookie
    assert "HttpOnly" not in csrf_cookie
    for cookie in cookies:
        assert "Secure" in cookie
        assert "SameSite=lax" in cookie
        assert "Path=/" in cookie
        assert "Domain=" not in cookie
