"""服务端 Session 服务测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.core.errors import AppError
from app.modules.auth.repository import ActiveSessionRecord, LoginUserRecord
from app.modules.auth.service import AuthService, DUMMY_PASSWORD_HASH


class AsyncContext:
    async def __aenter__(self) -> object:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> "AsyncContext":
        return self


class FakeSessionFactory:
    def __call__(self) -> AsyncContext:
        return AsyncContext()


class FakeAuthRepository:
    def __init__(self) -> None:
        self.login_user: LoginUserRecord | None = None
        self.active_session: ActiveSessionRecord | None = None
        self.added_sessions: list[object] = []
        self.touched: list[tuple[object, datetime]] = []

    async def find_login_user(self, username: str) -> LoginUserRecord | None:
        return self.login_user if self.login_user and self.login_user.username == username else None

    async def find_login_user_by_id(self, user_id: object) -> LoginUserRecord | None:
        return self.login_user if self.login_user and self.login_user.id == user_id else None

    def add_session(self, session: object) -> None:
        self.added_sessions.append(session)

    async def find_active_by_hash(
        self,
        token_hash: bytes,
        now: datetime,
    ) -> ActiveSessionRecord | None:
        return self.active_session

    async def touch_last_seen(self, session_id: object, now: datetime) -> None:
        self.touched.append((session_id, now))

    async def revoke(self, session_id: object, now: datetime) -> None:
        return None


def test_unknown_user_runs_dummy_verify_and_uses_generic_error(monkeypatch) -> None:
    repository = FakeAuthRepository()
    verified_hashes: list[str] = []

    async def fake_verify(_password: str, encoded_hash: str) -> bool:
        verified_hashes.append(encoded_hash)
        return False

    monkeypatch.setattr("app.modules.auth.service.verify_password", fake_verify)
    service = AuthService(
        FakeSessionFactory(),  # type: ignore[arg-type]
        repository_factory=lambda _session: repository,
        session_ttl_seconds=60,
        csrf_secret="s" * 32,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.login("unknown", "private-password", datetime.now(UTC)))

    assert exc_info.value.code == "AUTH_INVALID_CREDENTIALS"
    assert exc_info.value.status_code == 401
    assert verified_hashes == [DUMMY_PASSWORD_HASH]
    assert "private-password" not in str(exc_info.value)


def test_resolve_returns_immutable_snapshot_and_throttles_last_seen_write() -> None:
    repository = FakeAuthRepository()
    now = datetime(2026, 7, 31, tzinfo=UTC)
    session_id = uuid4()
    user_id = uuid4()
    repository.active_session = ActiveSessionRecord(
        session_id=session_id,
        user_id=user_id,
        username="alice",
        role="admin",
        token_hash=b"x" * 32,
        last_seen_at=now - timedelta(minutes=6),
    )
    service = AuthService(
        FakeSessionFactory(),  # type: ignore[arg-type]
        repository_factory=lambda _session: repository,
        session_ttl_seconds=60,
        csrf_secret="s" * 32,
    )

    principal = asyncio.run(service.resolve("raw-token", now))

    assert principal.user_id == user_id
    assert principal.session_id == session_id
    assert principal.role == "admin"
    assert repository.touched == [(session_id, now)]
    with pytest.raises(Exception):
        principal.username = "changed"  # type: ignore[misc]
