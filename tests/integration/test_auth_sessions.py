"""认证 Session 的真实 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.errors import AppError
from app.modules.auth.models import UserSession
from app.modules.auth.security import hash_password, hash_session_token
from app.modules.auth.service import AuthService
from app.modules.users.models import User
from tests.integration.database_safety import require_test_database_url


async def exercise_session_lifecycle(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    try:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(UserSession))
                await session.execute(delete(User).where(User.username == "session-user"))
                session.add(
                    User(
                        username="session-user",
                        password_hash=await hash_password("session-password"),
                        role="admin",
                        status="active",
                    )
                )

        service = AuthService(
            session_factory,
            session_ttl_seconds=3600,
            csrf_secret="s" * 32,
        )
        issued = await service.login("session-user", "session-password", now)
        principal = await service.resolve(issued.raw_token, now + timedelta(seconds=1))
        assert principal.user_id == issued.user.id
        assert principal.must_change_password is False

        async with session_factory() as session:
            async with session.begin():
                user = await session.get(User, issued.user.id)
                assert user is not None
                user.must_change_password = True
        refreshed_principal = await service.resolve(
            issued.raw_token,
            now + timedelta(seconds=2),
        )
        assert refreshed_principal.must_change_password is True

        async with session_factory() as session:
            stored = await session.scalar(
                select(UserSession).where(UserSession.id == issued.session_id)
            )
            assert stored is not None
            assert stored.token_hash == hash_session_token(issued.raw_token)
            assert len(stored.token_hash) == 32
            assert issued.raw_token.encode("utf-8") != stored.token_hash

        # 模拟应用容器时钟略落后于 PostgreSQL；撤销时间必须被钳制到 created_at。
        await service.logout(issued.session_id, now - timedelta(seconds=1))
        async with session_factory() as session:
            revoked = await session.get(UserSession, issued.session_id)
            assert revoked is not None and revoked.revoked_at is not None
            assert revoked.revoked_at >= revoked.created_at
        with pytest.raises(AppError) as revoked_error:
            await service.resolve(issued.raw_token, now + timedelta(seconds=3))
        assert revoked_error.value.code == "AUTH_SESSION_INVALID"

        expiring_service = AuthService(
            session_factory,
            session_ttl_seconds=1,
            csrf_secret="s" * 32,
        )
        expiring = await expiring_service.login("session-user", "session-password", now)
        with pytest.raises(AppError) as expired_error:
            await expiring_service.resolve(expiring.raw_token, now + timedelta(seconds=2))
        assert expired_error.value.code == "AUTH_SESSION_INVALID"

        issued = await service.login("session-user", "session-password", now)
        async with session_factory() as session:
            async with session.begin():
                user = await session.get(User, issued.user.id)
                assert user is not None
                user.status = "disabled"
                user.updated_at = now
        with pytest.raises(AppError) as disabled_error:
            await service.resolve(issued.raw_token, now + timedelta(seconds=1))
        assert disabled_error.value.code == "AUTH_SESSION_INVALID"
    finally:
        await engine.dispose()


def test_auth_session_lifecycle_and_token_hash_storage() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_session_lifecycle(database_url))
