"""登录、身份解析和退出用例。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.modules.auth.models import UserSession
from app.modules.auth.repository import AuthRepository, LoginUserRecord
from app.modules.auth.security import (
    generate_session_token,
    hash_session_token,
    issue_csrf_token,
    verify_password,
)
from app.modules.users.schemas import UserRead
from app.modules.users.types import UserRole


DUMMY_PASSWORD_HASH = (
    "$argon2id$v=19$m=65536,t=3,p=4$TNovdvPJDogVvNjr0m0sBQ$"
    "n/QK+wuqCYeXrw8nmUagrxpdWlx40N4gaQhiRJldoiI"
)
LAST_SEEN_WRITE_INTERVAL = timedelta(minutes=5)


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: UUID
    session_id: UUID
    username: str
    role: UserRole
    must_change_password: bool
    session_token_hash: bytes


@dataclass(frozen=True)
class IssuedSession:
    session_id: UUID
    user: UserRead
    raw_token: str
    csrf_token: str
    expires_at: datetime


class AuthService:
    """使用短 Session 完成认证，不向调用方暴露 ORM。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        repository_factory: Callable[[AsyncSession], AuthRepository] = AuthRepository,
        session_ttl_seconds: int,
        csrf_secret: str,
    ) -> None:
        self._session_factory = session_factory
        self._repository_factory = repository_factory
        self._session_ttl_seconds = session_ttl_seconds
        self._csrf_secret = csrf_secret

    async def login(self, username: str, password: str, now: datetime) -> IssuedSession:
        normalized_username = username.strip().lower()
        async with self._session_factory() as session:
            login_user = await self._repository_factory(session).find_login_user(
                normalized_username
            )

        encoded_hash = login_user.password_hash if login_user is not None else DUMMY_PASSWORD_HASH
        password_matches = await verify_password(password, encoded_hash)
        if login_user is None or not password_matches:
            raise _invalid_credentials()
        if login_user.status != "active":
            raise _invalid_session()

        raw_token = generate_session_token()
        token_hash = hash_session_token(raw_token)
        session_id = uuid4()
        expires_at = now + timedelta(seconds=self._session_ttl_seconds)
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                current_user = await repository.find_login_user_by_id(login_user.id)
                if current_user is None or current_user.status != "active":
                    raise _invalid_session()
                repository.add_session(
                    UserSession(
                        id=session_id,
                        user_id=current_user.id,
                        token_hash=token_hash,
                        expires_at=expires_at,
                        created_at=now,
                        last_seen_at=now,
                    )
                )

        return IssuedSession(
            session_id=session_id,
            user=_to_user_read(current_user),
            raw_token=raw_token,
            csrf_token=issue_csrf_token(token_hash, self._csrf_secret),
            expires_at=expires_at,
        )

    async def resolve(self, raw_token: str, now: datetime) -> AuthenticatedPrincipal:
        token_hash = hash_session_token(raw_token)
        async with self._session_factory() as session:
            async with session.begin():
                repository = self._repository_factory(session)
                active = await repository.find_active_by_hash(token_hash, now)
                if active is None:
                    raise _invalid_session()
                if active.last_seen_at <= now - LAST_SEEN_WRITE_INTERVAL:
                    await repository.touch_last_seen(active.session_id, now)
                principal = AuthenticatedPrincipal(
                    user_id=active.user_id,
                    session_id=active.session_id,
                    username=active.username,
                    role=active.role,
                    must_change_password=active.must_change_password,
                    session_token_hash=active.token_hash,
                )
        return principal

    async def get_user(self, user_id: UUID) -> UserRead:
        async with self._session_factory() as session:
            user = await self._repository_factory(session).find_login_user_by_id(user_id)
        if user is None or user.status != "active":
            raise _invalid_session()
        return _to_user_read(user)

    async def resolve_for_logout(self, raw_token: str) -> AuthenticatedPrincipal:
        token_hash = hash_session_token(raw_token)
        async with self._session_factory() as session:
            record = await self._repository_factory(session).find_by_hash(token_hash)
        if record is None:
            raise _invalid_session()
        return AuthenticatedPrincipal(
            user_id=record.user_id,
            session_id=record.session_id,
            username=record.username,
            role=record.role,
            must_change_password=record.must_change_password,
            session_token_hash=record.token_hash,
        )

    async def logout(self, session_id: UUID, now: datetime) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                await self._repository_factory(session).revoke(session_id, now)


def _to_user_read(user: LoginUserRecord) -> UserRead:
    return UserRead(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        status=user.status,
        created_by_user_id=user.created_by_user_id,
        must_change_password=user.must_change_password,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


def _invalid_credentials() -> AppError:
    return AppError(
        code="AUTH_INVALID_CREDENTIALS",
        message="用户名或密码错误。",
        status_code=401,
    )


def _invalid_session() -> AppError:
    return AppError(
        code="AUTH_SESSION_INVALID",
        message="登录状态无效或已过期。",
        status_code=401,
    )
