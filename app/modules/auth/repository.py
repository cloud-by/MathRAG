"""服务端 Session 与认证查询的持久化访问。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.auth.models import UserSession
from app.modules.users.models import User
from app.modules.users.types import UserRole, UserStatus


@dataclass(frozen=True)
class LoginUserRecord:
    id: UUID
    username: str
    email: str | None
    password_hash: str
    role: UserRole
    status: UserStatus
    created_by_user_id: UUID | None
    must_change_password: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ActiveSessionRecord:
    session_id: UUID
    user_id: UUID
    username: str
    role: UserRole
    must_change_password: bool
    token_hash: bytes
    last_seen_at: datetime


class AuthRepository:
    """只操作注入 Session，不控制事务或连接生命周期。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_login_user(self, username: str) -> LoginUserRecord | None:
        user = await self._session.scalar(select(User).where(User.username == username))
        return _to_login_record(user)

    async def find_login_user_by_id(self, user_id: UUID) -> LoginUserRecord | None:
        user = await self._session.get(User, user_id)
        return _to_login_record(user)

    def add_session(self, user_session: UserSession) -> None:
        self._session.add(user_session)

    async def find_active_by_hash(
        self,
        token_hash: bytes,
        now: datetime,
    ) -> ActiveSessionRecord | None:
        statement = (
            select(UserSession, User)
            .join(User, User.id == UserSession.user_id)
            .where(
                UserSession.token_hash == token_hash,
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
                User.status == "active",
            )
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        user_session, user = row
        return ActiveSessionRecord(
            session_id=user_session.id,
            user_id=user.id,
            username=user.username,
            role=user.role,  # type: ignore[arg-type]
            must_change_password=user.must_change_password,
            token_hash=user_session.token_hash,
            last_seen_at=user_session.last_seen_at,
        )

    async def find_by_hash(self, token_hash: bytes) -> ActiveSessionRecord | None:
        """读取任意状态的 Session，仅供幂等退出验证令牌归属。"""
        statement = (
            select(UserSession, User)
            .join(User, User.id == UserSession.user_id)
            .where(UserSession.token_hash == token_hash)
        )
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        user_session, user = row
        return ActiveSessionRecord(
            session_id=user_session.id,
            user_id=user.id,
            username=user.username,
            role=user.role,  # type: ignore[arg-type]
            must_change_password=user.must_change_password,
            token_hash=user_session.token_hash,
            last_seen_at=user_session.last_seen_at,
        )

    async def touch_last_seen(self, session_id: UUID, now: datetime) -> None:
        await self._session.execute(
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(last_seen_at=now)
        )

    async def revoke(self, session_id: UUID, now: datetime) -> None:
        await self._session.execute(
            update(UserSession)
            .where(UserSession.id == session_id, UserSession.revoked_at.is_(None))
            # 应用与 PostgreSQL 可能有毫秒级时钟偏差，不能写早于创建时间的值。
            .values(revoked_at=func.greatest(now, UserSession.created_at))
        )

    async def revoke_all_for_user(self, user_id: UUID, now: datetime) -> None:
        await self._session.execute(
            update(UserSession)
            .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
            .values(revoked_at=func.greatest(now, UserSession.created_at))
        )


def _to_login_record(user: User | None) -> LoginUserRecord | None:
    if user is None:
        return None
    return LoginUserRecord(
        id=user.id,
        username=user.username,
        email=user.email,
        password_hash=user.password_hash,
        role=user.role,  # type: ignore[arg-type]
        status=user.status,  # type: ignore[arg-type]
        created_by_user_id=user.created_by_user_id,
        must_change_password=user.must_change_password,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )
