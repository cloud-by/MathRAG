"""用户持久化访问。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.users.models import User


class UserRepository:
    """只操作调用方注入的 Session，不拥有事务生命周期。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_username(self, username: str) -> User | None:
        return await self._session.scalar(select(User).where(User.username == username))

    async def get_by_id(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)

    def add(self, user: User) -> None:
        self._session.add(user)

    async def email_exists(
        self,
        email: str,
        *,
        exclude_user_id: UUID | None = None,
    ) -> bool:
        statement = select(exists().where(User.email == email))
        if exclude_user_id is not None:
            statement = select(
                exists().where(User.email == email, User.id != exclude_user_id)
            )
        return bool(await self._session.scalar(statement))

    async def set_status(self, user: User, status: str, now: datetime) -> None:
        user.status = status
        user.updated_at = now

    async def set_password_hash(
        self,
        user: User,
        password_hash: str,
        now: datetime,
    ) -> None:
        user.password_hash = password_hash
        user.updated_at = now
