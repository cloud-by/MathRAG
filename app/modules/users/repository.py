"""用户持久化访问。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, exists, func, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import ColumnElement

from app.modules.users.models import User
from app.modules.users.types import UserActor, UserRole, UserStatus


class UserRepository:
    """只操作调用方注入的 Session，不拥有事务生命周期。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_username(self, username: str) -> User | None:
        return await self._session.scalar(select(User).where(User.username == username))

    async def get_by_id(
        self,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> User | None:
        statement = select(User).where(User.id == user_id)
        if for_update:
            statement = statement.with_for_update()
        return await self._session.scalar(statement)

    async def get_managed_by_id(
        self,
        actor: UserActor,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> tuple[User, str | None] | None:
        creator = aliased(User)
        statement = (
            select(User, creator.username)
            .outerjoin(creator, creator.id == User.created_by_user_id)
            .where(User.id == user_id, _visible_clause(actor))
        )
        if for_update:
            statement = statement.with_for_update(of=User)
        row = (await self._session.execute(statement)).one_or_none()
        if row is None:
            return None
        return row[0], row[1]

    async def list_managed(
        self,
        actor: UserActor,
        *,
        query: str | None,
        role: UserRole | None,
        status: UserStatus | None,
        page: int,
        page_size: int,
    ) -> tuple[list[tuple[User, str | None]], int]:
        creator = aliased(User)
        conditions: list[ColumnElement[bool]] = [_visible_clause(actor)]
        if query is not None:
            pattern = f"%{query}%"
            conditions.append(
                or_(User.username.ilike(pattern), User.email.ilike(pattern))
            )
        if role is not None:
            conditions.append(User.role == role)
        if status is not None:
            conditions.append(User.status == status)

        rows = (
            await self._session.execute(
                select(User, creator.username)
                .outerjoin(creator, creator.id == User.created_by_user_id)
                .where(*conditions)
                .order_by(User.created_at.desc(), User.id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
        ).all()
        total = await self._session.scalar(
            select(func.count()).select_from(User).where(*conditions)
        )
        return [(row[0], row[1]) for row in rows], int(total or 0)

    async def lock_active_admins(self) -> list[User]:
        result = await self._session.scalars(
            select(User)
            .where(User.role == "admin", User.status == "active")
            .order_by(User.id)
            .with_for_update()
        )
        return list(result)

    def add(self, user: User) -> None:
        self._session.add(user)

    async def flush(self) -> None:
        await self._session.flush()

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
        *,
        must_change_password: bool = True,
    ) -> None:
        user.password_hash = password_hash
        user.must_change_password = must_change_password
        user.updated_at = now


def _visible_clause(actor: UserActor) -> ColumnElement[bool]:
    if actor.role == "admin":
        return true()
    return and_(
        User.role == "student",
        User.created_by_user_id == actor.user_id,
    )
