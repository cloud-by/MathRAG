"""用户用例的事务级依赖装配。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.infrastructure.database.session import get_session_factory
from app.modules.auth.repository import AuthRepository
from app.modules.users.repository import UserRepository
from app.modules.users.service import UserService


async def get_user_service() -> AsyncIterator[UserService]:
    """为单次请求共享用户与 Session 变更事务。"""
    session_factory = get_session_factory()
    async with session_factory() as session:
        async with session.begin():
            yield UserService(UserRepository(session), AuthRepository(session))
