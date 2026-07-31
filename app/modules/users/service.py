"""用户创建、禁用和密码重置用例。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from app.core.errors import AppError
from app.modules.auth.security import hash_password
from app.modules.users.models import User
from app.modules.users.schemas import UserRead


USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,63}$", re.ASCII)
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128


class UserRepositoryProtocol(Protocol):
    async def get_by_username(self, username: str) -> User | None: ...

    async def get_by_id(self, user_id: UUID) -> User | None: ...

    async def email_exists(
        self,
        email: str,
        *,
        exclude_user_id: UUID | None = None,
    ) -> bool: ...

    def add(self, user: User) -> None: ...

    async def set_status(self, user: User, status: str, now: datetime) -> None: ...

    async def set_password_hash(
        self,
        user: User,
        password_hash: str,
        now: datetime,
    ) -> None: ...


class SessionRevoker(Protocol):
    async def revoke_all_for_user(self, user_id: UUID, now: datetime) -> None: ...


class UserService:
    """在调用方事务内编排用户与 Session 状态变更。"""

    def __init__(
        self,
        repository: UserRepositoryProtocol,
        session_revoker: SessionRevoker | None = None,
    ) -> None:
        self._repository = repository
        self._session_revoker = session_revoker

    async def create_user(
        self,
        *,
        username: str,
        password: str,
        email: str | None = None,
        role: str = "user",
    ) -> UserRead:
        normalized_username = _normalize_username(username)
        normalized_email = _normalize_email(email)
        _validate_password(password)
        if role not in {"admin", "user"}:
            raise _input_error("role 只能是 admin 或 user。")
        if await self._repository.get_by_username(normalized_username) is not None:
            raise AppError(
                code="USER_USERNAME_CONFLICT",
                message="用户名已存在。",
                status_code=409,
            )
        if normalized_email is not None and await self._repository.email_exists(
            normalized_email
        ):
            raise AppError(
                code="USER_EMAIL_CONFLICT",
                message="邮箱已存在。",
                status_code=409,
            )

        now = datetime.now(UTC)
        user = User(
            id=uuid4(),
            username=normalized_username,
            email=normalized_email,
            password_hash=await hash_password(password),
            role=role,
            status="active",
            created_at=now,
            updated_at=now,
        )
        self._repository.add(user)
        return UserRead.model_validate(user)

    async def set_status(self, user_id: UUID, status: str, now: datetime) -> UserRead:
        if status not in {"active", "disabled"}:
            raise _input_error("status 只能是 active 或 disabled。")
        user = await self._require_user(user_id)
        await self._repository.set_status(user, status, now)
        if status == "disabled":
            await self._revoke_sessions(user_id, now)
        return UserRead.model_validate(user)

    async def reset_password(
        self,
        user_id: UUID,
        password: str,
        now: datetime,
    ) -> UserRead:
        _validate_password(password)
        user = await self._require_user(user_id)
        encoded_hash = await hash_password(password)
        await self._repository.set_password_hash(user, encoded_hash, now)
        await self._revoke_sessions(user_id, now)
        return UserRead.model_validate(user)

    async def _require_user(self, user_id: UUID) -> User:
        user = await self._repository.get_by_id(user_id)
        if user is None:
            raise AppError(
                code="USER_NOT_FOUND",
                message="用户不存在。",
                status_code=404,
            )
        return user

    async def _revoke_sessions(self, user_id: UUID, now: datetime) -> None:
        if self._session_revoker is None:
            raise RuntimeError("用户状态变更必须配置 Session 撤销器")
        await self._session_revoker.revoke_all_for_user(user_id, now)


def _normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if USERNAME_PATTERN.fullmatch(normalized) is None:
        raise _input_error(
            "username 必须为 3 至 64 位小写 ASCII 字母、数字、点、下划线或连字符。"
        )
    return normalized


def _normalize_email(email: str | None) -> str | None:
    if email is None:
        return None
    normalized = email.strip().lower()
    if not normalized or len(normalized) > 320 or "@" not in normalized:
        raise _input_error("email 格式无效。")
    return normalized


def _validate_password(password: str) -> None:
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise _input_error("password 长度必须为 12 至 128 个字符。")


def _input_error(message: str) -> AppError:
    return AppError(
        code="USER_INPUT_INVALID",
        message=message,
        status_code=422,
    )
