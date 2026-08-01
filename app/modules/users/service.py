"""用户创建、禁用和密码重置用例。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from app.core.errors import AppError
from app.modules.auth.security import hash_password, verify_password
from app.modules.users.models import User
from app.modules.users.schemas import (
    ManagedUserRead,
    UserCreate,
    UserPage,
    UserRead,
    UserUpdate,
)
from app.modules.users.types import (
    USER_ROLES,
    UserActor,
    UserRole,
    UserStatus,
)


USERNAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,63}$", re.ASCII)
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128


class UserRepositoryProtocol(Protocol):
    async def get_by_username(self, username: str) -> User | None: ...

    async def get_by_id(
        self,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> User | None: ...

    async def get_managed_by_id(
        self,
        actor: UserActor,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> tuple[User, str | None] | None: ...

    async def list_managed(
        self,
        actor: UserActor,
        *,
        query: str | None,
        role: UserRole | None,
        status: UserStatus | None,
        page: int,
        page_size: int,
    ) -> tuple[list[tuple[User, str | None]], int]: ...

    async def lock_active_admins(self) -> list[User]: ...

    async def email_exists(
        self,
        email: str,
        *,
        exclude_user_id: UUID | None = None,
    ) -> bool: ...

    def add(self, user: User) -> None: ...

    async def flush(self) -> None: ...

    async def set_status(self, user: User, status: str, now: datetime) -> None: ...

    async def set_password_hash(
        self,
        user: User,
        password_hash: str,
        now: datetime,
        *,
        must_change_password: bool = True,
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
        role: UserRole = "student",
        created_by_user_id: UUID | None = None,
        must_change_password: bool = True,
    ) -> UserRead:
        user = await self._create(
            username=username,
            password=password,
            email=email,
            role=role,
            created_by_user_id=created_by_user_id,
            must_change_password=must_change_password,
        )
        return UserRead.model_validate(user)

    async def create_managed_user(
        self,
        actor: UserActor,
        request: UserCreate,
    ) -> ManagedUserRead:
        _require_manager(actor)
        if actor.role == "teacher" and request.role != "student":
            raise _error(
                "USER_ROLE_FORBIDDEN",
                "教师只能创建学生账号。",
                403,
            )
        user = await self._create(
            username=request.username,
            password=request.password,
            email=request.email,
            role=request.role,
            created_by_user_id=actor.user_id,
            must_change_password=True,
        )
        row = await self._repository.get_managed_by_id(actor, user.id)
        if row is None:
            raise RuntimeError("新建账号未出现在创建者的管理范围内")
        return _to_managed_read(*row)

    async def list_managed_users(
        self,
        actor: UserActor,
        *,
        query: str | None,
        role: UserRole | None,
        status: UserStatus | None,
        page: int,
        page_size: int,
    ) -> UserPage:
        _require_manager(actor)
        if page < 1 or not 1 <= page_size <= 100:
            raise _input_error("分页参数无效。")
        normalized_query = query.strip().lower() if query is not None else None
        if normalized_query == "":
            normalized_query = None
        rows, total = await self._repository.list_managed(
            actor,
            query=normalized_query,
            role=role,
            status=status,
            page=page,
            page_size=page_size,
        )
        return UserPage(
            items=[_to_managed_read(*row) for row in rows],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def get_managed_user(
        self,
        actor: UserActor,
        user_id: UUID,
    ) -> ManagedUserRead:
        _require_manager(actor)
        row = await self._require_managed_user(actor, user_id, for_update=False)
        return _to_managed_read(*row)

    async def update_managed_user(
        self,
        actor: UserActor,
        user_id: UUID,
        request: UserUpdate,
        now: datetime,
    ) -> ManagedUserRead:
        _require_manager(actor)
        fields = request.model_fields_set
        active_admins: list[User] = []
        if actor.role == "admin" and fields & {"role", "status"}:
            # 所有管理员转换先按固定 UUID 顺序锁定，避免并发保护失效。
            active_admins = await self._repository.lock_active_admins()
        row = await self._require_managed_user(actor, user_id, for_update=True)
        target, creator_username = row

        if actor.role == "teacher" and "role" in fields:
            raise _error(
                "USER_ROLE_FORBIDDEN",
                "教师不能修改学生角色。",
                403,
            )

        next_role = _updated_role(request, fields, target)
        next_status = _updated_status(request, fields, target)
        if actor.role == "admin":
            _guard_admin_transition(
                actor,
                target,
                next_role,
                next_status,
                active_admins,
            )

        next_username = target.username
        if "username" in fields:
            if request.username is None:
                raise _input_error("username 不能为空。")
            next_username = _normalize_username(request.username)
            existing = await self._repository.get_by_username(next_username)
            if existing is not None and existing.id != target.id:
                raise _error(
                    "USER_USERNAME_CONFLICT",
                    "用户名已存在。",
                    409,
                )

        next_email = target.email
        if "email" in fields:
            next_email = _normalize_email(request.email)
            if next_email is not None and await self._repository.email_exists(
                next_email,
                exclude_user_id=target.id,
            ):
                raise _error("USER_EMAIL_CONFLICT", "邮箱已存在。", 409)

        old_role = target.role
        old_status = target.status
        changed = (
            next_username != target.username
            or next_email != target.email
            or next_role != target.role
            or next_status != target.status
        )
        target.username = next_username
        target.email = next_email
        target.role = next_role
        target.status = next_status
        if changed:
            target.updated_at = now
            await self._flush_with_conflict_mapping()

        if old_role != next_role or (
            old_status != next_status and next_status == "disabled"
        ):
            await self._revoke_sessions(target.id, now)

        return _to_managed_read(target, creator_username)

    async def reset_managed_password(
        self,
        actor: UserActor,
        user_id: UUID,
        password: str,
        now: datetime,
    ) -> None:
        _require_manager(actor)
        _validate_password(password)
        target, _ = await self._require_managed_user(actor, user_id, for_update=True)
        await self._repository.set_password_hash(
            target,
            await hash_password(password),
            now,
            must_change_password=True,
        )
        await self._repository.flush()
        await self._revoke_sessions(target.id, now)

    async def change_own_password(
        self,
        user_id: UUID,
        current_password: str,
        new_password: str,
        now: datetime,
    ) -> None:
        _validate_password(new_password)
        user = await self._repository.get_by_id(user_id, for_update=True)
        if user is None or not await verify_password(
            current_password,
            user.password_hash,
        ):
            raise _error(
                "AUTH_CURRENT_PASSWORD_INVALID",
                "当前密码不正确。",
                422,
            )
        if await verify_password(new_password, user.password_hash):
            raise _input_error("新密码不能与当前密码相同。")
        await self._repository.set_password_hash(
            user,
            await hash_password(new_password),
            now,
            must_change_password=False,
        )
        await self._repository.flush()
        await self._revoke_sessions(user_id, now)

    async def _create(
        self,
        *,
        username: str,
        password: str,
        email: str | None,
        role: UserRole,
        created_by_user_id: UUID | None,
        must_change_password: bool,
    ) -> User:
        normalized_username = _normalize_username(username)
        normalized_email = _normalize_email(email)
        _validate_password(password)
        if role not in USER_ROLES:
            raise _input_error("role 只能是 student、teacher 或 admin。")
        if await self._repository.get_by_username(normalized_username) is not None:
            raise _error("USER_USERNAME_CONFLICT", "用户名已存在。", 409)
        if normalized_email is not None and await self._repository.email_exists(
            normalized_email
        ):
            raise _error("USER_EMAIL_CONFLICT", "邮箱已存在。", 409)

        now = datetime.now(UTC)
        user = User(
            id=uuid4(),
            username=normalized_username,
            email=normalized_email,
            password_hash=await hash_password(password),
            role=role,
            status="active",
            created_by_user_id=created_by_user_id,
            must_change_password=must_change_password,
            created_at=now,
            updated_at=now,
        )
        self._repository.add(user)
        await self._flush_with_conflict_mapping()
        return user

    async def set_status(self, user_id: UUID, status: str, now: datetime) -> UserRead:
        if status not in {"active", "disabled"}:
            raise _input_error("status 只能是 active 或 disabled。")
        user = await self._require_user(user_id)
        old_status = user.status
        await self._repository.set_status(user, status, now)
        if old_status != status and status == "disabled":
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
        user = await self._repository.get_by_id(user_id, for_update=True)
        if user is None:
            raise _error("USER_NOT_FOUND", "用户不存在。", 404)
        return user

    async def _require_managed_user(
        self,
        actor: UserActor,
        user_id: UUID,
        *,
        for_update: bool,
    ) -> tuple[User, str | None]:
        row = await self._repository.get_managed_by_id(
            actor,
            user_id,
            for_update=for_update,
        )
        if row is None:
            raise _error("USER_NOT_FOUND", "用户不存在。", 404)
        return row

    async def _flush_with_conflict_mapping(self) -> None:
        try:
            await self._repository.flush()
        except IntegrityError as error:
            constraint_name = _constraint_name(error)
            if constraint_name == "uq_users_username":
                raise _error("USER_USERNAME_CONFLICT", "用户名已存在。", 409) from error
            if constraint_name == "uq_users_email":
                raise _error("USER_EMAIL_CONFLICT", "邮箱已存在。", 409) from error
            raise

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


def _error(code: str, message: str, status_code: int) -> AppError:
    return AppError(code=code, message=message, status_code=status_code)


def _require_manager(actor: UserActor) -> None:
    if actor.role not in {"teacher", "admin"}:
        raise _error("AUTH_FORBIDDEN", "权限不足。", 403)


def _to_managed_read(user: User, creator_username: str | None) -> ManagedUserRead:
    return ManagedUserRead(
        **UserRead.model_validate(user).model_dump(),
        created_by_username=creator_username,
    )


def _updated_role(
    request: UserUpdate,
    fields: set[str],
    target: User,
) -> UserRole:
    if "role" not in fields:
        return cast(UserRole, target.role)
    if request.role is None:
        raise _input_error("role 不能为空。")
    return request.role


def _updated_status(
    request: UserUpdate,
    fields: set[str],
    target: User,
) -> UserStatus:
    if "status" not in fields:
        return cast(UserStatus, target.status)
    if request.status is None:
        raise _input_error("status 不能为空。")
    return request.status


def _guard_admin_transition(
    actor: UserActor,
    target: User,
    next_role: UserRole,
    next_status: UserStatus,
    active_admins: list[User],
) -> None:
    removes_active_admin = (
        target.role == "admin"
        and target.status == "active"
        and (next_role != "admin" or next_status != "active")
    )
    if not removes_active_admin:
        return
    if actor.user_id == target.id:
        raise _error(
            "USER_SELF_PROTECTED",
            "不能降级或禁用当前管理员。",
            409,
        )
    if len(active_admins) == 1:
        raise _error(
            "USER_LAST_ADMIN_PROTECTED",
            "必须保留一个启用的管理员。",
            409,
        )


def _constraint_name(error: IntegrityError) -> str | None:
    """沿驱动异常链读取约束名，不解析可能含敏感值的异常文本。"""
    current: BaseException | None = error.orig
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        direct = getattr(current, "constraint_name", None)
        if isinstance(direct, str):
            return direct
        diagnostic = getattr(current, "diag", None)
        name = getattr(diagnostic, "constraint_name", None)
        if isinstance(name, str):
            return name
        current = current.__cause__ or current.__context__
    return None
