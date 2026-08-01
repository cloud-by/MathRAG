"""用户领域服务测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.errors import AppError
from app.modules.users.models import User
from app.modules.users.schemas import UserCreate, UserUpdate
from app.modules.users.service import UserService
from app.modules.users.types import UserActor


NOW = datetime(2026, 8, 1, tzinfo=UTC)


class FakeUserRepository:
    def __init__(self) -> None:
        self.users: dict[UUID, User] = {}
        self.flush_error: BaseException | None = None

    async def get_by_username(self, username: str) -> User | None:
        return next((user for user in self.users.values() if user.username == username), None)

    async def get_by_id(
        self,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> User | None:
        return self.users.get(user_id)

    async def get_managed_by_id(
        self,
        actor: UserActor,
        user_id: UUID,
        *,
        for_update: bool = False,
    ) -> tuple[User, str | None] | None:
        user = self.users.get(user_id)
        if user is None or not self._visible(actor, user):
            return None
        creator = self.users.get(user.created_by_user_id)
        return user, creator.username if creator is not None else None

    async def list_managed(
        self,
        actor: UserActor,
        *,
        query: str | None,
        role: str | None,
        status: str | None,
        page: int,
        page_size: int,
    ) -> tuple[list[tuple[User, str | None]], int]:
        users = [user for user in self.users.values() if self._visible(actor, user)]
        if query is not None:
            normalized = query.lower()
            users = [
                user
                for user in users
                if normalized in user.username.lower()
                or (user.email is not None and normalized in user.email.lower())
            ]
        if role is not None:
            users = [user for user in users if user.role == role]
        if status is not None:
            users = [user for user in users if user.status == status]
        users.sort(key=lambda user: (user.created_at, user.id), reverse=True)
        total = len(users)
        start = (page - 1) * page_size
        rows = []
        for user in users[start : start + page_size]:
            creator = self.users.get(user.created_by_user_id)
            rows.append((user, creator.username if creator is not None else None))
        return rows, total

    async def lock_active_admins(self) -> list[User]:
        return sorted(
            (
                user
                for user in self.users.values()
                if user.role == "admin" and user.status == "active"
            ),
            key=lambda user: user.id,
        )

    async def email_exists(self, email: str, *, exclude_user_id: UUID | None = None) -> bool:
        return any(
            user.email == email and user.id != exclude_user_id
            for user in self.users.values()
        )

    def add(self, user: User) -> None:
        self.users[user.id] = user

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

    async def flush(self) -> None:
        if self.flush_error is not None:
            raise self.flush_error

    @staticmethod
    def _visible(actor: UserActor, user: User) -> bool:
        return actor.role == "admin" or (
            actor.role == "teacher"
            and user.role == "student"
            and user.created_by_user_id == actor.user_id
        )


class FakeSessionRevoker:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, datetime]] = []

    async def revoke_all_for_user(self, user_id: UUID, now: datetime) -> None:
        self.calls.append((user_id, now))


class ConstraintViolation(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__(constraint_name)
        self.constraint_name = constraint_name


def test_create_user_normalizes_username_and_exposes_no_hash() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())

    created = asyncio.run(
        service.create_user(
            username="  Math.Admin  ",
            password="very-private-password",
            email=" Admin@Example.Local ",
            role="admin",
        )
    )

    assert created.username == "math.admin"
    assert created.email == "admin@example.local"
    assert created.role == "admin"
    assert "password" not in created.model_dump()
    assert "password_hash" not in created.model_dump()
    persisted = repository.users[created.id]
    assert persisted.password_hash != "very-private-password"


@pytest.mark.parametrize(
    "username",
    ["ab", "a" * 65, "空用户", "bad space", "-leading"],
)
def test_create_user_rejects_invalid_normalized_username(username: str) -> None:
    service = UserService(FakeUserRepository(), FakeSessionRevoker())

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.create_user(username=username, password="p" * 12))

    assert exc_info.value.code == "USER_INPUT_INVALID"


@pytest.mark.parametrize("password", ["short", "p" * 129])
def test_create_user_rejects_password_outside_character_limits(password: str) -> None:
    service = UserService(FakeUserRepository(), FakeSessionRevoker())

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.create_user(username="valid-user", password=password))

    assert exc_info.value.code == "USER_INPUT_INVALID"
    assert password not in str(exc_info.value)


def test_duplicate_username_and_email_use_stable_conflicts() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    asyncio.run(
        service.create_user(
            username="first-user",
            password="p" * 12,
            email="first@example.local",
        )
    )

    with pytest.raises(AppError) as username_error:
        asyncio.run(service.create_user(username=" FIRST-USER ", password="q" * 12))
    with pytest.raises(AppError) as email_error:
        asyncio.run(
            service.create_user(
                username="second-user",
                password="q" * 12,
                email="FIRST@example.local",
            )
        )

    assert username_error.value.code == "USER_USERNAME_CONFLICT"
    assert username_error.value.status_code == 409
    assert email_error.value.code == "USER_EMAIL_CONFLICT"
    assert email_error.value.status_code == 409


def test_database_unique_race_uses_stable_conflict() -> None:
    repository = FakeUserRepository()
    driver_wrapper = RuntimeError("driver wrapper")
    driver_wrapper.__cause__ = ConstraintViolation("uq_users_username")
    repository.flush_error = IntegrityError("INSERT", {}, driver_wrapper)
    service = UserService(repository, FakeSessionRevoker())

    with pytest.raises(AppError) as exc_info:
        asyncio.run(
            service.create_user(
                username="racing-user",
                password="temporary-123",
            )
        )

    assert exc_info.value.code == "USER_USERNAME_CONFLICT"
    assert exc_info.value.status_code == 409


def test_disable_and_password_reset_revoke_all_sessions() -> None:
    repository = FakeUserRepository()
    revoker = FakeSessionRevoker()
    service = UserService(repository, revoker)
    created = asyncio.run(service.create_user(username="managed-user", password="p" * 12))
    now = datetime(2026, 7, 31, tzinfo=UTC)

    asyncio.run(service.set_status(created.id, "disabled", now))
    asyncio.run(service.reset_password(created.id, "new-password-123", now))

    assert revoker.calls == [(created.id, now), (created.id, now)]
    assert repository.users[created.id].status == "disabled"
    assert repository.users[created.id].password_hash != "new-password-123"


def add_user(
    repository: FakeUserRepository,
    *,
    role: str = "student",
    status: str = "active",
    created_by_user_id: UUID | None = None,
    username: str | None = None,
) -> User:
    user = User(
        id=uuid4(),
        username=username or f"user-{len(repository.users)}",
        email=None,
        password_hash="argon2-placeholder",
        role=role,
        status=status,
        created_by_user_id=created_by_user_id,
        must_change_password=False,
        created_at=NOW,
        updated_at=NOW,
    )
    repository.add(user)
    return user


def test_teacher_manages_only_students_created_by_self() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    teacher = add_user(repository, role="teacher")
    own = add_user(repository, created_by_user_id=teacher.id)
    other = add_user(repository, created_by_user_id=uuid4())
    actor = UserActor(teacher.id, "teacher")

    updated = asyncio.run(
        service.update_managed_user(
            actor,
            own.id,
            UserUpdate(email="new@example.local"),
            NOW,
        )
    )

    assert updated.email == "new@example.local"
    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.get_managed_user(actor, other.id))
    assert exc_info.value.code == "USER_NOT_FOUND"
    assert exc_info.value.status_code == 404


def test_teacher_list_contains_only_owned_students() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    teacher = add_user(repository, role="teacher")
    own = add_user(repository, created_by_user_id=teacher.id)
    add_user(repository, created_by_user_id=uuid4())
    add_user(repository, role="teacher", created_by_user_id=teacher.id)

    page = asyncio.run(
        service.list_managed_users(
            UserActor(teacher.id, "teacher"),
            query=None,
            role=None,
            status=None,
            page=1,
            page_size=20,
        )
    )

    assert page.total == 1
    assert [item.id for item in page.items] == [own.id]


@pytest.mark.parametrize("role", ["student", "teacher", "admin"])
def test_admin_can_create_all_roles_with_temporary_password(role: str) -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    admin = add_user(repository, role="admin")

    created = asyncio.run(
        service.create_managed_user(
            UserActor(admin.id, "admin"),
            UserCreate(
                username=f"created-{role}",
                password="temporary-123",
                role=role,
            ),
        )
    )

    assert created.role == role
    assert created.created_by_user_id == admin.id
    assert created.created_by_username == admin.username
    assert created.must_change_password is True


def test_teacher_can_only_create_students_and_cannot_update_role() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    teacher = add_user(repository, role="teacher")
    student = add_user(repository, created_by_user_id=teacher.id)
    actor = UserActor(teacher.id, "teacher")

    with pytest.raises(AppError) as create_error:
        asyncio.run(
            service.create_managed_user(
                actor,
                UserCreate(
                    username="forbidden-teacher",
                    password="temporary-123",
                    role="teacher",
                ),
            )
        )
    with pytest.raises(AppError) as update_error:
        asyncio.run(
            service.update_managed_user(
                actor,
                student.id,
                UserUpdate(role="student"),
                NOW,
            )
        )

    assert create_error.value.code == "USER_ROLE_FORBIDDEN"
    assert update_error.value.code == "USER_ROLE_FORBIDDEN"


def test_admin_cannot_disable_or_demote_self() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    admin = add_user(repository, role="admin")
    actor = UserActor(admin.id, "admin")

    for request in (UserUpdate(status="disabled"), UserUpdate(role="teacher")):
        with pytest.raises(AppError) as exc_info:
            asyncio.run(service.update_managed_user(actor, admin.id, request, NOW))
        assert exc_info.value.code == "USER_SELF_PROTECTED"


def test_last_active_admin_cannot_be_disabled() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    actor = add_user(repository, role="admin")
    target = add_user(repository, role="admin")
    actor.status = "disabled"

    with pytest.raises(AppError) as exc_info:
        asyncio.run(
            service.update_managed_user(
                UserActor(actor.id, "admin"),
                target.id,
                UserUpdate(status="disabled"),
                NOW,
            )
        )

    assert exc_info.value.code == "USER_LAST_ADMIN_PROTECTED"


def test_actual_disable_role_change_and_password_reset_revoke_sessions() -> None:
    repository = FakeUserRepository()
    revoker = FakeSessionRevoker()
    service = UserService(repository, revoker)
    admin = add_user(repository, role="admin")
    target = add_user(repository, role="student")
    actor = UserActor(admin.id, "admin")

    asyncio.run(
        service.update_managed_user(
            actor,
            target.id,
            UserUpdate(status="active"),
            NOW,
        )
    )
    asyncio.run(
        service.update_managed_user(
            actor,
            target.id,
            UserUpdate(role="teacher"),
            NOW,
        )
    )
    asyncio.run(
        service.update_managed_user(
            actor,
            target.id,
            UserUpdate(status="disabled"),
            NOW,
        )
    )
    asyncio.run(
        service.reset_managed_password(
            actor,
            target.id,
            "replacement-123",
            NOW,
        )
    )

    assert revoker.calls == [(target.id, NOW), (target.id, NOW), (target.id, NOW)]
    assert target.must_change_password is True
