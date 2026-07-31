"""用户领域服务测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.core.errors import AppError
from app.modules.users.models import User
from app.modules.users.service import UserService


class FakeUserRepository:
    def __init__(self) -> None:
        self.users: dict[UUID, User] = {}

    async def get_by_username(self, username: str) -> User | None:
        return next((user for user in self.users.values() if user.username == username), None)

    async def get_by_id(self, user_id: UUID) -> User | None:
        return self.users.get(user_id)

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
    ) -> None:
        user.password_hash = password_hash
        user.updated_at = now


class FakeSessionRevoker:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, datetime]] = []

    async def revoke_all_for_user(self, user_id: UUID, now: datetime) -> None:
        self.calls.append((user_id, now))


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
