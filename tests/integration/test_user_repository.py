"""用户 Repository 的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import ast
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.modules.auth.models import UserSession
from app.modules.conversations.models import Conversation
from app.modules.users.models import User
from app.modules.users.repository import UserRepository
from app.modules.users.types import UserActor
from tests.integration.database_safety import require_test_database_url


REPOSITORY_PATH = Path(__file__).resolve().parents[2] / "app" / "modules" / "users" / "repository.py"


async def exercise_repository(database_url: str) -> None:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(delete(UserSession))
                await session.execute(delete(Conversation))
                await session.execute(delete(User))

                repository = UserRepository(session)
                user = User(
                    username="repository-user",
                    email="repository@example.local",
                    password_hash="argon2-placeholder",
                )
                repository.add(user)
                await session.flush()

                assert await repository.get_by_username("repository-user") is user
                assert await repository.get_by_id(user.id) is user
                assert await repository.email_exists("repository@example.local") is True
                assert await repository.email_exists(
                    "repository@example.local",
                    exclude_user_id=user.id,
                ) is False

                now = datetime(2026, 7, 31, tzinfo=UTC)
                await repository.set_status(user, "disabled", now)
                await repository.set_password_hash(user, "new-hash", now)
                assert user.status == "disabled"
                assert user.password_hash == "new-hash"
                assert user.updated_at == now

                teacher_a = User(
                    username="teacher-a",
                    password_hash="argon2-placeholder",
                    role="teacher",
                )
                teacher_b = User(
                    username="teacher-b",
                    password_hash="argon2-placeholder",
                    role="teacher",
                )
                admin = User(
                    username="repository-admin",
                    password_hash="argon2-placeholder",
                    role="admin",
                )
                repository.add(teacher_a)
                repository.add(teacher_b)
                repository.add(admin)
                await session.flush()
                student_a = User(
                    username="student-a",
                    password_hash="argon2-placeholder",
                    created_by_user_id=teacher_a.id,
                )
                student_b = User(
                    username="student-b",
                    password_hash="argon2-placeholder",
                    created_by_user_id=teacher_b.id,
                )
                repository.add(student_a)
                repository.add(student_b)
                await session.flush()

                teacher_rows, teacher_total = await repository.list_managed(
                    UserActor(teacher_a.id, "teacher"),
                    query=None,
                    role=None,
                    status=None,
                    page=1,
                    page_size=20,
                )
                assert teacher_total == 1
                assert [row[0].id for row in teacher_rows] == [student_a.id]
                assert teacher_rows[0][1] == "teacher-a"

                admin_rows, admin_total = await repository.list_managed(
                    UserActor(admin.id, "admin"),
                    query=None,
                    role=None,
                    status=None,
                    page=1,
                    page_size=20,
                )
                assert admin_total == 6
                assert {row[0].id for row in admin_rows} == {
                    user.id,
                    teacher_a.id,
                    teacher_b.id,
                    student_a.id,
                    student_b.id,
                    admin.id,
                }

                filtered_rows, filtered_total = await repository.list_managed(
                    UserActor(admin.id, "admin"),
                    query="student",
                    role="student",
                    status="active",
                    page=2,
                    page_size=1,
                )
                assert filtered_total == 2
                assert len(filtered_rows) == 1
                assert filtered_rows[0][0].role == "student"

                hidden = await repository.get_managed_by_id(
                    UserActor(teacher_a.id, "teacher"),
                    student_b.id,
                    for_update=True,
                )
                assert hidden is None

                active_admins = await repository.lock_active_admins()
                assert [row.id for row in active_admins] == [admin.id]
    finally:
        await engine.dispose()


def test_user_repository_reads_and_writes_without_owning_transactions() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_repository(database_url))


def test_user_repository_does_not_control_session_lifecycle() -> None:
    tree = ast.parse(REPOSITORY_PATH.read_text(encoding="utf-8"))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_session"
        and node.func.attr in {"begin", "commit", "rollback", "close"}
    }

    assert forbidden == set()
