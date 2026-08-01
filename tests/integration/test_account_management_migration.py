"""账号管理迁移的数据与模式契约测试。"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import pytest

from tests.integration.database_safety import require_test_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
M5_REVISION = "0005_create_documents_ingestion_jobs"
M6_REVISION = "0006_add_account_management"


def run_alembic(
    database_url: str,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", *args],
        cwd=PROJECT_ROOT,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def require_database_url() -> str:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    return require_test_database_url(database_url, os.getenv("DATABASE_URL"))


def asyncpg_url(database_url: str) -> str:
    return database_url.replace("postgresql+asyncpg://", "postgresql://")


async def seed_legacy_users(database_url: str) -> tuple[UUID, UUID]:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    student_id = uuid4()
    admin_id = uuid4()
    suffix = uuid4().hex[:12]
    try:
        await connection.executemany(
            """
            INSERT INTO users (id, username, password_hash, role)
            VALUES ($1, $2, 'argon2-test-hash', $3)
            """,
            [
                (student_id, f"legacy-student-{suffix}", "user"),
                (admin_id, f"legacy-admin-{suffix}", "admin"),
            ],
        )
        return student_id, admin_id
    finally:
        await connection.close()


async def fetch_contract(
    database_url: str,
) -> tuple[
    dict[UUID, dict[str, object]],
    dict[str, dict[str, str]],
    dict[str, object],
    set[str],
]:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    try:
        rows = {
            row["id"]: dict(row)
            for row in await connection.fetch(
                """
                SELECT id, role, created_by_user_id, must_change_password
                FROM users
                """
            )
        }
        columns = {
            row["column_name"]: dict(row)
            for row in await connection.fetch(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'users'
                  AND column_name = ANY($1::text[])
                """,
                ["role", "created_by_user_id", "must_change_password"],
            )
        }
        constraint_rows = await connection.fetch(
            """
            SELECT constraint_record.conname,
                   constraint_record.contype::text AS contype,
                   pg_get_constraintdef(constraint_record.oid) AS definition
            FROM pg_constraint AS constraint_record
            JOIN pg_class AS relation
              ON relation.oid = constraint_record.conrelid
            JOIN pg_namespace AS namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'users'
              AND constraint_record.conname = ANY($1::text[])
            """,
            ["ck_users_role", "fk_users_created_by_user_id_users"],
        )
        constraints: dict[str, object] = {}
        for row in constraint_rows:
            if row["contype"] == "c":
                constraints[row["conname"]] = set(
                    re.findall(r"'([^']+)'", row["definition"])
                )
            else:
                constraints[row["conname"]] = (
                    "ON DELETE SET NULL"
                    if "ON DELETE SET NULL" in row["definition"]
                    else row["definition"]
                )
        indexes = {
            row["indexname"]
            for row in await connection.fetch(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'users'
                """
            )
        }
        return rows, columns, constraints, indexes
    finally:
        await connection.close()


async def set_creator_and_delete(
    database_url: str,
    creator_id: UUID,
    created_user_id: UUID,
) -> UUID | None:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    try:
        await connection.execute(
            "UPDATE users SET created_by_user_id = $1 WHERE id = $2",
            creator_id,
            created_user_id,
        )
        await connection.execute("DELETE FROM users WHERE id = $1", creator_id)
        return await connection.fetchval(
            "SELECT created_by_user_id FROM users WHERE id = $1",
            created_user_id,
        )
    finally:
        await connection.close()


async def seed_current_roles(database_url: str) -> tuple[UUID, UUID]:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    teacher_id = uuid4()
    student_id = uuid4()
    suffix = uuid4().hex[:12]
    try:
        await connection.executemany(
            """
            INSERT INTO users (id, username, password_hash, role)
            VALUES ($1, $2, 'argon2-test-hash', $3)
            """,
            [
                (teacher_id, f"current-teacher-{suffix}", "teacher"),
                (student_id, f"current-student-{suffix}", "student"),
            ],
        )
        return teacher_id, student_id
    finally:
        await connection.close()


async def delete_user_and_fetch_role(
    database_url: str,
    deleted_user_id: UUID,
    retained_user_id: UUID,
) -> str:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    try:
        await connection.execute("DELETE FROM users WHERE id = $1", deleted_user_id)
        role = await connection.fetchval(
            "SELECT role FROM users WHERE id = $1",
            retained_user_id,
        )
        assert isinstance(role, str)
        return role
    finally:
        await connection.close()


async def fetch_role(database_url: str, user_id: UUID) -> str:
    connection = await asyncpg.connect(asyncpg_url(database_url))
    try:
        role = await connection.fetchval("SELECT role FROM users WHERE id = $1", user_id)
        assert isinstance(role, str)
        return role
    finally:
        await connection.close()


def test_account_management_migration_converts_existing_roles() -> None:
    database_url = require_database_url()
    try:
        run_alembic(database_url, "upgrade", "head")
        run_alembic(database_url, "downgrade", M5_REVISION)
        student_id, admin_id = asyncio.run(seed_legacy_users(database_url))

        run_alembic(database_url, "upgrade", M6_REVISION)

        rows, columns, constraints, indexes = asyncio.run(fetch_contract(database_url))
        assert rows[student_id]["role"] == "student"
        assert rows[admin_id]["role"] == "admin"
        assert rows[student_id]["created_by_user_id"] is None
        assert rows[student_id]["must_change_password"] is False
        assert constraints["ck_users_role"] == {"student", "teacher", "admin"}
        assert (
            constraints["fk_users_created_by_user_id_users"] == "ON DELETE SET NULL"
        )
        assert "ix_users_created_by_user_id" in indexes
        assert columns["created_by_user_id"]["data_type"] == "uuid"
        assert columns["created_by_user_id"]["is_nullable"] == "YES"
        assert columns["must_change_password"]["is_nullable"] == "NO"
        assert "false" in columns["must_change_password"]["column_default"].lower()
        assert "student" in columns["role"]["column_default"]
    finally:
        run_alembic(database_url, "upgrade", "head")


def test_deleting_creator_sets_created_by_user_id_to_null() -> None:
    database_url = require_database_url()
    try:
        run_alembic(database_url, "upgrade", "head")
        run_alembic(database_url, "downgrade", M5_REVISION)
        creator_id, created_user_id = asyncio.run(seed_legacy_users(database_url))
        run_alembic(database_url, "upgrade", M6_REVISION)

        created_by_user_id = asyncio.run(
            set_creator_and_delete(database_url, creator_id, created_user_id)
        )

        assert created_by_user_id is None
    finally:
        run_alembic(database_url, "upgrade", "head")


def test_downgrade_rejects_teacher_and_restores_student_role() -> None:
    database_url = require_database_url()
    try:
        run_alembic(database_url, "upgrade", "head")
        run_alembic(database_url, "upgrade", M6_REVISION)
        teacher_id, student_id = asyncio.run(seed_current_roles(database_url))

        failed_downgrade = run_alembic(
            database_url,
            "downgrade",
            M5_REVISION,
            check=False,
        )

        assert failed_downgrade.returncode != 0
        assert "存在 teacher 账号，无法降级账号管理迁移。" in (
            failed_downgrade.stdout + failed_downgrade.stderr
        )
        assert asyncio.run(
            delete_user_and_fetch_role(database_url, teacher_id, student_id)
        ) == "student"

        run_alembic(database_url, "downgrade", M5_REVISION)

        assert asyncio.run(fetch_role(database_url, student_id)) == "user"
    finally:
        run_alembic(database_url, "upgrade", "head")
