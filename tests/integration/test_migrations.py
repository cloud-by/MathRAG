from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest

from tests.integration.database_safety import require_test_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_alembic(database_url: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", *args],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


async def vector_extension_version(database_url: str) -> str | None:
    connection = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        return await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
    finally:
        await connection.close()


def test_migration_upgrade_downgrade_upgrade_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    run_alembic(database_url, "downgrade", "base")
    assert asyncio.run(vector_extension_version(database_url)) is None

    run_alembic(database_url, "upgrade", "head")
    assert asyncio.run(vector_extension_version(database_url)) == "0.8.5"
    current = run_alembic(database_url, "current")
    assert "0004_create_identity_conversation_rag_tables (head)" in current.stdout

    run_alembic(database_url, "downgrade", "base")
    assert asyncio.run(vector_extension_version(database_url)) is None

    run_alembic(database_url, "upgrade", "head")
    assert asyncio.run(vector_extension_version(database_url)) == "0.8.5"
    current = run_alembic(database_url, "current")
    assert "0004_create_identity_conversation_rag_tables (head)" in current.stdout
