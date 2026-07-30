"""知识持久化迁移的 PostgreSQL 模式测试。"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def run_alembic(database_url: str, *args: str) -> None:
    """在测试数据库中运行指定的 Alembic 命令。"""
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", *args],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


async def fetch_schema(database_url: str) -> tuple[set[str], str | None, set[str], set[str], bool]:
    """读取知识表、向量类型、约束、索引及 vector 扩展状态。"""
    connection = await asyncpg.connect(
        database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    try:
        tables = {
            row["table_name"]
            for row in await connection.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name IN ('knowledge_items', 'knowledge_chunks')
                """
            )
        }
        embedding_format = await connection.fetchval(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'knowledge_chunks'
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        constraints = {
            row["conname"]
            for row in await connection.fetch(
                """
                SELECT table_constraint.conname
                FROM pg_constraint AS table_constraint
                JOIN pg_class AS relation ON relation.oid = table_constraint.conrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname IN ('knowledge_items', 'knowledge_chunks')
                """
            )
        }
        indexes = {
            row["indexname"]
            for row in await connection.fetch(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename IN ('knowledge_items', 'knowledge_chunks')
                """
            )
        }
        vector_extension_exists = await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )
        return tables, embedding_format, constraints, indexes, vector_extension_exists
    finally:
        await connection.close()


def test_knowledge_schema_upgrade_and_downgrade_round_trip() -> None:
    """知识表迁移创建完整模式，降级仅移除知识表。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")

    try:
        run_alembic(database_url, "downgrade", "base")
        run_alembic(database_url, "upgrade", "head")
        tables, embedding_format, constraints, indexes, vector_extension_exists = asyncio.run(
            fetch_schema(database_url)
        )

        assert tables == {"knowledge_items", "knowledge_chunks"}
        assert embedding_format == "vector(1024)"
        assert {
            "uq_knowledge_items_legacy_id",
            "ck_knowledge_items_difficulty",
            "ck_knowledge_items_visibility",
            "ck_knowledge_items_status",
            "ck_knowledge_items_revision",
            "fk_knowledge_chunks_knowledge_item_id_knowledge_items",
            "ck_knowledge_chunks_chunk_index",
            "ck_knowledge_chunks_status",
            "uq_knowledge_chunks_knowledge_item_id_chunk_index",
        } <= constraints
        assert {
            "ix_knowledge_items_category",
            "ix_knowledge_items_status",
            "ix_knowledge_chunks_status",
        } <= indexes
        assert vector_extension_exists is True

        run_alembic(database_url, "downgrade", "0001_enable_vector_extension")
        tables, _, _, _, vector_extension_exists = asyncio.run(fetch_schema(database_url))
        assert tables == set()
        assert vector_extension_exists is True
    finally:
        run_alembic(database_url, "upgrade", "head")
