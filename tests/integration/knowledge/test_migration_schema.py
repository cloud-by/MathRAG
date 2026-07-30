"""知识持久化迁移的 PostgreSQL 模式测试。"""

from __future__ import annotations

import asyncio
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIGRATION_PATH = PROJECT_ROOT / "alembic" / "versions" / "0002_create_knowledge_tables.py"


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


async def fetch_schema(
    database_url: str,
) -> tuple[set[str], str | None, set[str], dict[str, str], bool]:
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
            row["indexname"]: row["indexdef"]
            for row in await connection.fetch(
                """
                SELECT indexname, indexdef
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


def test_knowledge_migration_is_self_contained() -> None:
    """历史迁移只能依赖 Alembic、SQLAlchemy 和已安装的数据库类型。"""
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert all(not module.startswith("app.") for module in imported_modules)
    assert source.count("sa.DateTime(timezone=True)") == 3
    assert "onupdate=" not in source


async def fetch_column_contract(database_url: str) -> dict[tuple[str, str], asyncpg.Record]:
    """读取知识表每一列的 PostgreSQL 类型、可空性和服务端默认值。"""
    connection = await asyncpg.connect(
        database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    try:
        records = await connection.fetch(
            """
            SELECT table_name, column_name, data_type, udt_name,
                   character_maximum_length, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ('knowledge_items', 'knowledge_chunks')
            """
        )
        return {(record["table_name"], record["column_name"]): record for record in records}
    finally:
        await connection.close()


async def fetch_constraint_definitions(database_url: str) -> dict[str, str]:
    """读取知识表约束的实际 PostgreSQL 定义。"""
    connection = await asyncpg.connect(
        database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    try:
        records = await connection.fetch(
            """
            SELECT table_constraint.conname, pg_get_constraintdef(table_constraint.oid) AS definition
            FROM pg_constraint AS table_constraint
            JOIN pg_class AS relation ON relation.oid = table_constraint.conrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname IN ('knowledge_items', 'knowledge_chunks')
            """
        )
        return {record["conname"]: record["definition"] for record in records}
    finally:
        await connection.close()


async def verify_database_constraints(database_url: str) -> None:
    """通过真实写入验证检查约束和外键级联删除。"""
    connection = await asyncpg.connect(
        database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    item_id = uuid4()
    chunk_id = uuid4()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await connection.execute(
            """
            INSERT INTO knowledge_items (id, category, title, content, difficulty)
            VALUES ($1, 'algebra', '测试条目', '测试内容', 'easy')
            """,
            item_id,
        )
        await connection.execute(
            """
            INSERT INTO knowledge_chunks
                (id, knowledge_item_id, chunk_index, retrieval_text, answer_context)
            VALUES ($1, $2, 0, '检索文本', '回答上下文')
            """,
            chunk_id,
            item_id,
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_items (id, category, title, content, difficulty)
            VALUES ($1, 'algebra', '非法难度', '测试内容', 'invalid')
            """,
            uuid4(),
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_items (id, category, title, content, difficulty, visibility)
            VALUES ($1, 'algebra', '非法可见性', '测试内容', 'easy', 'invalid')
            """,
            uuid4(),
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_items (id, category, title, content, difficulty, status)
            VALUES ($1, 'algebra', '非法状态', '测试内容', 'easy', 'invalid')
            """,
            uuid4(),
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_items (id, category, title, content, difficulty, revision)
            VALUES ($1, 'algebra', '非法版本', '测试内容', 'easy', 0)
            """,
            uuid4(),
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_chunks
                (id, knowledge_item_id, chunk_index, retrieval_text, answer_context)
            VALUES ($1, $2, -1, '非法分块', '回答上下文')
            """,
            uuid4(),
            item_id,
        )
        await assert_check_violation(
            connection,
            """
            INSERT INTO knowledge_chunks
                (id, knowledge_item_id, chunk_index, retrieval_text, answer_context, status)
            VALUES ($1, $2, 1, '非法状态分块', '回答上下文', 'invalid')
            """,
            uuid4(),
            item_id,
        )

        await connection.execute("DELETE FROM knowledge_items WHERE id = $1", item_id)
        assert await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM knowledge_chunks WHERE id = $1)", chunk_id
        ) is False
    finally:
        await transaction.rollback()
        await connection.close()


async def assert_check_violation(
    connection: asyncpg.Connection,
    query: str,
    *arguments: object,
) -> None:
    """在 savepoint 中确认约束拒绝写入，并无条件回滚该写入。"""
    savepoint = connection.transaction()
    await savepoint.start()
    try:
        with pytest.raises(asyncpg.CheckViolationError):
            await connection.execute(query, *arguments)
    finally:
        await savepoint.rollback()


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
        columns = asyncio.run(fetch_column_contract(database_url))
        constraint_definitions = asyncio.run(fetch_constraint_definitions(database_url))

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
        } <= indexes.keys()
        assert indexes["ix_knowledge_items_category"] == (
            "CREATE INDEX ix_knowledge_items_category "
            "ON public.knowledge_items USING btree (category)"
        )
        assert indexes["ix_knowledge_items_status"] == (
            "CREATE INDEX ix_knowledge_items_status "
            "ON public.knowledge_items USING btree (status)"
        )
        assert indexes["ix_knowledge_chunks_status"] == (
            "CREATE INDEX ix_knowledge_chunks_status "
            "ON public.knowledge_chunks USING btree (status)"
        )
        assert vector_extension_exists is True
        assert set(columns) == {
            ("knowledge_items", "id"),
            ("knowledge_items", "legacy_id"),
            ("knowledge_items", "category"),
            ("knowledge_items", "title"),
            ("knowledge_items", "keywords"),
            ("knowledge_items", "content"),
            ("knowledge_items", "example"),
            ("knowledge_items", "steps"),
            ("knowledge_items", "difficulty"),
            ("knowledge_items", "visibility"),
            ("knowledge_items", "status"),
            ("knowledge_items", "revision"),
            ("knowledge_items", "created_at"),
            ("knowledge_items", "updated_at"),
            ("knowledge_chunks", "id"),
            ("knowledge_chunks", "knowledge_item_id"),
            ("knowledge_chunks", "chunk_index"),
            ("knowledge_chunks", "retrieval_text"),
            ("knowledge_chunks", "answer_context"),
            ("knowledge_chunks", "embedding"),
            ("knowledge_chunks", "embedding_model"),
            ("knowledge_chunks", "metadata"),
            ("knowledge_chunks", "status"),
            ("knowledge_chunks", "created_at"),
        }
        for table_name, column_name, data_type, maximum_length, nullable in (
            ("knowledge_items", "id", "uuid", None, "NO"),
            ("knowledge_items", "legacy_id", "character varying", 64, "YES"),
            ("knowledge_items", "category", "character varying", 128, "NO"),
            ("knowledge_items", "title", "character varying", 255, "NO"),
            ("knowledge_items", "keywords", "jsonb", None, "NO"),
            ("knowledge_items", "content", "text", None, "NO"),
            ("knowledge_items", "example", "text", None, "NO"),
            ("knowledge_items", "steps", "jsonb", None, "NO"),
            ("knowledge_items", "difficulty", "character varying", 16, "NO"),
            ("knowledge_items", "visibility", "character varying", 16, "NO"),
            ("knowledge_items", "status", "character varying", 16, "NO"),
            ("knowledge_items", "revision", "integer", None, "NO"),
            ("knowledge_items", "created_at", "timestamp with time zone", None, "NO"),
            ("knowledge_items", "updated_at", "timestamp with time zone", None, "NO"),
            ("knowledge_chunks", "id", "uuid", None, "NO"),
            ("knowledge_chunks", "knowledge_item_id", "uuid", None, "NO"),
            ("knowledge_chunks", "chunk_index", "integer", None, "NO"),
            ("knowledge_chunks", "retrieval_text", "text", None, "NO"),
            ("knowledge_chunks", "answer_context", "text", None, "NO"),
            ("knowledge_chunks", "embedding", "USER-DEFINED", None, "YES"),
            ("knowledge_chunks", "embedding_model", "character varying", 128, "YES"),
            ("knowledge_chunks", "metadata", "jsonb", None, "NO"),
            ("knowledge_chunks", "status", "character varying", 16, "NO"),
            ("knowledge_chunks", "created_at", "timestamp with time zone", None, "NO"),
        ):
            column = columns[(table_name, column_name)]
            assert (
                column["data_type"],
                column["character_maximum_length"],
                column["is_nullable"],
            ) == (data_type, maximum_length, nullable)
        assert columns[("knowledge_chunks", "embedding")]["udt_name"] == "vector"
        expected_defaults: dict[tuple[str, str], str | None] = {
            ("knowledge_items", "id"): None,
            ("knowledge_items", "legacy_id"): None,
            ("knowledge_items", "category"): None,
            ("knowledge_items", "title"): None,
            ("knowledge_items", "keywords"): "[] jsonb",
            ("knowledge_items", "content"): None,
            ("knowledge_items", "example"): "'' text",
            ("knowledge_items", "steps"): "[] jsonb",
            ("knowledge_items", "difficulty"): None,
            ("knowledge_items", "visibility"): "public character varying",
            ("knowledge_items", "status"): "indexing character varying",
            ("knowledge_items", "revision"): "1",
            ("knowledge_items", "created_at"): "now",
            ("knowledge_items", "updated_at"): "now",
            ("knowledge_chunks", "id"): None,
            ("knowledge_chunks", "knowledge_item_id"): None,
            ("knowledge_chunks", "chunk_index"): None,
            ("knowledge_chunks", "retrieval_text"): None,
            ("knowledge_chunks", "answer_context"): None,
            ("knowledge_chunks", "embedding"): None,
            ("knowledge_chunks", "embedding_model"): None,
            ("knowledge_chunks", "metadata"): "{} jsonb",
            ("knowledge_chunks", "status"): "pending character varying",
            ("knowledge_chunks", "created_at"): "now",
        }
        assert set(expected_defaults) == set(columns)
        for column_key, expected_default in expected_defaults.items():
            actual_default = columns[column_key]["column_default"]
            if expected_default is None:
                assert actual_default is None
            else:
                assert actual_default is not None
                assert all(token in actual_default.lower() for token in expected_default.split())
        assert allowed_values(constraint_definitions["ck_knowledge_items_difficulty"]) == {
            "easy",
            "medium",
            "hard",
        }
        assert allowed_values(constraint_definitions["ck_knowledge_items_visibility"]) == {
            "public",
            "private",
        }
        assert allowed_values(constraint_definitions["ck_knowledge_items_status"]) == {
            "draft",
            "indexing",
            "ready",
            "failed",
            "archived",
        }
        assert allowed_values(constraint_definitions["ck_knowledge_chunks_status"]) == {
            "pending",
            "ready",
            "failed",
        }
        assert re.search(
            r"revision\s*>\s*0",
            constraint_definitions["ck_knowledge_items_revision"],
        )
        assert re.search(
            r"chunk_index\s*>=\s*0",
            constraint_definitions["ck_knowledge_chunks_chunk_index"],
        )
        assert constraint_definitions["uq_knowledge_items_legacy_id"] == "UNIQUE (legacy_id)"
        assert (
            constraint_definitions["uq_knowledge_chunks_knowledge_item_id_chunk_index"]
            == "UNIQUE (knowledge_item_id, chunk_index)"
        )
        assert (
            constraint_definitions["fk_knowledge_chunks_knowledge_item_id_knowledge_items"]
            == "FOREIGN KEY (knowledge_item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE"
        )
        asyncio.run(verify_database_constraints(database_url))

        run_alembic(database_url, "downgrade", "0001_enable_vector_extension")
        tables, _, _, _, vector_extension_exists = asyncio.run(fetch_schema(database_url))
        assert tables == set()
        assert vector_extension_exists is True
    finally:
        run_alembic(database_url, "upgrade", "head")


def allowed_values(constraint_definition: str) -> set[str]:
    """从 PostgreSQL 的枚举式 CHECK 定义中提取全部允许值。"""
    return set(re.findall(r"'([^']+)'", constraint_definition))
