"""M4 身份、会话与 RAG 持久化模式测试。"""

from __future__ import annotations

import asyncio
import ast
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from tests.integration.database_safety import require_test_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = (
    PROJECT_ROOT / "alembic" / "versions" / "0004_create_identity_conversation_rag_tables.py"
)
M4_TABLES = {
    "users",
    "user_sessions",
    "conversations",
    "messages",
    "rag_runs",
    "rag_references",
}


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


async def fetch_contract(database_url: str) -> tuple[set[str], set[str], set[str], bool]:
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
                WHERE table_schema = 'public'
                  AND table_name = ANY($1::text[])
                """,
                sorted(M4_TABLES),
            )
        }
        constraints = {
            row["conname"]
            for row in await connection.fetch(
                """
                SELECT constraint_record.conname
                FROM pg_constraint AS constraint_record
                JOIN pg_class AS relation ON relation.oid = constraint_record.conrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = ANY($1::text[])
                """,
                sorted(M4_TABLES | {"knowledge_items"}),
            )
        }
        indexes = {
            row["indexname"]
            for row in await connection.fetch(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = ANY($1::text[])
                """,
                sorted(M4_TABLES | {"knowledge_items"}),
            )
        }
        owner_exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'knowledge_items'
                  AND column_name = 'owner_id'
                  AND is_nullable = 'YES'
            )
            """
        )
        return tables, constraints, indexes, owner_exists
    finally:
        await connection.close()


async def assert_integrity_error(
    connection: asyncpg.Connection,
    error_type: type[Exception],
    query: str,
    *arguments: object,
) -> None:
    savepoint = connection.transaction()
    await savepoint.start()
    try:
        with pytest.raises(error_type):
            await connection.execute(query, *arguments)
    finally:
        await savepoint.rollback()


async def verify_relational_guards(database_url: str) -> None:
    connection = await asyncpg.connect(
        database_url.replace("postgresql+asyncpg://", "postgresql://")
    )
    transaction = connection.transaction()
    await transaction.start()
    try:
        user_id = uuid4()
        other_user_id = uuid4()
        conversation_id = uuid4()
        other_conversation_id = uuid4()
        question_id = uuid4()
        other_question_id = uuid4()
        run_id = uuid4()
        item_id = uuid4()
        chunk_id = uuid4()

        await connection.executemany(
            """
            INSERT INTO users (id, username, password_hash)
            VALUES ($1, $2, 'argon2-test-hash')
            """,
            [(user_id, "m4-user"), (other_user_id, "m4-other")],
        )
        await connection.executemany(
            """
            INSERT INTO conversations (id, user_id, title)
            VALUES ($1, $2, $3)
            """,
            [
                (conversation_id, user_id, "主会话"),
                (other_conversation_id, other_user_id, "其他会话"),
            ],
        )
        await connection.executemany(
            """
            INSERT INTO messages (id, conversation_id, role, content, status)
            VALUES ($1, $2, 'user', '问题', 'completed')
            """,
            [
                (question_id, conversation_id),
                (other_question_id, other_conversation_id),
            ],
        )
        await assert_integrity_error(
            connection,
            asyncpg.ForeignKeyViolationError,
            """
            INSERT INTO rag_runs
                (id, conversation_id, question_message_id, client_request_id, top_k)
            VALUES ($1, $2, $3, $4, 3)
            """,
            uuid4(),
            conversation_id,
            other_question_id,
            uuid4(),
        )
        client_request_id = uuid4()
        await connection.execute(
            """
            INSERT INTO rag_runs
                (id, conversation_id, question_message_id, client_request_id, top_k)
            VALUES ($1, $2, $3, $4, 3)
            """,
            run_id,
            conversation_id,
            question_id,
            client_request_id,
        )
        await assert_integrity_error(
            connection,
            asyncpg.UniqueViolationError,
            """
            INSERT INTO rag_runs
                (id, conversation_id, question_message_id, client_request_id, top_k)
            VALUES ($1, $2, $3, $4, 3)
            """,
            uuid4(),
            conversation_id,
            question_id,
            client_request_id,
        )
        await connection.execute(
            """
            INSERT INTO knowledge_items
                (id, owner_id, category, title, content, difficulty)
            VALUES ($1, $2, 'algebra', '快照条目', '原始内容', 'easy')
            """,
            item_id,
            user_id,
        )
        await connection.execute(
            """
            INSERT INTO knowledge_chunks
                (id, knowledge_item_id, chunk_index, retrieval_text, answer_context)
            VALUES ($1, $2, 0, '检索内容', '回答内容')
            """,
            chunk_id,
            item_id,
        )
        await connection.execute(
            """
            INSERT INTO rag_references (rag_run_id, chunk_id, rank, score, snapshot)
            VALUES ($1, $2, 1, 0.75, '{"source_id":"snapshot"}'::jsonb)
            """,
            run_id,
            chunk_id,
        )
        await connection.execute("DELETE FROM knowledge_chunks WHERE id = $1", chunk_id)
        reference = await connection.fetchrow(
            "SELECT chunk_id, snapshot FROM rag_references WHERE rag_run_id = $1 AND rank = 1",
            run_id,
        )
        assert reference is not None
        assert reference["chunk_id"] is None
        assert reference["snapshot"] == '{"source_id": "snapshot"}'

        await connection.execute("DELETE FROM users WHERE id = $1", user_id)
        assert await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM conversations WHERE id = $1)",
            conversation_id,
        ) is False
        assert await connection.fetchval(
            "SELECT EXISTS (SELECT 1 FROM rag_runs WHERE id = $1)",
            run_id,
        ) is False
        assert await connection.fetchval(
            "SELECT owner_id FROM knowledge_items WHERE id = $1",
            item_id,
        ) is None
    finally:
        await transaction.rollback()
        await connection.close()


def test_m4_migration_is_self_contained() -> None:
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert all(not module.startswith("app.") for module in imported_modules)
    assert 'revision: str = "0004_create_identity_conversation_rag_tables"' in source
    assert 'down_revision: str | None = "0003_enforce_vector_readiness"' in source


def test_m4_schema_upgrade_and_downgrade_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    expected_constraints = {
        "ck_users_username_format",
        "ck_users_role",
        "ck_users_status",
        "uq_users_username",
        "uq_users_email",
        "fk_user_sessions_user_id_users",
        "uq_user_sessions_token_hash",
        "ck_user_sessions_expires_after_created",
        "ck_user_sessions_revoked_after_created",
        "fk_conversations_user_id_users",
        "ck_conversations_status",
        "fk_messages_conversation_id_conversations",
        "ck_messages_role",
        "ck_messages_status",
        "uq_messages_conversation_id_id",
        "fk_rag_runs_conversation_id_conversations",
        "fk_rag_runs_question_message_conversation_messages",
        "fk_rag_runs_answer_message_conversation_messages",
        "uq_rag_runs_conversation_id_client_request_id",
        "ck_rag_runs_top_k",
        "ck_rag_runs_status",
        "ck_rag_runs_latency_ms",
        "fk_rag_references_rag_run_id_rag_runs",
        "fk_rag_references_chunk_id_knowledge_chunks",
        "pk_rag_references",
        "uq_rag_references_rag_run_id_chunk_id",
        "ck_rag_references_rank",
        "fk_knowledge_items_owner_id_users",
    }
    expected_indexes = {
        "ix_user_sessions_user_id_expires_at_active",
        "ix_conversations_user_id_updated_at_id",
        "ix_messages_conversation_id_created_at_id",
        "ix_knowledge_items_owner_id",
    }

    try:
        run_alembic(database_url, "upgrade", "head")
        tables, constraints, indexes, owner_exists = asyncio.run(fetch_contract(database_url))
        assert tables == M4_TABLES
        assert expected_constraints <= constraints
        assert expected_indexes <= indexes
        assert owner_exists is True
        asyncio.run(verify_relational_guards(database_url))

        run_alembic(database_url, "downgrade", "0003_enforce_vector_readiness")
        tables, _, _, owner_exists = asyncio.run(fetch_contract(database_url))
        assert tables == set()
        assert owner_exists is False

        run_alembic(database_url, "upgrade", "head")
        current = run_alembic(database_url, "current")
        assert "0004_create_identity_conversation_rag_tables (head)" in current.stdout
    finally:
        run_alembic(database_url, "upgrade", "head")
