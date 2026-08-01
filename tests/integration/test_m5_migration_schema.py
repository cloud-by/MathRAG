"""M5 文档与导入任务持久化模式测试。"""

from __future__ import annotations

import asyncio
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest

from tests.integration.database_safety import require_test_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = (
    PROJECT_ROOT / "alembic" / "versions" / "0005_create_documents_ingestion_jobs.py"
)
M5_TABLES = {"documents", "ingestion_jobs"}


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
        encoding="utf-8",
    )


async def fetch_contract(
    database_url: str,
) -> tuple[
    set[str],
    dict[tuple[str, str], tuple[str, str, int | None, str | None]],
    dict[
        str,
        tuple[str, str, tuple[str, ...], str | None, tuple[str, ...]],
    ],
    dict[str, tuple[str, bool, tuple[str, ...], str | None]],
]:
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
                sorted(M5_TABLES),
            )
        }
        columns = {
            (row["table_name"], row["column_name"]): (
                row["data_type"],
                row["is_nullable"],
                row["character_maximum_length"],
                row["column_default"],
            )
            for row in await connection.fetch(
                """
                SELECT table_name, column_name, data_type, is_nullable,
                       character_maximum_length, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = ANY($1::text[])
                """,
                sorted(M5_TABLES | {"knowledge_items", "knowledge_chunks"}),
            )
        }
        constraints = {
            row["conname"]: (
                row["contype"],
                row["definition"],
                tuple(row["local_columns"]),
                row["foreign_table"],
                tuple(row["foreign_columns"]),
            )
            for row in await connection.fetch(
                """
                SELECT constraint_record.conname,
                       constraint_record.contype::text AS contype,
                       pg_get_constraintdef(constraint_record.oid) AS definition,
                       ARRAY(
                           SELECT attribute.attname
                           FROM unnest(constraint_record.conkey)
                                WITH ORDINALITY AS key(attnum, ordinal_position)
                           JOIN pg_attribute AS attribute
                             ON attribute.attrelid = relation.oid
                            AND attribute.attnum = key.attnum
                           ORDER BY key.ordinal_position
                       ) AS local_columns,
                       referenced_relation.relname AS foreign_table,
                       ARRAY(
                           SELECT attribute.attname
                           FROM unnest(constraint_record.confkey)
                                WITH ORDINALITY AS key(attnum, ordinal_position)
                           JOIN pg_attribute AS attribute
                             ON attribute.attrelid = constraint_record.confrelid
                            AND attribute.attnum = key.attnum
                           ORDER BY key.ordinal_position
                       ) AS foreign_columns
                FROM pg_constraint AS constraint_record
                JOIN pg_class AS relation ON relation.oid = constraint_record.conrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                LEFT JOIN pg_class AS referenced_relation
                  ON referenced_relation.oid = constraint_record.confrelid
                WHERE namespace.nspname = 'public'
                  AND relation.relname = ANY($1::text[])
                """,
                sorted(M5_TABLES | {"knowledge_items", "knowledge_chunks"}),
            )
        }
        indexes = {
            row["index_name"]: (
                row["table_name"],
                row["is_unique"],
                tuple(row["columns"]),
                row["predicate"],
            )
            for row in await connection.fetch(
                """
                SELECT relation.relname AS table_name,
                       index_relation.relname AS index_name,
                       index_record.indisunique AS is_unique,
                       ARRAY(
                           SELECT attribute.attname
                           FROM unnest(index_record.indkey)
                                WITH ORDINALITY AS key(attnum, ordinal_position)
                           JOIN pg_attribute AS attribute
                             ON attribute.attrelid = relation.oid
                            AND attribute.attnum = key.attnum
                           ORDER BY key.ordinal_position
                       ) AS columns,
                       pg_get_expr(index_record.indpred, index_record.indrelid) AS predicate
                FROM pg_index AS index_record
                JOIN pg_class AS relation ON relation.oid = index_record.indrelid
                JOIN pg_class AS index_relation ON index_relation.oid = index_record.indexrelid
                JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = 'public'
                  AND relation.relname = ANY($1::text[])
                """,
                sorted(M5_TABLES | {"knowledge_items", "knowledge_chunks"}),
            )
        }
        return tables, columns, constraints, indexes
    finally:
        await connection.close()


def assert_m5_absent(
    contract: tuple[
        set[str],
        dict[tuple[str, str], tuple[str, str, int | None, str | None]],
        dict[
            str,
            tuple[str, str, tuple[str, ...], str | None, tuple[str, ...]],
        ],
        dict[str, tuple[str, bool, tuple[str, ...], str | None]],
    ],
) -> None:
    tables, columns, _, _ = contract
    assert tables == set()
    assert ("knowledge_items", "ingestion_job_id") not in columns
    assert ("knowledge_chunks", "document_id") not in columns


def assert_column_contract(
    columns: dict[tuple[str, str], tuple[str, str, int | None, str | None]],
    table_name: str,
    column_name: str,
    *,
    data_type: str,
    nullable: str,
    maximum_length: int | None = None,
    default_tokens: tuple[str, ...] | None = None,
) -> None:
    """逐列验证 PostgreSQL 类型、可空性、长度和服务端默认值。"""
    actual_type, actual_nullable, actual_length, actual_default = columns[
        (table_name, column_name)
    ]
    assert (actual_type, actual_nullable, actual_length) == (
        data_type,
        nullable,
        maximum_length,
    )
    if default_tokens is None:
        assert actual_default is None
    else:
        assert actual_default is not None
        lowered_default = actual_default.lower()
        assert all(token in lowered_default for token in default_tokens)


def allowed_values(constraint_definition: str) -> set[str]:
    """从 PostgreSQL 枚举式 CHECK 定义中提取允许值。"""
    return set(re.findall(r"'([^']+)'", constraint_definition))


def test_m5_migration_is_self_contained() -> None:
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert all(not module.startswith("app.") for module in imported_modules)
    assert 'revision: str = "0005_create_documents_ingestion_jobs"' in source
    assert (
        'down_revision: str | None = "0004_create_identity_conversation_rag_tables"'
        in source
    )


def test_m5_schema_upgrade_and_downgrade_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    expected_constraints = {
        "pk_documents",
        "fk_documents_owner_id_users",
        "uq_documents_storage_path",
        "uq_documents_owner_id_sha256",
        "ck_documents_size_bytes",
        "ck_documents_sha256_format",
        "ck_documents_status",
        "pk_ingestion_jobs",
        "fk_ingestion_jobs_requested_by_users",
        "fk_ingestion_jobs_document_id_documents",
        "ck_ingestion_jobs_job_type",
        "ck_ingestion_jobs_status",
        "ck_ingestion_jobs_progress",
        "ck_ingestion_jobs_attempt_count",
        "fk_knowledge_items_ingestion_job_id_ingestion_jobs",
        "fk_knowledge_chunks_document_id_documents",
    }
    expected_column_contracts = {
        ("documents", "id"): ("uuid", "NO", None, None),
        ("documents", "owner_id"): ("uuid", "YES", None, None),
        ("documents", "original_name"): ("character varying", "NO", 255, None),
        ("documents", "storage_path"): ("character varying", "NO", 512, None),
        ("documents", "mime_type"): ("character varying", "NO", 128, None),
        ("documents", "size_bytes"): ("bigint", "NO", None, None),
        ("documents", "sha256"): ("character varying", "NO", 64, None),
        ("documents", "status"): (
            "character varying",
            "NO",
            16,
            ("pending",),
        ),
        ("documents", "created_at"): (
            "timestamp with time zone",
            "NO",
            None,
            ("now",),
        ),
        ("documents", "updated_at"): (
            "timestamp with time zone",
            "NO",
            None,
            ("now",),
        ),
        ("ingestion_jobs", "id"): ("uuid", "NO", None, None),
        ("ingestion_jobs", "requested_by"): ("uuid", "YES", None, None),
        ("ingestion_jobs", "document_id"): ("uuid", "YES", None, None),
        ("ingestion_jobs", "job_type"): (
            "character varying",
            "NO",
            16,
            None,
        ),
        ("ingestion_jobs", "status"): (
            "character varying",
            "NO",
            16,
            ("pending",),
        ),
        ("ingestion_jobs", "progress"): ("integer", "NO", None, ("0",)),
        ("ingestion_jobs", "request_payload"): (
            "jsonb",
            "NO",
            None,
            ("{}", "jsonb"),
        ),
        ("ingestion_jobs", "attempt_count"): (
            "integer",
            "NO",
            None,
            ("0",),
        ),
        ("ingestion_jobs", "error_code"): (
            "character varying",
            "YES",
            64,
            None,
        ),
        ("ingestion_jobs", "error_message"): (
            "character varying",
            "YES",
            500,
            None,
        ),
        ("ingestion_jobs", "started_at"): (
            "timestamp with time zone",
            "YES",
            None,
            None,
        ),
        ("ingestion_jobs", "finished_at"): (
            "timestamp with time zone",
            "YES",
            None,
            None,
        ),
        ("ingestion_jobs", "created_at"): (
            "timestamp with time zone",
            "NO",
            None,
            ("now",),
        ),
        ("ingestion_jobs", "updated_at"): (
            "timestamp with time zone",
            "NO",
            None,
            ("now",),
        ),
    }

    try:
        # 无论其他迁移测试留下什么状态，都从 0004 边界开始验证 M5。
        run_alembic(database_url, "upgrade", "head")
        run_alembic(database_url, "downgrade", "0004_create_identity_conversation_rag_tables")
        assert_m5_absent(asyncio.run(fetch_contract(database_url)))

        run_alembic(database_url, "upgrade", "0005_create_documents_ingestion_jobs")
        tables, columns, constraints, indexes = asyncio.run(fetch_contract(database_url))

        assert tables == M5_TABLES
        assert expected_constraints <= constraints.keys()
        assert {
            key for key in columns if key[0] in M5_TABLES
        } == expected_column_contracts.keys()
        for (table_name, column_name), (
            data_type,
            nullable,
            maximum_length,
            default_tokens,
        ) in expected_column_contracts.items():
            assert_column_contract(
                columns,
                table_name,
                column_name,
                data_type=data_type,
                nullable=nullable,
                maximum_length=maximum_length,
                default_tokens=default_tokens,
            )
        assert_column_contract(
            columns,
            "knowledge_items",
            "ingestion_job_id",
            data_type="uuid",
            nullable="YES",
        )
        assert_column_contract(
            columns,
            "knowledge_chunks",
            "document_id",
            data_type="uuid",
            nullable="YES",
        )
        assert_column_contract(
            columns,
            "knowledge_chunks",
            "knowledge_item_id",
            data_type="uuid",
            nullable="NO",
        )

        check_definitions = {
            name: constraints[name][1]
            for name in (
                "ck_documents_size_bytes",
                "ck_documents_sha256_format",
                "ck_documents_status",
                "ck_ingestion_jobs_job_type",
                "ck_ingestion_jobs_status",
                "ck_ingestion_jobs_progress",
                "ck_ingestion_jobs_attempt_count",
            )
        }
        assert all(constraints[name][0] == "c" for name in check_definitions)
        assert re.search(r"size_bytes\s*>\s*0", check_definitions["ck_documents_size_bytes"])
        assert "^[0-9a-f]{64}$" in check_definitions["ck_documents_sha256_format"]
        assert allowed_values(check_definitions["ck_documents_status"]) == {
            "pending",
            "processing",
            "ready",
            "failed",
            "archived",
        }
        assert allowed_values(check_definitions["ck_ingestion_jobs_job_type"]) == {
            "text",
            "pdf",
            "web",
            "reindex",
        }
        assert allowed_values(check_definitions["ck_ingestion_jobs_status"]) == {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }
        progress_definition = check_definitions["ck_ingestion_jobs_progress"]
        assert re.search(r"progress\s*>=\s*0", progress_definition)
        assert re.search(r"progress\s*<=\s*100", progress_definition)
        assert re.search(
            r"attempt_count\s*>=\s*0",
            check_definitions["ck_ingestion_jobs_attempt_count"],
        )

        expected_foreign_keys = {
            "fk_documents_owner_id_users": (
                ("owner_id",),
                "users",
                ("id",),
                "FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE SET NULL",
            ),
            "fk_ingestion_jobs_requested_by_users": (
                ("requested_by",),
                "users",
                ("id",),
                "FOREIGN KEY (requested_by) REFERENCES users(id) ON DELETE SET NULL",
            ),
            "fk_ingestion_jobs_document_id_documents": (
                ("document_id",),
                "documents",
                ("id",),
                "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL",
            ),
            "fk_knowledge_items_ingestion_job_id_ingestion_jobs": (
                ("ingestion_job_id",),
                "ingestion_jobs",
                ("id",),
                "FOREIGN KEY (ingestion_job_id) REFERENCES ingestion_jobs(id) ON DELETE SET NULL",
            ),
            "fk_knowledge_chunks_document_id_documents": (
                ("document_id",),
                "documents",
                ("id",),
                "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL",
            ),
        }
        for name, expected in expected_foreign_keys.items():
            constraint_type, definition, local_columns, foreign_table, foreign_columns = (
                constraints[name]
            )
            assert constraint_type == "f"
            assert (local_columns, foreign_table, foreign_columns, definition) == expected

        assert constraints["pk_documents"][2] == ("id",)
        assert constraints["pk_ingestion_jobs"][2] == ("id",)
        assert constraints["uq_documents_storage_path"][:3] == (
            "u",
            "UNIQUE (storage_path)",
            ("storage_path",),
        )
        assert constraints["uq_documents_owner_id_sha256"][:3] == (
            "u",
            "UNIQUE (owner_id, sha256)",
            ("owner_id", "sha256"),
        )

        expected_m5_table_indexes = {
            "pk_documents": ("documents", True, ("id",), None),
            "uq_documents_storage_path": (
                "documents",
                True,
                ("storage_path",),
                None,
            ),
            "uq_documents_owner_id_sha256": (
                "documents",
                True,
                ("owner_id", "sha256"),
                None,
            ),
            "ix_documents_owner_id": ("documents", False, ("owner_id",), None),
            "ix_documents_status_created_at": (
                "documents",
                False,
                ("status", "created_at"),
                None,
            ),
            "pk_ingestion_jobs": ("ingestion_jobs", True, ("id",), None),
            "ix_ingestion_jobs_requested_by": (
                "ingestion_jobs",
                False,
                ("requested_by",),
                None,
            ),
            "ix_ingestion_jobs_status_created_at": (
                "ingestion_jobs",
                False,
                ("status", "created_at"),
                None,
            ),
        }
        actual_m5_table_indexes = {
            name: contract
            for name, contract in indexes.items()
            if contract[0] in M5_TABLES
            and name != "uq_ingestion_jobs_document_id_job_type"
        }
        assert actual_m5_table_indexes == expected_m5_table_indexes
        for name, expected_without_predicate in {
            "uq_ingestion_jobs_document_id_job_type": (
                "ingestion_jobs",
                True,
                ("document_id", "job_type"),
            ),
            "uq_knowledge_chunks_document_id_chunk_index": (
                "knowledge_chunks",
                True,
                ("document_id", "chunk_index"),
            ),
        }.items():
            table_name, is_unique, index_columns, predicate = indexes[name]
            assert (table_name, is_unique, index_columns) == expected_without_predicate
            assert predicate is not None
            assert re.fullmatch(r"\(?document_id IS NOT NULL\)?", predicate)
        assert indexes["ix_knowledge_items_ingestion_job_id"] == (
            "knowledge_items",
            False,
            ("ingestion_job_id",),
            None,
        )
        assert indexes["ix_knowledge_chunks_document_id"] == (
            "knowledge_chunks",
            False,
            ("document_id",),
            None,
        )

        run_alembic(database_url, "downgrade", "0004_create_identity_conversation_rag_tables")
        assert_m5_absent(asyncio.run(fetch_contract(database_url)))

        run_alembic(database_url, "upgrade", "head")
        current = run_alembic(database_url, "current")
        assert "0006_add_account_management (head)" in current.stdout
    finally:
        run_alembic(database_url, "upgrade", "head")
