"""知识持久化 ORM 映射测试。"""

from __future__ import annotations

from datetime import datetime
from typing import get_type_hints

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, UniqueConstraint, inspect
from sqlalchemy.orm import Mapped

from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem


def constraint_names(table: object, constraint_type: type[object]) -> set[str | None]:
    """返回指定类型的数据库约束名称。"""
    return {
        constraint.name
        for constraint in table.constraints  # type: ignore[attr-defined]
        if isinstance(constraint, constraint_type)
    }


def index_columns(table: object) -> dict[str | None, tuple[str, ...]]:
    """返回索引名称及其有序列名。"""
    return {
        index.name: tuple(column.name for column in index.columns)
        for index in table.indexes  # type: ignore[attr-defined]
        if isinstance(index, Index)
    }


def foreign_keys_by_columns(table: object) -> dict[tuple[str, ...], ForeignKeyConstraint]:
    """按本表受约束列精确索引外键。"""
    return {
        tuple(column.name for column in constraint.columns): constraint
        for constraint in table.constraints  # type: ignore[attr-defined]
        if isinstance(constraint, ForeignKeyConstraint)
    }


def test_knowledge_item_table_maps_required_columns_and_constraints() -> None:
    """知识条目表保留迁移所需的列、默认值和约束名称。"""
    table = KnowledgeItem.__table__

    assert table.name == "knowledge_items"
    assert table.c.legacy_id.nullable is True
    assert table.c.owner_id.nullable is True
    assert table.c.ingestion_job_id.nullable is True
    assert table.c.category.index is True
    assert table.c.status.index is True
    assert table.c.keywords.default.arg(None) == []
    assert table.c.steps.default.arg(None) == []
    assert table.c.example.default.arg == ""
    assert table.c.visibility.default.arg == "public"
    assert table.c.status.default.arg == "indexing"
    assert table.c.revision.default.arg == 1
    assert constraint_names(table, UniqueConstraint) == {"uq_knowledge_items_legacy_id"}
    foreign_keys = foreign_keys_by_columns(table)
    assert set(foreign_keys) == {("owner_id",), ("ingestion_job_id",)}
    owner_foreign_key = foreign_keys[("owner_id",)]
    assert owner_foreign_key.name == "fk_knowledge_items_owner_id_users"
    assert owner_foreign_key.ondelete == "SET NULL"
    assert owner_foreign_key.elements[0].column.table.name == "users"
    ingestion_foreign_key = foreign_keys[("ingestion_job_id",)]
    assert (
        ingestion_foreign_key.name
        == "fk_knowledge_items_ingestion_job_id_ingestion_jobs"
    )
    assert ingestion_foreign_key.ondelete == "SET NULL"
    assert ingestion_foreign_key.elements[0].column.table.name == "ingestion_jobs"
    assert index_columns(table)["ix_knowledge_items_owner_id"] == ("owner_id",)
    assert index_columns(table)["ix_knowledge_items_ingestion_job_id"] == (
        "ingestion_job_id",
    )
    assert constraint_names(table, CheckConstraint) == {
        "ck_knowledge_items_difficulty",
        "ck_knowledge_items_visibility",
        "ck_knowledge_items_status",
        "ck_knowledge_items_revision",
    }
    assert index_columns(table)["ix_knowledge_items_visibility_status"] == (
        "visibility",
        "status",
    )


def test_knowledge_chunk_table_maps_foreign_key_vector_and_metadata_column() -> None:
    """知识分块表使用级联外键、1024 维向量和 metadata 数据库列。"""
    table = KnowledgeChunk.__table__
    foreign_keys = foreign_keys_by_columns(table)

    assert table.name == "knowledge_chunks"
    assert isinstance(table.c.embedding.type, Vector)
    assert table.c.embedding.type.dim == 1024
    assert table.c.metadata.name == "metadata"
    assert KnowledgeChunk.metadata_.property.columns[0].name == "metadata"
    assert table.c.metadata.default.arg(None) == {}
    assert table.c.status.default.arg == "pending"
    assert table.c.document_id.nullable is True
    assert set(foreign_keys) == {("knowledge_item_id",), ("document_id",)}
    item_foreign_key = foreign_keys[("knowledge_item_id",)]
    assert (
        item_foreign_key.name
        == "fk_knowledge_chunks_knowledge_item_id_knowledge_items"
    )
    assert item_foreign_key.ondelete == "CASCADE"
    document_foreign_key = foreign_keys[("document_id",)]
    assert document_foreign_key.name == "fk_knowledge_chunks_document_id_documents"
    assert document_foreign_key.ondelete == "SET NULL"
    assert document_foreign_key.elements[0].column.table.name == "documents"
    assert constraint_names(table, UniqueConstraint) == {
        "uq_knowledge_chunks_knowledge_item_id_chunk_index"
    }
    assert constraint_names(table, CheckConstraint) == {
        "ck_knowledge_chunks_chunk_index",
        "ck_knowledge_chunks_status",
        "ck_knowledge_chunks_ready_requires_embedding",
    }
    readiness_constraint = next(
        constraint
        for constraint in table.constraints
        if constraint.name == "ck_knowledge_chunks_ready_requires_embedding"
    )
    assert str(readiness_constraint.sqltext) == (
        "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)"
    )
    assert index_columns(table)["ix_knowledge_chunks_status_embedding_model"] == (
        "status",
        "embedding_model",
    )
    assert index_columns(table)["ix_knowledge_chunks_document_id"] == ("document_id",)
    document_chunk_index = next(
        index
        for index in table.indexes
        if index.name == "uq_knowledge_chunks_document_id_chunk_index"
    )
    assert document_chunk_index.unique is True
    assert tuple(column.name for column in document_chunk_index.columns) == (
        "document_id",
        "chunk_index",
    )
    assert str(document_chunk_index.dialect_options["postgresql"]["where"]) == (
        "document_id IS NOT NULL"
    )


def test_knowledge_item_and_chunk_relationships_use_orphan_cascade() -> None:
    """条目删除时其分块通过 ORM 和数据库级联清理。"""
    item_relationship = inspect(KnowledgeItem).relationships["chunks"]
    chunk_relationship = inspect(KnowledgeChunk).relationships["knowledge_item"]

    assert item_relationship.back_populates == "knowledge_item"
    assert item_relationship.cascade.delete_orphan is True
    assert item_relationship.passive_deletes is True
    assert chunk_relationship.back_populates == "chunks"


def test_timestamp_attributes_are_typed_as_datetime() -> None:
    """时间列的 ORM 类型契约应明确表达 datetime。"""
    item_hints = get_type_hints(KnowledgeItem)
    chunk_hints = get_type_hints(KnowledgeChunk)

    assert item_hints["created_at"] == Mapped[datetime]
    assert item_hints["updated_at"] == Mapped[datetime]
    assert chunk_hints["created_at"] == Mapped[datetime]
