"""创建知识持久化表。"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "0002_create_knowledge_tables"
down_revision: str | None = "0001_enable_vector_extension"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """创建知识条目及其检索分块。"""
    op.create_table(
        "knowledge_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("legacy_id", sa.String(length=64), nullable=True),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column(
            "keywords",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("example", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "steps",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        sa.Column(
            "visibility",
            sa.String(length=16),
            server_default=sa.text("'public'"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'indexing'"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')",
            name=sa.schema.conv("ck_knowledge_items_difficulty"),
        ),
        sa.CheckConstraint(
            "visibility IN ('public', 'private')",
            name=sa.schema.conv("ck_knowledge_items_visibility"),
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'indexing', 'ready', 'failed', 'archived')",
            name=sa.schema.conv("ck_knowledge_items_status"),
        ),
        sa.CheckConstraint("revision > 0", name=sa.schema.conv("ck_knowledge_items_revision")),
        sa.PrimaryKeyConstraint("id", name="pk_knowledge_items"),
        sa.UniqueConstraint("legacy_id", name="uq_knowledge_items_legacy_id"),
    )
    op.create_index("ix_knowledge_items_category", "knowledge_items", ["category"], unique=False)
    op.create_index("ix_knowledge_items_status", "knowledge_items", ["status"], unique=False)

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("knowledge_item_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("answer_context", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dim=1024), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name=sa.schema.conv("ck_knowledge_chunks_chunk_index"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name=sa.schema.conv("ck_knowledge_chunks_status"),
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_item_id"],
            ["knowledge_items.id"],
            name="fk_knowledge_chunks_knowledge_item_id_knowledge_items",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_knowledge_chunks"),
        sa.UniqueConstraint(
            "knowledge_item_id",
            "chunk_index",
            name="uq_knowledge_chunks_knowledge_item_id_chunk_index",
        ),
    )
    op.create_index("ix_knowledge_chunks_status", "knowledge_chunks", ["status"], unique=False)


def downgrade() -> None:
    """删除知识表，保留 pgvector 扩展。"""
    op.drop_index("ix_knowledge_chunks_status", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_index("ix_knowledge_items_status", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_category", table_name="knowledge_items")
    op.drop_table("knowledge_items")
