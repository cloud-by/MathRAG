"""创建文档、导入任务及知识来源关联。"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0005_create_documents_ingestion_jobs"
down_revision: str | None = "0004_create_identity_conversation_rag_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """按依赖顺序创建 M5 模式。"""
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("storage_path", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "size_bytes > 0",
            name=op.f("ck_documents_size_bytes"),
        ),
        sa.CheckConstraint(
            "sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_documents_sha256_format"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed', 'archived')",
            name=op.f("ck_documents_status"),
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_documents_owner_id_users",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("storage_path", name="uq_documents_storage_path"),
        sa.UniqueConstraint(
            "owner_id",
            "sha256",
            name="uq_documents_owner_id_sha256",
        ),
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("requested_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("job_type", sa.String(length=16), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("progress", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "request_payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "job_type IN ('text', 'pdf', 'web', 'reindex')",
            name=op.f("ck_ingestion_jobs_job_type"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name=op.f("ck_ingestion_jobs_status"),
        ),
        sa.CheckConstraint(
            "progress BETWEEN 0 AND 100",
            name=op.f("ck_ingestion_jobs_progress"),
        ),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name=op.f("ck_ingestion_jobs_attempt_count"),
        ),
        sa.ForeignKeyConstraint(
            ["requested_by"],
            ["users.id"],
            name="fk_ingestion_jobs_requested_by_users",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_ingestion_jobs_document_id_documents",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingestion_jobs")),
    )

    op.add_column(
        "knowledge_items",
        sa.Column("ingestion_job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_items_ingestion_job_id_ingestion_jobs",
        "knowledge_items",
        "ingestion_jobs",
        ["ingestion_job_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "knowledge_chunks",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_chunks_document_id_documents",
        "knowledge_chunks",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_index("ix_documents_owner_id", "documents", ["owner_id"], unique=False)
    op.create_index(
        "ix_documents_status_created_at",
        "documents",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_ingestion_jobs_requested_by",
        "ingestion_jobs",
        ["requested_by"],
        unique=False,
    )
    op.create_index(
        "ix_ingestion_jobs_status_created_at",
        "ingestion_jobs",
        ["status", "created_at"],
        unique=False,
    )
    op.create_index(
        "uq_ingestion_jobs_document_id_job_type",
        "ingestion_jobs",
        ["document_id", "job_type"],
        unique=True,
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )
    op.create_index(
        "ix_knowledge_items_ingestion_job_id",
        "knowledge_items",
        ["ingestion_job_id"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_chunks_document_id",
        "knowledge_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        "uq_knowledge_chunks_document_id_chunk_index",
        "knowledge_chunks",
        ["document_id", "chunk_index"],
        unique=True,
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )


def downgrade() -> None:
    """严格按升级的逆序移除 M5 模式。"""
    op.drop_index(
        "uq_knowledge_chunks_document_id_chunk_index",
        table_name="knowledge_chunks",
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )
    op.drop_index("ix_knowledge_chunks_document_id", table_name="knowledge_chunks")
    op.drop_index(
        "ix_knowledge_items_ingestion_job_id",
        table_name="knowledge_items",
    )
    op.drop_index(
        "uq_ingestion_jobs_document_id_job_type",
        table_name="ingestion_jobs",
        postgresql_where=sa.text("document_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_ingestion_jobs_status_created_at",
        table_name="ingestion_jobs",
    )
    op.drop_index("ix_ingestion_jobs_requested_by", table_name="ingestion_jobs")
    op.drop_index("ix_documents_status_created_at", table_name="documents")
    op.drop_index("ix_documents_owner_id", table_name="documents")

    op.drop_constraint(
        "fk_knowledge_chunks_document_id_documents",
        "knowledge_chunks",
        type_="foreignkey",
    )
    op.drop_column("knowledge_chunks", "document_id")
    op.drop_constraint(
        "fk_knowledge_items_ingestion_job_id_ingestion_jobs",
        "knowledge_items",
        type_="foreignkey",
    )
    op.drop_column("knowledge_items", "ingestion_job_id")
    op.drop_table("ingestion_jobs")
    op.drop_table("documents")
