"""创建身份、会话和 RAG 持久化模式。"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0004_create_identity_conversation_rag_tables"
down_revision: str | None = "0003_enforce_vector_readiness"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """按依赖顺序创建 M4 表和知识所有者关联。"""
    # M4 的描述性 revision 标识超过 Alembic 默认的 32 字符上限。
    op.alter_column(
        "alembic_version",
        "version_num",
        existing_type=sa.String(length=32),
        type_=sa.String(length=64),
        existing_nullable=False,
    )
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=16), server_default="user", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="active", nullable=False),
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
            "username ~ '^[a-z0-9][a-z0-9_.-]{2,63}$'",
            name=op.f("ck_users_username_format"),
        ),
        sa.CheckConstraint(
            "role IN ('admin', 'user')",
            name=op.f("ck_users_role"),
        ),
        sa.CheckConstraint(
            "status IN ('active', 'disabled')",
            name=op.f("ck_users_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("username", name="uq_users_username"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "user_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.LargeBinary(length=32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "expires_at > created_at",
            name=op.f("ck_user_sessions_expires_after_created"),
        ),
        sa.CheckConstraint(
            "revoked_at IS NULL OR revoked_at >= created_at",
            name=op.f("ck_user_sessions_revoked_after_created"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_sessions_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_sessions")),
        sa.UniqueConstraint("token_hash", name="uq_user_sessions_token_hash"),
    )
    op.create_index(
        "ix_user_sessions_user_id_expires_at_active",
        "user_sessions",
        ["user_id", "expires_at"],
        unique=False,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), server_default="新对话", nullable=False),
        sa.Column("status", sa.String(length=16), server_default="active", nullable=False),
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
            "status IN ('active', 'archived')",
            name=op.f("ck_conversations_status"),
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_conversations_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )
    op.create_index(
        "ix_conversations_user_id_updated_at_id",
        "conversations",
        ["user_id", "updated_at", "id"],
        unique=False,
    )

    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column(
            "model_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'system')",
            name=op.f("ck_messages_role"),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'completed', 'failed')",
            name=op.f("ck_messages_status"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "conversation_id",
            "id",
            name="uq_messages_conversation_id_id",
        ),
    )
    op.create_index(
        "ix_messages_conversation_id_created_at_id",
        "messages",
        ["conversation_id", "created_at", "id"],
        unique=False,
    )

    op.create_table(
        "rag_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("question_message_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("answer_message_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("client_request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=True),
        sa.Column(
            "retrieval_queries",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("llm_model", sa.String(length=128), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="running", nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "top_k BETWEEN 1 AND 10",
            name=op.f("ck_rag_runs_top_k"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled')",
            name=op.f("ck_rag_runs_status"),
        ),
        sa.CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name=op.f("ck_rag_runs_latency_ms"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_rag_runs_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "question_message_id"],
            ["messages.conversation_id", "messages.id"],
            name="fk_rag_runs_question_message_conversation_messages",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id", "answer_message_id"],
            ["messages.conversation_id", "messages.id"],
            name="fk_rag_runs_answer_message_conversation_messages",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_runs")),
        sa.UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_rag_runs_conversation_id_client_request_id",
        ),
    )

    op.create_table(
        "rag_references",
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.CheckConstraint("rank >= 1", name=op.f("ck_rag_references_rank")),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_rag_references_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["knowledge_chunks.id"],
            name="fk_rag_references_chunk_id_knowledge_chunks",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("rag_run_id", "rank", name=op.f("pk_rag_references")),
        sa.UniqueConstraint(
            "rag_run_id",
            "chunk_id",
            name="uq_rag_references_rag_run_id_chunk_id",
        ),
    )

    op.add_column(
        "knowledge_items",
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_knowledge_items_owner_id_users",
        "knowledge_items",
        "users",
        ["owner_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_knowledge_items_owner_id",
        "knowledge_items",
        ["owner_id"],
        unique=False,
    )


def downgrade() -> None:
    """严格按升级的逆序移除 M4 模式。"""
    op.drop_index("ix_knowledge_items_owner_id", table_name="knowledge_items")
    op.drop_constraint(
        "fk_knowledge_items_owner_id_users",
        "knowledge_items",
        type_="foreignkey",
    )
    op.drop_column("knowledge_items", "owner_id")
    op.drop_table("rag_references")
    op.drop_table("rag_runs")
    op.drop_index(
        "ix_messages_conversation_id_created_at_id",
        table_name="messages",
    )
    op.drop_table("messages")
    op.drop_index(
        "ix_conversations_user_id_updated_at_id",
        table_name="conversations",
    )
    op.drop_table("conversations")
    op.drop_index(
        "ix_user_sessions_user_id_expires_at_active",
        table_name="user_sessions",
    )
    op.drop_table("user_sessions")
    op.drop_table("users")
