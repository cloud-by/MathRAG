"""增加向量就绪约束与普通过滤索引。"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "0003_enforce_vector_readiness"
down_revision: str | None = "0002_create_knowledge_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """约束就绪分块必须携带向量，并增加常用过滤索引。"""
    op.create_check_constraint(
        op.f("ck_knowledge_chunks_ready_requires_embedding"),
        "knowledge_chunks",
        "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)",
    )
    op.create_index(
        "ix_knowledge_items_visibility_status",
        "knowledge_items",
        ["visibility", "status"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_chunks_status_embedding_model",
        "knowledge_chunks",
        ["status", "embedding_model"],
        unique=False,
    )


def downgrade() -> None:
    """移除普通过滤索引与向量就绪约束。"""
    op.drop_index(
        "ix_knowledge_chunks_status_embedding_model",
        table_name="knowledge_chunks",
    )
    op.drop_index(
        "ix_knowledge_items_visibility_status",
        table_name="knowledge_items",
    )
    op.drop_constraint(
        op.f("ck_knowledge_chunks_ready_requires_embedding"),
        "knowledge_chunks",
        type_="check",
    )
