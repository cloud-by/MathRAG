"""知识条目及其检索分块的持久化模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.infrastructure.database.base import Base
from app.infrastructure.database.types import UTCDateTime


class KnowledgeItem(Base):
    """可检索的知识条目。"""

    __tablename__ = "knowledge_items"
    __table_args__ = (
        UniqueConstraint("legacy_id", name="uq_knowledge_items_legacy_id"),
        CheckConstraint("difficulty IN ('easy', 'medium', 'hard')", name="difficulty"),
        CheckConstraint("visibility IN ('public', 'private')", name="visibility"),
        CheckConstraint(
            "status IN ('draft', 'indexing', 'ready', 'failed', 'archived')",
            name="status",
        ),
        CheckConstraint("revision > 0", name="revision"),
        Index("ix_knowledge_items_visibility_status", "visibility", "status"),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    legacy_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    keywords: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    example: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default=text("''"),
    )
    steps: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    difficulty: Mapped[str] = mapped_column(String(16), nullable=False)
    visibility: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="public",
        server_default=text("'public'"),
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="indexing",
        server_default=text("'indexing'"),
        index=True,
    )
    revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="knowledge_item",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class KnowledgeChunk(Base):
    """知识条目的独立检索分块。"""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["knowledge_item_id"],
            ["knowledge_items.id"],
            name="fk_knowledge_chunks_knowledge_item_id_knowledge_items",
            ondelete="CASCADE",
        ),
        CheckConstraint("chunk_index >= 0", name="chunk_index"),
        CheckConstraint("status IN ('pending', 'ready', 'failed')", name="status"),
        CheckConstraint(
            "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)",
            name="ready_requires_embedding",
        ),
        UniqueConstraint(
            "knowledge_item_id",
            "chunk_index",
            name="uq_knowledge_chunks_knowledge_item_id_chunk_index",
        ),
        Index("ix_knowledge_chunks_status_embedding_model", "status", "embedding_model"),
    )

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    knowledge_item_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_context: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    metadata_: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
        server_default=text("'pending'"),
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        server_default=func.now(),
    )

    knowledge_item: Mapped[KnowledgeItem] = relationship(back_populates="chunks")
