"""RAG 运行和引用快照持久化模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.database.base import Base
from app.infrastructure.database.types import UTCDateTime


class RAGRun(Base):
    """一次可幂等重放的持久化 RAG 运行。"""

    __tablename__ = "rag_runs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_rag_runs_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "question_message_id"],
            ["messages.conversation_id", "messages.id"],
            name="fk_rag_runs_question_message_conversation_messages",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["conversation_id", "answer_message_id"],
            ["messages.conversation_id", "messages.id"],
            name="fk_rag_runs_answer_message_conversation_messages",
            ondelete="RESTRICT",
        ),
        UniqueConstraint(
            "conversation_id",
            "client_request_id",
            name="uq_rag_runs_conversation_id_client_request_id",
        ),
        CheckConstraint("top_k BETWEEN 1 AND 10", name="top_k"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled')",
            name="status",
        ),
        CheckConstraint("latency_ms IS NULL OR latency_ms >= 0", name="latency_ms"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    conversation_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    question_message_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    answer_message_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    client_request_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=False,
    )
    strategy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_queries: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=text("'[]'::jsonb"),
    )
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="running",
        server_default=text("'running'"),
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        nullable=False,
        server_default=func.now(),
    )


class RAGReference(Base):
    """一次 RAG 运行使用的不可变知识快照。"""

    __tablename__ = "rag_references"
    __table_args__ = (
        ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name="fk_rag_references_rag_run_id_rag_runs",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["chunk_id"],
            ["knowledge_chunks.id"],
            name="fk_rag_references_chunk_id_knowledge_chunks",
            ondelete="SET NULL",
        ),
        UniqueConstraint(
            "rag_run_id",
            "chunk_id",
            name="uq_rag_references_rag_run_id_chunk_id",
        ),
        CheckConstraint("rank >= 1", name="rank"),
    )

    rag_run_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        nullable=True,
    )
    score: Mapped[float] = mapped_column(Float, nullable=False)
    snapshot: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
