"""持久化聊天 API 的请求与响应模型。"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.chat import ChatResponse


class ChatV1Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    client_request_id: UUID
    question: str
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not 1 <= len(normalized) <= 8000:
            raise ValueError("question 长度必须在 1 到 8000 个字符之间")
        return normalized

    @field_validator("top_k", mode="before")
    @classmethod
    def reject_boolean_top_k(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("top_k 必须是整数")
        return value


class ChatV1Response(ChatResponse):
    conversation_id: UUID
    question_message_id: UUID
    answer_message_id: UUID
    rag_run_id: UUID
    client_request_id: UUID

