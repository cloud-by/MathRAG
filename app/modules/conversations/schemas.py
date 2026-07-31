"""会话与消息公开 API 模型。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="新对话", min_length=1, max_length=255)

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> str:
        return " ".join(str(value or "").split())


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=255)
    status: Literal["active", "archived"] | None = None

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, value: object) -> str | None:
        if value is None:
            return None
        return " ".join(str(value).split())


class ConversationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    title: str
    status: Literal["active", "archived"]
    created_at: datetime
    updated_at: datetime


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    conversation_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    status: Literal["pending", "completed", "failed"]
    model_metadata: dict[str, object]
    created_at: datetime


class ConversationPage(BaseModel):
    items: list[ConversationRead]
    page: int
    page_size: int
    total: int


class MessagePage(BaseModel):
    items: list[MessageRead]
    page: int
    page_size: int
    total: int
