"""知识管理 API 的公开数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _KnowledgeItemValues(BaseModel):
    """创建和更新共用的可编辑知识字段。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: str | None = Field(default=None, min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    keywords: list[str] | None = None
    content: str | None = Field(default=None, min_length=1)
    example: str | None = None
    steps: list[str] | None = None
    difficulty: Literal["easy", "medium", "hard"] | None = None
    visibility: Literal["public", "private"] | None = None

    @model_validator(mode="after")
    def normalize_collections(self) -> _KnowledgeItemValues:
        for field_name in ("keywords", "steps"):
            values = getattr(self, field_name)
            if values is not None:
                normalized = list(dict.fromkeys(value for value in values if value))
                setattr(self, field_name, normalized)
        return self


class KnowledgeItemCreate(_KnowledgeItemValues):
    """管理员创建知识条目的请求模型。"""

    category: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    keywords: list[str]
    content: str = Field(min_length=1)
    example: str = ""
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    visibility: Literal["public", "private"] = "public"

    @model_validator(mode="after")
    def require_nonempty_collections(self) -> KnowledgeItemCreate:
        if not self.keywords:
            raise ValueError("keywords 不能为空")
        if not self.steps:
            raise ValueError("steps 不能为空")
        return self


class KnowledgeItemUpdate(_KnowledgeItemValues):
    """管理员更新知识条目的请求模型。"""

    revision: int = Field(ge=1)


class KnowledgeItemRead(BaseModel):
    """知识条目的安全公开表示。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    id: UUID
    legacy_id: str | None
    owner_id: UUID | None
    category: str
    title: str
    keywords: list[str]
    content: str
    example: str
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    visibility: Literal["public", "private"]
    status: Literal["draft", "indexing", "ready", "failed", "archived"]
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class KnowledgeItemPage(BaseModel):
    """知识条目分页响应。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    items: list[KnowledgeItemRead]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
