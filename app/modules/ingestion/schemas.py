"""文档与导入任务 API 的公开数据契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentRead(BaseModel):
    """不暴露受控存储路径的文档表示。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    id: UUID
    owner_id: UUID | None
    original_name: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["pending", "processing", "ready", "failed", "archived"]
    created_at: datetime
    updated_at: datetime


class DocumentPage(BaseModel):
    """文档分页响应。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    items: list[DocumentRead]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)


class IngestionJobRead(BaseModel):
    """不暴露重试载荷的导入任务表示。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    id: UUID
    requested_by: UUID | None
    document_id: UUID | None
    job_type: Literal["text", "pdf", "web", "reindex"]
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    progress: int = Field(ge=0, le=100)
    attempt_count: int = Field(ge=0)
    error_code: str | None = Field(default=None, max_length=64)
    error_message: str | None = Field(default=None, max_length=500)
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class IngestionJobPage(BaseModel):
    """导入任务偏移分页响应。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    items: list[IngestionJobRead]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)


class DocumentAccepted(BaseModel):
    """上传成功后返回的文档及其待执行任务。"""

    model_config = ConfigDict(extra="forbid", from_attributes=True, frozen=True)

    document: DocumentRead
    job: IngestionJobRead
