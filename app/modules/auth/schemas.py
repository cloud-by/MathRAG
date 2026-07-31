"""认证 API 请求模型。"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AuthUserRead(BaseModel):
    """认证端点冻结的最小用户资源。"""

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    username: str
    email: str | None
    role: str
    status: str


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("username 不能为空")
        return normalized
