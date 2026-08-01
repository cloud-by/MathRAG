"""用户公开数据传输模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.modules.users.types import UserRole, UserStatus


class UserRead(BaseModel):
    """不包含密码或 Session 摘要的用户公开视图。"""

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    username: str
    email: str | None
    role: UserRole
    status: UserStatus
    created_by_user_id: UUID | None
    must_change_password: bool
    created_at: datetime
    updated_at: datetime


class ManagedUserRead(UserRead):
    """包含创建者显示名的账号管理视图。"""

    created_by_username: str | None


class UserPage(BaseModel):
    """用户管理分页结果。"""

    model_config = ConfigDict(frozen=True)

    items: list[ManagedUserRead]
    page: int
    page_size: int
    total: int


class UserCreate(BaseModel):
    """管理员或教师创建账号的输入。"""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=64)
    email: str | None = Field(default=None, max_length=320)
    password: str = Field(min_length=12, max_length=128)
    role: UserRole = "student"


class UserUpdate(BaseModel):
    """账号管理的部分更新输入。"""

    model_config = ConfigDict(extra="forbid")

    username: str | None = Field(default=None, min_length=1, max_length=64)
    email: str | None = Field(default=None, max_length=320)
    role: UserRole | None = None
    status: UserStatus | None = None

    @model_validator(mode="after")
    def require_change(self) -> "UserUpdate":
        if not self.model_fields_set:
            raise ValueError("至少提供一个需要修改的字段")
        return self


class UserPasswordReset(BaseModel):
    """管理员或教师重置临时密码的输入。"""

    model_config = ConfigDict(extra="forbid")

    password: str = Field(min_length=12, max_length=128)
