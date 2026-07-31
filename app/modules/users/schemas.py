"""用户公开数据传输模型。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UserRead(BaseModel):
    """不包含密码或 Session 摘要的用户公开视图。"""

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID
    username: str
    email: str | None
    role: str
    status: str
    created_at: datetime
    updated_at: datetime
