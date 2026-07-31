"""会话领域稳定错误。"""

from __future__ import annotations

from app.core.errors import AppError


class ConversationNotFoundError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="CONVERSATION_NOT_FOUND",
            message="会话不存在。",
            status_code=404,
        )
