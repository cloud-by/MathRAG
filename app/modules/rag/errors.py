"""RAG 持久化状态机的稳定领域错误。"""

from __future__ import annotations

from app.core.errors import AppError


class RAGStateConflictError(RuntimeError):
    """数据库状态不满足预期的 CAS 前置条件。"""


class RAGRequestInProgressError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="RAG_REQUEST_IN_PROGRESS",
            message="相同请求仍在处理中。",
            status_code=409,
        )


class ConversationArchivedError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="CONVERSATION_ARCHIVED",
            message="已归档会话不能继续提问。",
            status_code=409,
        )


def persisted_rag_error(code: str, message: str) -> AppError:
    """从已持久化的稳定错误码重建公开异常。"""
    status_codes = {
        "EMBEDDING_UNAVAILABLE": 502,
        "LLM_UNAVAILABLE": 502,
        "LLM_RATE_LIMITED": 429,
        "RAG_UPSTREAM_TIMEOUT": 504,
        "RAG_CANCELLED": 409,
        "DATABASE_UNAVAILABLE": 503,
        "INTERNAL_ERROR": 500,
    }
    return AppError(
        code=code if code in status_codes else "INTERNAL_ERROR",
        message=message or "请求处理失败。",
        status_code=status_codes.get(code, 500),
    )

