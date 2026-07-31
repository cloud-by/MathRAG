"""旧知识迁移的领域异常。"""

from __future__ import annotations

from app.core.errors import AppError


class KnowledgeNotFoundError(AppError):
    """知识条目不存在或对当前身份不可见。"""

    def __init__(self) -> None:
        super().__init__(
            code="KNOWLEDGE_NOT_FOUND",
            message="知识条目不存在。",
            status_code=404,
        )


class KnowledgeSearchError(Exception):
    """知识向量化或检索失败。"""


class EmbeddingInputError(KnowledgeSearchError, ValueError):
    """待向量化文本或配置不满足固定契约。"""


class EmbeddingResponseError(KnowledgeSearchError):
    """Embedding Provider 返回无效结果。"""


class EmbeddingUnavailableError(KnowledgeSearchError):
    """Embedding Provider 暂时不可用。"""


class LegacyKnowledgeImportError(Exception):
    """旧知识导入过程中发生的基础异常。"""


class LegacyKnowledgeInputError(LegacyKnowledgeImportError):
    """旧知识输入不符合迁移契约。"""


class DuplicateLegacyIdError(LegacyKnowledgeInputError):
    """旧知识输入中出现重复的条目 ID。"""


class LegacyKnowledgeConflictError(LegacyKnowledgeImportError):
    """旧知识与已持久化数据发生不可自动解决的冲突。"""
