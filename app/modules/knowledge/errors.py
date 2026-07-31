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


class KnowledgeRevisionConflictError(AppError):
    """知识条目 revision 已被其他写入推进。"""

    def __init__(self) -> None:
        super().__init__(
            code="KNOWLEDGE_REVISION_CONFLICT",
            message="知识点已被其他用户更新，请刷新后重试。",
            status_code=409,
        )


def map_knowledge_embedding_error(_exc: Exception) -> AppError:
    """把 Provider 细节折叠为稳定且不泄密的公开错误。"""
    return AppError(
        code="EMBEDDING_UNAVAILABLE",
        message="知识向量化服务暂时不可用。",
        status_code=502,
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
