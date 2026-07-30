"""旧知识迁移的领域异常。"""


class LegacyKnowledgeImportError(Exception):
    """旧知识导入过程中发生的基础异常。"""


class LegacyKnowledgeInputError(LegacyKnowledgeImportError):
    """旧知识输入不符合迁移契约。"""


class DuplicateLegacyIdError(LegacyKnowledgeInputError):
    """旧知识输入中出现重复的条目 ID。"""


class LegacyKnowledgeConflictError(LegacyKnowledgeImportError):
    """旧知识与已持久化数据发生不可自动解决的冲突。"""
