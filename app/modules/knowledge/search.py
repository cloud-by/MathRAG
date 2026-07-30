"""知识向量检索的稳定 DTO 与结果合并规则。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID

from app.modules.knowledge.errors import KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem


@dataclass(frozen=True)
class KnowledgeSearchHit:
    """不依赖 ORM 会话生命周期的单条知识检索结果。"""

    database_chunk_id: UUID
    legacy_chunk_id: str
    source_id: str
    category: str
    title: str
    keywords: tuple[str, ...]
    content: str
    example: str
    steps: tuple[str, ...]
    difficulty: str
    answer_context: str
    retrieval_text: str
    source_line: int | None
    metadata: dict[str, object]
    distance: float

    def __post_init__(self) -> None:
        """复制可变载荷，并拒绝不能安全排序或引用的结果。"""
        chunk_id = self.database_chunk_id
        if not isinstance(chunk_id, UUID):
            raise KnowledgeSearchError("知识检索结果缺少有效的数据库分块 UUID")
        if type(self.keywords) is not tuple or not all(
            type(keyword) is str for keyword in self.keywords
        ):
            raise _row_error(chunk_id, "keywords 类型无效")
        if type(self.steps) is not tuple or not all(
            type(step) is str for step in self.steps
        ):
            raise _row_error(chunk_id, "steps 类型无效")
        if type(self.metadata) is not dict:
            raise _row_error(chunk_id, "metadata 类型无效")
        if self.source_line is not None and (
            type(self.source_line) is not int or self.source_line < 1
        ):
            raise _row_error(chunk_id, "source_line 类型无效")
        if type(self.legacy_chunk_id) is not str or not self.legacy_chunk_id:
            raise _row_error(chunk_id, "legacy_chunk_id 无效")
        if type(self.source_id) is not str or not self.source_id:
            raise _row_error(chunk_id, "source_id 无效")

        try:
            distance = float(self.distance)
            metadata = deepcopy(self.metadata)
        except Exception:
            raise _row_error(chunk_id, "结果载荷无法复制或转换") from None
        if not math.isfinite(distance):
            raise _row_error(chunk_id, "distance 必须是有限数值")
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "metadata", metadata)

    @property
    def score(self) -> float:
        """返回与旧检索接口一致的相似度分数。"""
        return 1.0 - self.distance

    def to_reference(self, rank: int) -> dict[str, object]:
        """映射为现有回答链消费的引用字典。"""
        if type(rank) is not int or rank < 1:
            raise KnowledgeSearchError("知识引用 rank 必须是大于等于 1 的整数")
        try:
            metadata = deepcopy(self.metadata)
        except Exception:
            raise _row_error(self.database_chunk_id, "metadata 无法复制") from None
        return {
            "rank": rank,
            "score": self.score,
            "index": None,
            "chunk_id": self.legacy_chunk_id,
            "source_id": self.source_id,
            "category": self.category,
            "title": self.title,
            "keywords": list(self.keywords),
            "content": self.content,
            "example": self.example,
            "steps": list(self.steps),
            "difficulty": self.difficulty,
            "answer_context": self.answer_context,
            "retrieval_text": self.retrieval_text,
            "source_line": self.source_line,
            "metadata": metadata,
        }


def merge_search_hits(
    groups: Sequence[Sequence[KnowledgeSearchHit]],
    top_k: int,
) -> list[KnowledgeSearchHit]:
    """合并多次检索结果，按数据库分块去重并稳定截取 Top-K。"""
    if type(top_k) is not int or top_k <= 0:
        raise KnowledgeSearchError("top_k 必须是大于 0 的整数")

    best_by_chunk: dict[UUID, KnowledgeSearchHit] = {}
    for group in groups:
        for hit in group:
            if not isinstance(hit, KnowledgeSearchHit):
                raise KnowledgeSearchError("检索结果组包含无效命中")
            current = best_by_chunk.get(hit.database_chunk_id)
            if current is None or hit.score > current.score:
                best_by_chunk[hit.database_chunk_id] = hit

    ordered = sorted(
        best_by_chunk.values(),
        key=lambda hit: (-hit.score, str(hit.database_chunk_id)),
    )
    return ordered[:top_k]


def search_hit_from_row(
    chunk: KnowledgeChunk,
    item: KnowledgeItem,
    distance: object,
) -> KnowledgeSearchHit:
    """严格地把数据库 chunk/item/distance 行转换为脱离会话的 DTO。"""
    chunk_id = getattr(chunk, "id", None)
    if not isinstance(chunk_id, UUID):
        raise KnowledgeSearchError("知识检索行缺少有效的数据库分块 UUID")

    try:
        numeric_distance = float(distance)
    except Exception:
        raise _row_error(chunk_id, "distance 无法转换") from None
    if not math.isfinite(numeric_distance):
        raise _row_error(chunk_id, "distance 必须是有限数值")

    metadata_value = getattr(chunk, "metadata_", None)
    if type(metadata_value) is not dict:
        raise _row_error(chunk_id, "metadata 类型无效")
    try:
        metadata = deepcopy(metadata_value)
    except Exception:
        raise _row_error(chunk_id, "metadata 无法复制") from None

    legacy_chunk_id = metadata.pop("legacy_chunk_id", None)
    metadata_source_id = metadata.pop("legacy_source_id", None)
    source_line = metadata.pop("source_line", None)
    if type(legacy_chunk_id) is not str or not legacy_chunk_id:
        raise _row_error(chunk_id, "legacy_chunk_id 无效")
    if type(metadata_source_id) is not str or not metadata_source_id:
        raise _row_error(chunk_id, "legacy_source_id 无效")

    model_source_id = getattr(item, "legacy_id", None)
    if model_source_id is not None and (
        type(model_source_id) is not str or not model_source_id
    ):
        raise _row_error(chunk_id, "模型 legacy_id 无效")
    source_id = model_source_id or metadata_source_id

    keywords = _string_list_from_model(
        getattr(item, "keywords", None),
        field_name="keywords",
        chunk_id=chunk_id,
    )
    steps = _string_list_from_model(
        getattr(item, "steps", None),
        field_name="steps",
        chunk_id=chunk_id,
    )
    if source_line is not None and (type(source_line) is not int or source_line < 1):
        raise _row_error(chunk_id, "source_line 类型无效")

    return KnowledgeSearchHit(
        database_chunk_id=chunk_id,
        legacy_chunk_id=legacy_chunk_id,
        source_id=source_id,
        category=_string_from_model(item, "category", chunk_id),
        title=_string_from_model(item, "title", chunk_id),
        keywords=keywords,
        content=_string_from_model(item, "content", chunk_id),
        example=_string_from_model(item, "example", chunk_id),
        steps=steps,
        difficulty=_string_from_model(item, "difficulty", chunk_id),
        answer_context=_string_from_model(chunk, "answer_context", chunk_id),
        retrieval_text=_string_from_model(chunk, "retrieval_text", chunk_id),
        source_line=source_line,
        metadata=metadata,
        distance=numeric_distance,
    )


def _string_list_from_model(
    value: object,
    *,
    field_name: str,
    chunk_id: UUID,
) -> tuple[str, ...]:
    """严格读取 JSON 字符串数组，不接受其他可迭代类型。"""
    if type(value) is not list or not all(type(entry) is str for entry in value):
        raise _row_error(chunk_id, f"{field_name} 类型无效")
    return tuple(value)


def _string_from_model(model: object, field_name: str, chunk_id: UUID) -> str:
    """安全读取模型字符串列，不在异常中包含列值。"""
    value = getattr(model, field_name, None)
    if type(value) is not str:
        raise _row_error(chunk_id, f"{field_name} 类型无效")
    return value


def _row_error(chunk_id: UUID, reason: str) -> KnowledgeSearchError:
    """构造只携带数据库 UUID、绝不回显载荷值的领域错误。"""
    return KnowledgeSearchError(
        f"知识检索行无效 (database_chunk_id={chunk_id}): {reason}"
    )
