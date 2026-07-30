"""知识条目的持久化访问封装。"""

from __future__ import annotations

import math
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.knowledge.errors import KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.search import KnowledgeSearchHit, search_hit_from_row


class KnowledgeRepository:
    """只封装知识查询和挂载，事务由调用方管理。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_legacy_id(self, legacy_id: str) -> KnowledgeItem | None:
        """按历史标识获取条目及其已加载的分块。"""
        statement = (
            select(KnowledgeItem)
            .options(selectinload(KnowledgeItem.chunks))
            .where(KnowledgeItem.legacy_id == legacy_id)
        )
        result = await self._session.execute(statement)
        return result.scalar_one_or_none()

    def add(self, item: KnowledgeItem) -> None:
        """将知识条目交由当前会话持久化。"""
        self._session.add(item)

    async def count_legacy_items(self) -> int:
        """统计带有历史标识的知识条目。"""
        statement = select(func.count()).select_from(KnowledgeItem).where(
            KnowledgeItem.legacy_id.is_not(None)
        )
        result = await self._session.execute(statement)
        return result.scalar_one()

    async def count_legacy_chunks(self) -> int:
        """统计归属带有历史标识条目的知识分块。"""
        statement = (
            select(func.count())
            .select_from(KnowledgeChunk)
            .join(KnowledgeItem)
            .where(KnowledgeItem.legacy_id.is_not(None))
        )
        result = await self._session.execute(statement)
        return result.scalar_one()

    async def list_legacy_items_ordered(self) -> list[KnowledgeItem]:
        """按历史标识升序列出条目及其已加载的分块。"""
        statement = (
            select(KnowledgeItem)
            .options(selectinload(KnowledgeItem.chunks))
            .where(KnowledgeItem.legacy_id.is_not(None))
            .order_by(KnowledgeItem.legacy_id.asc())
        )
        result = await self._session.execute(statement)
        return list(result.scalars().unique())

    async def search_ready_chunks(
        self,
        *,
        query_vector: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> list[KnowledgeSearchHit]:
        """精确检索当前模型下公开且就绪的知识分块。"""
        vector = _validated_query_vector(query_vector)
        if type(embedding_model) is not str or not embedding_model.strip():
            raise KnowledgeSearchError("embedding_model 必须是非空字符串")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise KnowledgeSearchError("limit 必须是 1 到 10 的整数")

        current_model = embedding_model.strip()
        distance = KnowledgeChunk.embedding.cosine_distance(vector).label("distance")
        statement = (
            select(KnowledgeChunk, KnowledgeItem, distance)
            .join(KnowledgeItem)
            .where(
                KnowledgeItem.status == "ready",
                KnowledgeItem.visibility == "public",
                KnowledgeChunk.status == "ready",
                KnowledgeChunk.embedding.is_not(None),
                KnowledgeChunk.embedding_model == current_model,
            )
            .order_by(distance.asc(), KnowledgeChunk.id.asc())
            .limit(limit)
        )
        result = await self._session.execute(statement)
        return [
            search_hit_from_row(chunk, item, row_distance)
            for chunk, item, row_distance in result.all()
        ]


def _validated_query_vector(query_vector: Sequence[float]) -> list[float]:
    """验证固定 1024 维且所有元素有限的查询向量。"""
    if isinstance(query_vector, (str, bytes)):
        raise KnowledgeSearchError("query_vector 必须是 1024 维有限向量")
    try:
        vector = [float(value) for value in query_vector]
    except Exception:
        raise KnowledgeSearchError("query_vector 必须是 1024 维有限向量") from None
    if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
        raise KnowledgeSearchError("query_vector 必须是 1024 维有限向量")
    return vector
