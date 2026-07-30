"""知识条目的持久化访问封装。"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem


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
