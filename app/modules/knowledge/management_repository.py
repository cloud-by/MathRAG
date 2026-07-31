"""权限条件内嵌 SQL 的知识管理读取仓储。"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import and_, func, select, true
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.models import KnowledgeItem


def _visibility_predicate(
    principal: AuthenticatedPrincipal,
) -> ColumnElement[bool]:
    """生成不可被调用方筛选参数放宽的基础可见性条件。"""
    if principal.role == "admin":
        return true()
    return and_(
        KnowledgeItem.visibility == "public",
        KnowledgeItem.status == "ready",
    )


class KnowledgeManagementRepository:
    """只操作调用方会话，不拥有事务生命周期。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_visible(
        self,
        item_id: UUID,
        principal: AuthenticatedPrincipal,
    ) -> KnowledgeItem | None:
        return await self._session.scalar(
            select(KnowledgeItem).where(
                KnowledgeItem.id == item_id,
                _visibility_predicate(principal),
            )
        )

    async def list_visible(
        self,
        principal: AuthenticatedPrincipal,
        *,
        status: str | None = None,
        visibility: str | None = None,
        category: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[KnowledgeItem], int]:
        filters = [_visibility_predicate(principal)]
        if status is not None:
            filters.append(KnowledgeItem.status == status)
        if visibility is not None:
            filters.append(KnowledgeItem.visibility == visibility)
        if category is not None:
            filters.append(KnowledgeItem.category == category)

        items = list(
            (
                await self._session.scalars(
                    select(KnowledgeItem)
                    .where(*filters)
                    .order_by(KnowledgeItem.updated_at.desc(), KnowledgeItem.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        )
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(KnowledgeItem).where(*filters)
            )
            or 0
        )
        return items, total
