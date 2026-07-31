"""知识管理读取用例。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import KnowledgeNotFoundError
from app.modules.knowledge.management_repository import KnowledgeManagementRepository
from app.modules.knowledge.management_schemas import KnowledgeItemPage, KnowledgeItemRead
from app.modules.knowledge.models import KnowledgeItem


class KnowledgeManagementRepositoryProtocol(Protocol):
    async def get_visible(
        self,
        item_id: UUID,
        principal: AuthenticatedPrincipal,
    ) -> KnowledgeItem | None: ...

    async def list_visible(
        self,
        principal: AuthenticatedPrincipal,
        *,
        status: str | None,
        visibility: str | None,
        category: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[KnowledgeItem], int]: ...


class KnowledgeManagementService:
    """将权限感知 ORM 查询转换成安全公开 DTO。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        repository_factory: Callable[
            [AsyncSession], KnowledgeManagementRepositoryProtocol
        ] = KnowledgeManagementRepository,
    ) -> None:
        self._session_factory = session_factory
        self._repository_factory = repository_factory

    async def get(
        self,
        item_id: UUID,
        principal: AuthenticatedPrincipal,
    ) -> KnowledgeItemRead:
        async with self._session_factory() as session:
            item = await self._repository_factory(session).get_visible(item_id, principal)
        if item is None:
            raise KnowledgeNotFoundError()
        return KnowledgeItemRead.model_validate(item)

    async def list(
        self,
        principal: AuthenticatedPrincipal,
        *,
        status: str | None = None,
        visibility: str | None = None,
        category: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> KnowledgeItemPage:
        if (
            type(page) is not int
            or page < 1
            or type(page_size) is not int
            or not 1 <= page_size <= 100
        ):
            raise AppError(
                code="REQUEST_VALIDATION_FAILED",
                message="分页参数无效。",
                status_code=422,
            )
        async with self._session_factory() as session:
            items, total = await self._repository_factory(session).list_visible(
                principal,
                status=status,
                visibility=visibility,
                category=category,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
        return KnowledgeItemPage(
            items=[KnowledgeItemRead.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
            total=total,
        )
