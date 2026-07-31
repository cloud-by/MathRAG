"""知识管理读取用例。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError, ConfigurationError
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    validate_and_normalize_vector,
)
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    KnowledgeNotFoundError,
    KnowledgeRevisionConflictError,
    map_knowledge_embedding_error,
)
from app.modules.knowledge.management_repository import (
    IndexingSnapshot,
    KnowledgeManagementRepository,
)
from app.modules.knowledge.management_schemas import (
    KnowledgeItemCreate,
    KnowledgeItemPage,
    KnowledgeItemRead,
    KnowledgeItemUpdate,
)
from app.modules.knowledge.models import KnowledgeItem


class KnowledgeManagementRepositoryProtocol(Protocol):
    async def create_indexing(
        self,
        *,
        owner_id: UUID,
        values: Mapping[str, object],
    ) -> IndexingSnapshot: ...

    async def update_with_revision(
        self,
        item_id: UUID,
        *,
        expected_revision: int,
        values: Mapping[str, object],
        reindex: bool,
    ) -> IndexingSnapshot | KnowledgeItem | None: ...

    async def archive_with_revision(
        self,
        item_id: UUID,
        expected_revision: int,
    ) -> bool: ...

    async def complete_indexing(
        self,
        snapshot: IndexingSnapshot,
        vector: Sequence[float],
        model: str,
    ) -> KnowledgeItem | None: ...

    async def fail_indexing(self, snapshot: IndexingSnapshot) -> None: ...

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
        provider: EmbeddingProvider | None = None,
        *,
        repository_factory: Callable[
            [AsyncSession], KnowledgeManagementRepositoryProtocol
        ] = KnowledgeManagementRepository,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._repository_factory = repository_factory

    async def create(
        self,
        owner_id: UUID,
        payload: KnowledgeItemCreate | Mapping[str, object],
    ) -> KnowledgeItemRead:
        """先提交 indexing 快照，再在会话外向量化并完成写回。"""
        request = KnowledgeItemCreate.model_validate(payload)
        async with self._session_factory() as session:
            async with session.begin():
                snapshot = await self._repository_factory(session).create_indexing(
                    owner_id=owner_id,
                    values=request.model_dump(),
                )
        return await self._embed_and_finalize(snapshot)

    async def update(
        self,
        item_id: UUID,
        payload: KnowledgeItemUpdate | Mapping[str, object],
    ) -> KnowledgeItemRead:
        """用 revision CAS 更新；内容变化才执行事务外向量化。"""
        request = KnowledgeItemUpdate.model_validate(payload)
        values = request.model_dump(exclude={"revision"}, exclude_unset=True)
        reindex = bool(
            set(values)
            & {
                "category",
                "title",
                "keywords",
                "content",
                "example",
                "steps",
                "difficulty",
            }
        )
        async with self._session_factory() as session:
            async with session.begin():
                updated = await self._repository_factory(
                    session
                ).update_with_revision(
                    item_id,
                    expected_revision=request.revision,
                    values=values,
                    reindex=reindex,
                )
        if updated is None:
            raise KnowledgeNotFoundError()
        if isinstance(updated, IndexingSnapshot):
            return await self._embed_and_finalize(updated)
        return KnowledgeItemRead.model_validate(updated)

    async def archive(self, item_id: UUID, expected_revision: int) -> None:
        """原子归档知识条目；仅真正不存在的 ID 返回 404。"""
        async with self._session_factory() as session:
            async with session.begin():
                archived = await self._repository_factory(
                    session
                ).archive_with_revision(item_id, expected_revision)
        if not archived:
            raise KnowledgeNotFoundError()

    async def _embed_and_finalize(
        self,
        snapshot: IndexingSnapshot,
    ) -> KnowledgeItemRead:
        """确保 Provider 调用发生在两个完全关闭的 Session 之间。"""
        provider = self._provider
        if provider is None:
            raise ConfigurationError("Embedding Provider 未配置")
        try:
            vectors = await provider.embed_texts([snapshot.retrieval_text])
            if len(vectors) != 1:
                raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
            vector = validate_and_normalize_vector(
                vectors[0],
                provider.dimensions,
            )
            model = provider.model
        except (
            EmbeddingInputError,
            EmbeddingResponseError,
            EmbeddingUnavailableError,
        ) as exc:
            await self._mark_failed(snapshot)
            raise map_knowledge_embedding_error(exc) from None

        async with self._session_factory() as session:
            async with session.begin():
                item = await self._repository_factory(session).complete_indexing(
                    snapshot,
                    vector,
                    model,
                )
        if item is None:
            raise KnowledgeRevisionConflictError()
        return KnowledgeItemRead.model_validate(item)

    async def _mark_failed(self, snapshot: IndexingSnapshot) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                await self._repository_factory(session).fail_indexing(snapshot)

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
