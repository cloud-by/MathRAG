"""权限条件内嵌 SQL 的知识管理读取仓储。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from sqlalchemy import and_, func, select, true, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import KnowledgeRevisionConflictError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.rendering import build_answer_context, build_retrieval_text


@dataclass(frozen=True)
class IndexingSnapshot:
    """跨事务传递的纯标量向量化快照。"""

    item_id: UUID
    revision: int
    chunk_id: UUID
    retrieval_text: str
    answer_context: str


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

    async def create_indexing(
        self,
        *,
        owner_id: UUID,
        values: Mapping[str, object],
    ) -> IndexingSnapshot:
        """创建待向量化条目与唯一分块，但不拥有事务提交。"""
        item = KnowledgeItem(
            owner_id=owner_id,
            status="indexing",
            revision=1,
            **dict(values),
        )
        self._session.add(item)
        await self._session.flush()

        retrieval_text = build_retrieval_text(values)
        answer_context = build_answer_context(values)
        chunk = KnowledgeChunk(
            id=uuid4(),
            knowledge_item_id=item.id,
            chunk_index=0,
            retrieval_text=retrieval_text,
            answer_context=answer_context,
            embedding=None,
            embedding_model=None,
            metadata_={},
            status="pending",
        )
        self._session.add(chunk)
        await self._session.flush()
        return IndexingSnapshot(
            item_id=item.id,
            revision=item.revision,
            chunk_id=chunk.id,
            retrieval_text=retrieval_text,
            answer_context=answer_context,
        )

    async def update_with_revision(
        self,
        item_id: UUID,
        *,
        expected_revision: int,
        values: Mapping[str, object],
        reindex: bool,
    ) -> IndexingSnapshot | KnowledgeItem | None:
        """以 revision 和非归档状态为 CAS 条件更新条目。"""
        assignments: dict[str, object] = dict(values)
        assignments["revision"] = KnowledgeItem.revision + 1
        assignments["updated_at"] = func.now()
        if reindex:
            assignments["status"] = "indexing"

        result = await self._session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id == item_id,
                KnowledgeItem.revision == expected_revision,
                KnowledgeItem.status != "archived",
            )
            .values(**assignments)
        )
        if result.rowcount != 1:
            await self._raise_if_item_exists(item_id)
            return None

        item = await self._session.scalar(
            select(KnowledgeItem).where(KnowledgeItem.id == item_id)
        )
        if item is None:
            return None
        if not reindex:
            return item

        rendered_values = {
            "category": item.category,
            "title": item.title,
            "keywords": item.keywords,
            "content": item.content,
            "example": item.example,
            "steps": item.steps,
            "difficulty": item.difficulty,
        }
        retrieval_text = build_retrieval_text(rendered_values)
        answer_context = build_answer_context(rendered_values)
        chunk = await self._session.scalar(
            select(KnowledgeChunk)
            .where(
                KnowledgeChunk.knowledge_item_id == item_id,
                KnowledgeChunk.chunk_index == 0,
            )
            .with_for_update()
        )
        if chunk is None:
            chunk = KnowledgeChunk(
                id=uuid4(),
                knowledge_item_id=item_id,
                chunk_index=0,
                retrieval_text=retrieval_text,
                answer_context=answer_context,
                metadata_={},
                status="pending",
            )
            self._session.add(chunk)
        else:
            chunk.retrieval_text = retrieval_text
            chunk.answer_context = answer_context
            chunk.embedding = None
            chunk.embedding_model = None
            chunk.status = "pending"
        await self._session.flush()
        return IndexingSnapshot(
            item_id=item.id,
            revision=item.revision,
            chunk_id=chunk.id,
            retrieval_text=retrieval_text,
            answer_context=answer_context,
        )

    async def archive_with_revision(
        self,
        item_id: UUID,
        expected_revision: int,
    ) -> bool:
        """以单条 UPDATE 原子归档并推进 revision。"""
        result = await self._session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id == item_id,
                KnowledgeItem.revision == expected_revision,
                KnowledgeItem.status != "archived",
            )
            .values(
                status="archived",
                revision=KnowledgeItem.revision + 1,
                updated_at=func.now(),
            )
        )
        if result.rowcount == 1:
            return True
        await self._raise_if_item_exists(item_id)
        return False

    async def complete_indexing(
        self,
        snapshot: IndexingSnapshot,
        vector: Sequence[float],
        model: str,
    ) -> KnowledgeItem | None:
        """仅在条目 revision 与分块身份仍匹配时完成向量写回。"""
        item = await self._lock_snapshot_item(snapshot)
        if item is None:
            return None
        result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                KnowledgeChunk.id == snapshot.chunk_id,
                KnowledgeChunk.knowledge_item_id == snapshot.item_id,
                KnowledgeChunk.status == "pending",
                KnowledgeChunk.retrieval_text == snapshot.retrieval_text,
                KnowledgeChunk.answer_context == snapshot.answer_context,
            )
            .values(
                embedding=list(vector),
                embedding_model=model,
                status="ready",
            )
        )
        if result.rowcount != 1:
            return None
        item.status = "ready"
        await self._session.flush()
        return item

    async def fail_indexing(self, snapshot: IndexingSnapshot) -> None:
        """仅把仍属于该快照的处理中条目和分块标记为失败。"""
        item = await self._lock_snapshot_item(snapshot)
        if item is None:
            return
        result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                KnowledgeChunk.id == snapshot.chunk_id,
                KnowledgeChunk.knowledge_item_id == snapshot.item_id,
                KnowledgeChunk.status == "pending",
                KnowledgeChunk.retrieval_text == snapshot.retrieval_text,
                KnowledgeChunk.answer_context == snapshot.answer_context,
            )
            .values(status="failed", embedding=None, embedding_model=None)
        )
        if result.rowcount != 1:
            return
        item.status = "failed"
        await self._session.flush()

    async def _lock_snapshot_item(
        self,
        snapshot: IndexingSnapshot,
    ) -> KnowledgeItem | None:
        return await self._session.scalar(
            select(KnowledgeItem)
            .where(
                KnowledgeItem.id == snapshot.item_id,
                KnowledgeItem.revision == snapshot.revision,
                KnowledgeItem.status == "indexing",
            )
            .with_for_update()
        )

    async def _raise_if_item_exists(self, item_id: UUID) -> None:
        existing_id = await self._session.scalar(
            select(KnowledgeItem.id).where(KnowledgeItem.id == item_id)
        )
        if existing_id is not None:
            raise KnowledgeRevisionConflictError()

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
