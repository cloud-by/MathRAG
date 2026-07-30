"""知识条目的持久化访问封装。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import exists, func, or_, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.knowledge.errors import EmbeddingInputError, KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.search import (
    EmbeddingUpdate,
    KnowledgeSearchHit,
    ReindexCandidate,
    search_hit_from_row,
)


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

    async def list_reindex_candidates(
        self, embedding_model: str
    ) -> list[ReindexCandidate]:
        """按 UUID 返回需要使用当前模型重建的文本快照。"""
        current_model = _validated_embedding_model(embedding_model)
        statement = (
            select(
                KnowledgeChunk.id,
                KnowledgeChunk.knowledge_item_id,
                KnowledgeChunk.retrieval_text,
            )
            .where(
                or_(
                    KnowledgeChunk.embedding.is_(None),
                    KnowledgeChunk.status.in_(("pending", "failed")),
                    KnowledgeChunk.embedding_model.is_distinct_from(current_model),
                )
            )
            .order_by(KnowledgeChunk.id.asc())
        )
        result = await self._session.execute(statement)
        candidates: list[ReindexCandidate] = []
        for chunk_id, item_id, retrieval_text in result.all():
            if not isinstance(chunk_id, UUID) or not isinstance(item_id, UUID):
                raise KnowledgeSearchError("重建候选缺少有效 UUID")
            if type(retrieval_text) is not str or not retrieval_text.strip():
                raise KnowledgeSearchError(
                    f"重建候选 retrieval_text 不能为空 (chunk_id={chunk_id})"
                )
            candidates.append(
                ReindexCandidate(
                    chunk_id=chunk_id,
                    item_id=item_id,
                    retrieval_text=retrieval_text,
                )
            )
        return candidates

    async def mark_candidates_indexing(
        self, candidates: Sequence[ReindexCandidate]
    ) -> int:
        """将候选分块置 pending，并将相关条目置 indexing。"""
        selected = _validated_candidates(candidates)
        if not selected:
            return 0
        chunk_pairs = [
            (candidate.chunk_id, candidate.item_id) for candidate in selected
        ]
        item_ids = _unique_item_ids(selected)

        chunk_result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                tuple_(
                    KnowledgeChunk.id,
                    KnowledgeChunk.knowledge_item_id,
                ).in_(chunk_pairs)
            )
            .values(status="pending")
        )
        _require_rowcount(chunk_result.rowcount, len(chunk_pairs), "候选分块标记")
        item_result = await self._session.execute(
            update(KnowledgeItem)
            .where(KnowledgeItem.id.in_(item_ids))
            .values(status="indexing")
        )
        _require_rowcount(item_result.rowcount, len(item_ids), "候选条目标记")
        return len(chunk_pairs)

    async def write_ready_embeddings(
        self,
        updates: Sequence[EmbeddingUpdate],
        embedding_model: str,
    ) -> int:
        """按分块 UUID 与检索文本快照 CAS 写入有效向量。"""
        current_model = _validated_embedding_model(embedding_model)
        prepared = _validated_embedding_updates(updates)
        if not prepared:
            return 0

        affected = 0
        for embedding_update, vector in prepared:
            result = await self._session.execute(
                update(KnowledgeChunk)
                .where(
                    KnowledgeChunk.id == embedding_update.chunk_id,
                    KnowledgeChunk.knowledge_item_id == embedding_update.item_id,
                    KnowledgeChunk.retrieval_text
                    == embedding_update.expected_retrieval_text,
                )
                .values(
                    embedding=vector,
                    embedding_model=current_model,
                    status="ready",
                )
            )
            rowcount = result.rowcount
            if type(rowcount) is not int or rowcount != 1:
                raise KnowledgeSearchError("Embedding CAS 写回数量与输入不一致")
            affected += rowcount
        _require_rowcount(affected, len(prepared), "Embedding CAS 写回")
        return affected

    async def mark_chunks_failed(
        self, candidates: Sequence[ReindexCandidate]
    ) -> int:
        """清空当前失败批次的向量，并把相关条目置 failed。"""
        selected = _validated_candidates(candidates)
        if not selected:
            return 0
        chunk_pairs = [
            (candidate.chunk_id, candidate.item_id) for candidate in selected
        ]
        item_ids = _unique_item_ids(selected)

        chunk_result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                tuple_(
                    KnowledgeChunk.id,
                    KnowledgeChunk.knowledge_item_id,
                ).in_(chunk_pairs)
            )
            .values(embedding=None, embedding_model=None, status="failed")
        )
        _require_rowcount(chunk_result.rowcount, len(chunk_pairs), "失败分块标记")
        item_result = await self._session.execute(
            update(KnowledgeItem)
            .where(KnowledgeItem.id.in_(item_ids))
            .values(status="failed")
        )
        _require_rowcount(item_result.rowcount, len(item_ids), "失败条目标记")
        return len(chunk_pairs)

    async def refresh_item_statuses(self, item_ids: Sequence[UUID]) -> int:
        """只把至少有一条分块且全部分块有效就绪的条目置 ready。"""
        selected_item_ids = _validated_item_ids(item_ids)
        if not selected_item_ids:
            return 0

        has_chunks = exists().where(
            KnowledgeChunk.knowledge_item_id == KnowledgeItem.id
        )
        has_unready_chunks = exists().where(
            KnowledgeChunk.knowledge_item_id == KnowledgeItem.id,
            or_(
                KnowledgeChunk.status != "ready",
                KnowledgeChunk.embedding.is_(None),
                KnowledgeChunk.embedding_model.is_(None),
            ),
        )
        result = await self._session.execute(
            update(KnowledgeItem)
            .where(
                KnowledgeItem.id.in_(selected_item_ids),
                has_chunks,
                ~has_unready_chunks,
            )
            .values(status="ready")
        )
        rowcount = result.rowcount
        if type(rowcount) is not int or rowcount < 0:
            raise KnowledgeSearchError("条目就绪状态刷新数量无效")
        return rowcount

    async def count_ready_chunks(self, embedding_model: str) -> int:
        """统计当前模型下具有非零向量的就绪分块。"""
        current_model = _validated_embedding_model(embedding_model)
        statement = (
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(
                KnowledgeChunk.status == "ready",
                KnowledgeChunk.embedding.is_not(None),
                KnowledgeChunk.embedding_model == current_model,
                func.vector_norm(KnowledgeChunk.embedding) > 0,
            )
        )
        result = await self._session.execute(statement)
        count = result.scalar_one()
        if type(count) is not int or count < 0:
            raise KnowledgeSearchError("就绪分块统计结果无效")
        return count

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
                func.vector_norm(KnowledgeChunk.embedding) > 0,
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
    if not any(value != 0.0 for value in vector):
        raise EmbeddingInputError("query_vector 不能是零向量")
    return vector


def _validated_embedding_model(embedding_model: str) -> str:
    """验证并清洗用于持久化边界的模型标识。"""
    if type(embedding_model) is not str or not embedding_model.strip():
        raise KnowledgeSearchError("embedding_model 必须是非空字符串")
    cleaned_model = embedding_model.strip()
    if len(cleaned_model) > 128:
        raise KnowledgeSearchError("embedding_model 长度不能超过 128")
    return cleaned_model


def _validated_candidates(
    candidates: Sequence[ReindexCandidate],
) -> list[ReindexCandidate]:
    """复制并验证候选 DTO，拒绝重复分块。"""
    if isinstance(candidates, (str, bytes)):
        raise KnowledgeSearchError("重建候选必须是 DTO 序列")
    selected = list(candidates)
    chunk_ids: set[UUID] = set()
    for candidate in selected:
        if not isinstance(candidate, ReindexCandidate):
            raise KnowledgeSearchError("重建候选必须是 DTO 序列")
        if (
            not isinstance(candidate.chunk_id, UUID)
            or not isinstance(candidate.item_id, UUID)
        ):
            raise KnowledgeSearchError("重建候选缺少有效 UUID")
        if (
            type(candidate.retrieval_text) is not str
            or not candidate.retrieval_text.strip()
        ):
            raise KnowledgeSearchError("重建候选 retrieval_text 不能为空")
        if candidate.chunk_id in chunk_ids:
            raise KnowledgeSearchError("重建候选包含重复分块 UUID")
        chunk_ids.add(candidate.chunk_id)
    return selected


def _validated_embedding_updates(
    updates: Sequence[EmbeddingUpdate],
) -> list[tuple[EmbeddingUpdate, list[float]]]:
    """在执行任何 SQL 前验证全部 CAS 写回载荷。"""
    if isinstance(updates, (str, bytes)):
        raise KnowledgeSearchError("Embedding 写回必须是 DTO 序列")
    prepared: list[tuple[EmbeddingUpdate, list[float]]] = []
    chunk_ids: set[UUID] = set()
    for embedding_update in updates:
        if not isinstance(embedding_update, EmbeddingUpdate):
            raise KnowledgeSearchError("Embedding 写回必须是 DTO 序列")
        if (
            not isinstance(embedding_update.chunk_id, UUID)
            or not isinstance(embedding_update.item_id, UUID)
        ):
            raise KnowledgeSearchError("Embedding 写回缺少有效 UUID")
        if (
            type(embedding_update.expected_retrieval_text) is not str
            or not embedding_update.expected_retrieval_text.strip()
        ):
            raise KnowledgeSearchError("Embedding 写回 retrieval_text 不能为空")
        if embedding_update.chunk_id in chunk_ids:
            raise KnowledgeSearchError("Embedding 写回包含重复分块 UUID")
        chunk_ids.add(embedding_update.chunk_id)
        prepared.append(
            (
                embedding_update,
                _validated_embedding_vector(embedding_update.vector),
            )
        )
    return prepared


def _validated_embedding_vector(values: tuple[float, ...]) -> list[float]:
    """验证持久化向量为固定 1024 维有限非零元组。"""
    if type(values) is not tuple:
        raise KnowledgeSearchError("Embedding 向量必须是 1024 维元组")
    try:
        vector = [float(value) for value in values]
    except Exception:
        raise KnowledgeSearchError("Embedding 向量必须包含有限数值") from None
    if len(vector) != 1024 or not all(math.isfinite(value) for value in vector):
        raise KnowledgeSearchError("Embedding 向量必须是 1024 维有限向量")
    if not any(value != 0.0 for value in vector):
        raise KnowledgeSearchError("Embedding 向量不能是零向量")
    return vector


def _validated_item_ids(item_ids: Sequence[UUID]) -> list[UUID]:
    """验证条目 UUID，并按首次出现顺序去重。"""
    if isinstance(item_ids, (str, bytes)):
        raise KnowledgeSearchError("item_ids 必须是 UUID 序列")
    selected = list(item_ids)
    if not all(isinstance(item_id, UUID) for item_id in selected):
        raise KnowledgeSearchError("item_ids 必须是 UUID 序列")
    return list(dict.fromkeys(selected))


def _unique_item_ids(candidates: Sequence[ReindexCandidate]) -> list[UUID]:
    """去重并按字典序提取候选的条目 UUID。"""
    return sorted({candidate.item_id for candidate in candidates}, key=str)


def _require_rowcount(actual: object, expected: int, operation: str) -> None:
    """要求写操作精确影响预期行数。"""
    if type(actual) is not int or actual != expected:
        raise KnowledgeSearchError(f"{operation}数量与输入不一致")
