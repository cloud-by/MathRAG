"""知识分块离线向量重建的事务编排。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    validate_and_normalize_vector,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.search import EmbeddingUpdate, ReindexCandidate


T = TypeVar("T")


@dataclass(frozen=True)
class ReindexSummary:
    """一次知识向量重建成功完成后的稳定摘要。"""

    selected: int
    ready: int
    skipped: int
    failed: int
    embedding_model: str
    dimensions: int


def chunked(values: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """按输入顺序切分序列，并严格验证批大小。"""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是大于 0 的整数")
    for offset in range(0, len(values), batch_size):
        yield list(values[offset : offset + batch_size])


class KnowledgeReindexService:
    """在网络调用两侧使用独立短事务重建知识向量。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: EmbeddingProvider,
        *,
        batch_size: int,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size 必须是大于 0 的整数")

        try:
            model = provider.model
            dimensions = provider.dimensions
        except Exception:
            raise EmbeddingInputError("Embedding Provider 配置无效") from None
        if type(model) is not str or not model.strip():
            raise EmbeddingInputError("Embedding 模型不能为空")
        cleaned_model = model.strip()
        if len(cleaned_model) > 128:
            raise EmbeddingInputError("Embedding 模型长度不能超过 128")
        if type(dimensions) is not int or dimensions != 1024:
            raise EmbeddingInputError("Embedding 维度必须为 1024")

        self._session_factory = session_factory
        self._provider = provider
        self._batch_size = batch_size
        self._embedding_model = cleaned_model
        self._dimensions = dimensions

    async def reindex(self) -> ReindexSummary:
        """重建全部候选；每个已成功批次独立提交。"""
        candidates, skipped = await self._load_and_mark_candidates()
        if not candidates:
            return ReindexSummary(
                selected=0,
                ready=0,
                skipped=skipped,
                failed=0,
                embedding_model=self._embedding_model,
                dimensions=self._dimensions,
            )

        ready = 0
        for batch in chunked(candidates, self._batch_size):
            vectors = await self._embed_batch(batch)
            updates = [
                EmbeddingUpdate(
                    chunk_id=selected.chunk_id,
                    item_id=selected.item_id,
                    expected_retrieval_text=selected.retrieval_text,
                    vector=tuple(vector),
                )
                for selected, vector in zip(batch, vectors, strict=True)
            ]
            await self._write_batch(updates)
            ready += len(updates)

        return ReindexSummary(
            selected=len(candidates),
            ready=ready,
            skipped=skipped,
            failed=0,
            embedding_model=self._embedding_model,
            dimensions=self._dimensions,
        )

    async def _load_and_mark_candidates(self) -> tuple[list[ReindexCandidate], int]:
        """在单个短事务中取得文本快照并把候选标记为处理中。"""
        async with self._session_factory() as session:
            async with session.begin():
                repository = KnowledgeRepository(session)
                candidates = await repository.list_reindex_candidates(
                    self._embedding_model
                )
                skipped = await repository.count_ready_chunks(self._embedding_model)
                if candidates:
                    await repository.mark_candidates_indexing(candidates)
                return candidates, skipped

    async def _embed_batch(
        self, batch: Sequence[ReindexCandidate]
    ) -> list[list[float]]:
        """在没有数据库会话时调用 Provider，并统一脱敏失败。"""
        try:
            response = await self._provider.embed_texts(
                [candidate.retrieval_text for candidate in batch]
            )
        except EmbeddingInputError:
            await self._mark_batch_failed(batch)
            raise EmbeddingInputError("Embedding 输入或配置无效") from None
        except EmbeddingResponseError:
            await self._mark_batch_failed(batch)
            raise EmbeddingResponseError("Embedding Provider 返回无效结果") from None
        except EmbeddingUnavailableError:
            await self._mark_batch_failed(batch)
            raise EmbeddingUnavailableError("Embedding Provider 暂时不可用") from None
        except Exception:
            await self._mark_batch_failed(batch)
            raise EmbeddingUnavailableError("Embedding Provider 暂时不可用") from None

        try:
            if len(response) != len(batch):
                raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
            vectors: list[list[float]] = []
            for values in response:
                vectors.append(
                    validate_and_normalize_vector(values, self._dimensions)
                )
            if len(vectors) != len(batch):
                raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
        except Exception:
            await self._mark_batch_failed(batch)
            raise EmbeddingResponseError("Embedding Provider 返回无效结果") from None
        return vectors

    async def _write_batch(self, updates: Sequence[EmbeddingUpdate]) -> None:
        """在当前批次的独立短事务中 CAS 写回并刷新条目状态。"""
        async with self._session_factory() as session:
            async with session.begin():
                repository = KnowledgeRepository(session)
                await repository.write_ready_embeddings(
                    updates,
                    self._embedding_model,
                )
                await repository.refresh_item_statuses(_item_ids(updates))

    async def _mark_batch_failed(
        self, candidates: Sequence[ReindexCandidate]
    ) -> None:
        """只清理并标记当前 Provider 失败批次。"""
        async with self._session_factory() as session:
            async with session.begin():
                repository = KnowledgeRepository(session)
                await repository.mark_chunks_failed(candidates)
                await repository.refresh_item_statuses(_item_ids(candidates))


def _item_ids(values: Sequence[ReindexCandidate | EmbeddingUpdate]) -> list[UUID]:
    """按首次出现顺序提取条目 UUID。"""
    return list(dict.fromkeys(value.item_id for value in values))
