"""旧知识的幂等导入服务。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from copy import deepcopy

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.knowledge.errors import (
    DuplicateLegacyIdError,
    LegacyKnowledgeConflictError,
    LegacyKnowledgeImportError,
)
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.schemas import (
    LegacyImportSummary,
    LegacyKnowledgeBundle,
    LegacyKnowledgeChunkInput,
    LegacyKnowledgeItemInput,
    collection_sha256,
)


def model_from_bundle(bundle: LegacyKnowledgeBundle) -> KnowledgeItem:
    """将一条旧知识映射为待持久化的 ORM 条目与唯一分块。"""
    item_data = deepcopy(bundle.item.model_dump(mode="python"))
    metadata = deepcopy(bundle.chunk.metadata)
    metadata.update(
        {
            "legacy_chunk_id": bundle.chunk.chunk_id,
            "legacy_source_id": bundle.chunk.source_id,
            "source_line": bundle.chunk.source_line,
        }
    )
    item = KnowledgeItem(
        legacy_id=bundle.item.id,
        category=item_data["category"],
        title=item_data["title"],
        keywords=item_data["keywords"],
        content=item_data["content"],
        example=item_data["example"],
        steps=item_data["steps"],
        difficulty=item_data["difficulty"],
        visibility="public",
        status="indexing",
        revision=1,
    )
    item.chunks.append(
        KnowledgeChunk(
            chunk_index=bundle.chunk_index,
            retrieval_text=bundle.chunk.retrieval_text,
            answer_context=bundle.chunk.answer_context,
            embedding=None,
            embedding_model=None,
            metadata_=metadata,
            status="pending",
        )
    )
    return item


def bundle_from_model(item: KnowledgeItem) -> LegacyKnowledgeBundle:
    """从 ORM 条目严格重建旧知识载荷，拒绝任何损坏的持久化数据。"""
    legacy_id = item.legacy_id
    try:
        if not isinstance(legacy_id, str) or not legacy_id:
            raise TypeError("legacy_id 必须是非空字符串")
        if len(item.chunks) != 1:
            raise ValueError("旧知识条目必须恰有一个分块")

        chunk = item.chunks[0]
        if not isinstance(chunk.metadata_, dict):
            raise TypeError("metadata 必须是字典")
        metadata = deepcopy(chunk.metadata_)
        legacy_chunk_id = metadata.pop("legacy_chunk_id")
        legacy_source_id = metadata.pop("legacy_source_id")
        source_line = metadata.pop("source_line")
        if not isinstance(legacy_chunk_id, str):
            raise TypeError("legacy_chunk_id 必须是字符串")
        if not isinstance(legacy_source_id, str):
            raise TypeError("legacy_source_id 必须是字符串")
        if type(source_line) is not int:
            raise TypeError("source_line 必须是整数")
        if type(chunk.chunk_index) is not int:
            raise TypeError("chunk_index 必须是整数")

        legacy_item = LegacyKnowledgeItemInput(
            id=legacy_id,
            category=deepcopy(item.category),
            title=deepcopy(item.title),
            keywords=deepcopy(item.keywords),
            content=deepcopy(item.content),
            example=deepcopy(item.example),
            steps=deepcopy(item.steps),
            difficulty=deepcopy(item.difficulty),
        )
        legacy_chunk = LegacyKnowledgeChunkInput(
            chunk_id=legacy_chunk_id,
            source_id=legacy_source_id,
            category=deepcopy(item.category),
            title=deepcopy(item.title),
            keywords=deepcopy(item.keywords),
            content=deepcopy(item.content),
            example=deepcopy(item.example),
            steps=deepcopy(item.steps),
            difficulty=deepcopy(item.difficulty),
            source_line=source_line,
            retrieval_text=deepcopy(chunk.retrieval_text),
            answer_context=deepcopy(chunk.answer_context),
            metadata=metadata,
        )
        return LegacyKnowledgeBundle(
            item=legacy_item,
            chunk=legacy_chunk,
            chunk_index=deepcopy(chunk.chunk_index),
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise LegacyKnowledgeImportError(
            f"无法还原 legacy_id={legacy_id!r} 的旧知识持久化载荷: {exc}"
        ) from exc


class LegacyKnowledgeImportService:
    """在调用方提供的会话上执行可审计、可回滚的旧知识导入。"""

    def __init__(self, session: AsyncSession, repository: KnowledgeRepository) -> None:
        self._session = session
        self._repository = repository

    async def import_bundles(
        self, bundles: Sequence[LegacyKnowledgeBundle]
    ) -> LegacyImportSummary:
        """导入一批旧知识；相同载荷跳过，冲突载荷整体回滚。"""
        duplicate_ids = sorted(
            legacy_id
            for legacy_id, count in Counter(bundle.item.id for bundle in bundles).items()
            if count > 1
        )
        if duplicate_ids:
            raise DuplicateLegacyIdError(f"发现重复的旧知识 ID: {', '.join(duplicate_ids)}")

        input_sha256 = collection_sha256(bundles)
        created = 0
        skipped = 0
        async with self._session.begin():
            for bundle in sorted(bundles, key=lambda current: current.item.id):
                existing = await self._repository.get_by_legacy_id(bundle.item.id)
                if existing is None:
                    self._repository.add(model_from_bundle(bundle))
                    created += 1
                    await self._session.flush()
                    continue

                if bundle_from_model(existing).sha256() == bundle.sha256():
                    skipped += 1
                    continue
                raise LegacyKnowledgeConflictError(
                    f"legacy_id={bundle.item.id} 的旧知识载荷与数据库不一致"
                )

            await self._session.flush()
            database_items = await self._repository.count_legacy_items()
            database_chunks = await self._repository.count_legacy_chunks()
            database_bundles = [
                bundle_from_model(item)
                for item in await self._repository.list_legacy_items_ordered()
            ]
            database_sha256 = collection_sha256(database_bundles)

        return LegacyImportSummary(
            input_items=len(bundles),
            input_chunks=len(bundles),
            created=created,
            skipped=skipped,
            conflicts=0,
            failed=0,
            database_items=database_items,
            database_chunks=database_chunks,
            input_sha256=input_sha256,
            database_sha256=database_sha256,
        )
