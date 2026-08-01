"""知识分块重建流程的 PostgreSQL 集成测试。"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Sequence
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.knowledge.errors import EmbeddingUnavailableError, KnowledgeSearchError
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.reindex_service import KnowledgeReindexService, ReindexSummary
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.search import EmbeddingUpdate, ReindexCandidate
from tests.integration.database_safety import require_test_database_url


MODEL_V1 = "integration-reindex-v1"
MODEL_V2 = "integration-reindex-v2"


def vector(axis: int) -> list[float]:
    """构造固定 1024 维单位向量。"""
    values = [0.0] * 1024
    values[axis] = 1.0
    return values


def make_item(
    number: int,
    *,
    retrieval_text: str | None = None,
    item_status: str = "indexing",
    chunk_status: str = "pending",
    embedding: Sequence[float] | None = None,
    embedding_model: str | None = None,
) -> KnowledgeItem:
    """构造固定 UUID 的单分块条目。"""
    item = KnowledgeItem(
        id=UUID(f"10000000-0000-0000-0000-{number:012d}"),
        legacy_id=f"reindex-{number}",
        category="reindex-integration",
        title=f"重建条目 {number}",
        keywords=["重建"],
        content=f"条目正文 {number}",
        example="示例",
        steps=["步骤"],
        difficulty="easy",
        status=item_status,
    )
    item.chunks.append(
        KnowledgeChunk(
            id=UUID(f"00000000-0000-0000-0000-{number:012d}"),
            chunk_index=0,
            retrieval_text=(
                retrieval_text
                if retrieval_text is not None
                else f"重建检索文本 {number}"
            ),
            answer_context=f"回答上下文 {number}",
            embedding=list(embedding) if embedding is not None else None,
            embedding_model=embedding_model,
            metadata_={"number": number},
            status=chunk_status,
        )
    )
    return item


class FakeProvider:
    """真实服务使用的无网络 Provider，支持按批注入故障。"""

    dimensions = 1024

    def __init__(
        self,
        model: str,
        responses: Sequence[object] = (),
    ) -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        if self._responses:
            response = self._responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return [list(values) for values in response]  # type: ignore[union-attr]
        offset = sum(len(call) for call in self.calls[:-1])
        return [vector((offset + index) % 1024) for index in range(len(batch))]

    async def aclose(self) -> None:
        return None


class MutatingProvider(FakeProvider):
    """在网络阶段模拟其他事务修改 CAS 文本快照。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        chunk_id: UUID,
    ) -> None:
        super().__init__(MODEL_V1)
        self._session_factory = session_factory
        self._chunk_id = chunk_id

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        async with self._session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(KnowledgeChunk)
                    .where(KnowledgeChunk.id == self._chunk_id)
                    .values(retrieval_text="并发修改后的检索文本")
                )
        return await super().embed_texts(texts)


async def cleanup(session: AsyncSession) -> None:
    """按外键顺序清空专用测试库知识表。"""
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def reset(
    session_factory: async_sessionmaker[AsyncSession],
    items: Sequence[KnowledgeItem] = (),
) -> None:
    """恢复空表并写入一个独立测试场景。"""
    async with session_factory() as session:
        async with session.begin():
            await cleanup(session)
            session.add_all(items)


async def stored_states(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[tuple[int, str, str, bool, str | None, str]]:
    """按 UUID 返回条目/分块关键状态。"""
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(KnowledgeChunk, KnowledgeItem)
                .join(KnowledgeItem)
                .order_by(KnowledgeChunk.id)
            )
        ).all()
        return [
            (
                int(chunk.metadata_["number"]),
                item.status,
                chunk.status,
                chunk.embedding is not None,
                chunk.embedding_model,
                chunk.retrieval_text,
            )
            for chunk, item in rows
        ]


async def stored_dimensions(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[int]:
    """按 UUID 返回已写入向量的数据库维度。"""
    async with session_factory() as session:
        result = await session.execute(
            select(func.vector_dims(KnowledgeChunk.embedding)).order_by(
                KnowledgeChunk.id
            )
        )
        return list(result.scalars())


async def assert_database_restored(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """确认收尾状态精确为当前迁移 head 且两张知识表为空。"""
    async with session_factory() as session:
        revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
        item_count = await session.scalar(select(func.count()).select_from(KnowledgeItem))
        chunk_count = await session.scalar(select(func.count()).select_from(KnowledgeChunk))
        assert (revision, item_count, chunk_count) == (
            "0006_add_account_management",
            0,
            0,
        )


async def exercise_reindex(database_url: str) -> None:
    """覆盖幂等、模型切换、批次隔离、CAS 与数据库约束。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        await reset(session_factory)
        await assert_database_restored(session_factory)

        await reset(
            session_factory,
            [make_item(number) for number in range(26, 0, -1)],
        )
        provider_v1 = FakeProvider(MODEL_V1)
        first = await KnowledgeReindexService(
            session_factory, provider_v1, batch_size=10
        ).reindex()
        assert first == ReindexSummary(26, 26, 0, 0, MODEL_V1, 1024)
        assert [len(batch) for batch in provider_v1.calls] == [10, 10, 6]
        assert [text for batch in provider_v1.calls for text in batch] == [
            f"重建检索文本 {number}" for number in range(1, 27)
        ]
        assert await stored_dimensions(session_factory) == [1024] * 26
        assert await stored_states(session_factory) == [
            (number, "ready", "ready", True, MODEL_V1, f"重建检索文本 {number}")
            for number in range(1, 27)
        ]

        calls_before_second_run = list(provider_v1.calls)
        second = await KnowledgeReindexService(
            session_factory, provider_v1, batch_size=10
        ).reindex()
        assert second == ReindexSummary(0, 0, 26, 0, MODEL_V1, 1024)
        assert provider_v1.calls == calls_before_second_run

        provider_v2 = FakeProvider(MODEL_V2)
        changed = await KnowledgeReindexService(
            session_factory, provider_v2, batch_size=8
        ).reindex()
        assert changed == ReindexSummary(26, 26, 0, 0, MODEL_V2, 1024)
        assert [len(batch) for batch in provider_v2.calls] == [8, 8, 8, 2]
        assert all(state[4] == MODEL_V2 for state in await stored_states(session_factory))

        await reset(session_factory, [make_item(number) for number in range(1, 6)])
        secret = "raw-body https://private.example/v1 sk-private-key"
        failing_provider = FakeProvider(
            MODEL_V1,
            [
                [vector(0), vector(1)],
                EmbeddingUnavailableError(secret),
            ],
        )
        with pytest.raises(EmbeddingUnavailableError) as captured:
            await KnowledgeReindexService(
                session_factory, failing_provider, batch_size=2
            ).reindex()
        assert secret not in str(captured.value)
        assert await stored_states(session_factory) == [
            (1, "ready", "ready", True, MODEL_V1, "重建检索文本 1"),
            (2, "ready", "ready", True, MODEL_V1, "重建检索文本 2"),
            (3, "failed", "failed", False, None, "重建检索文本 3"),
            (4, "failed", "failed", False, None, "重建检索文本 4"),
            (5, "indexing", "pending", False, None, "重建检索文本 5"),
        ]

        await reset(session_factory, [make_item(1), make_item(2)])
        mutating_provider = MutatingProvider(
            session_factory,
            UUID("00000000-0000-0000-0000-000000000002"),
        )
        with pytest.raises(KnowledgeSearchError, match="数量"):
            await KnowledgeReindexService(
                session_factory, mutating_provider, batch_size=2
            ).reindex()
        assert await stored_states(session_factory) == [
            (1, "indexing", "pending", False, None, "重建检索文本 1"),
            (2, "indexing", "pending", False, None, "并发修改后的检索文本"),
        ]

        await reset(session_factory, [make_item(1, retrieval_text=" \t ")])
        async with session_factory() as session:
            with pytest.raises(KnowledgeSearchError, match="retrieval_text"):
                await KnowledgeRepository(session).list_reindex_candidates(MODEL_V1)

        await reset(session_factory, [make_item(1)])
        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            bad_update = EmbeddingUpdate(
                chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
                item_id=UUID("10000000-0000-0000-0000-000000000001"),
                expected_retrieval_text="重建检索文本 1",
                vector=tuple([0.0] * 1024),
            )
            with pytest.raises(KnowledgeSearchError, match="零向量"):
                await repository.write_ready_embeddings([bad_update], MODEL_V1)

        await reset(session_factory, [make_item(1)])
        async with session_factory() as session:
            with pytest.raises(IntegrityError):
                async with session.begin():
                    await session.execute(
                        update(KnowledgeChunk)
                        .values(status="ready", embedding=None, embedding_model=None)
                    )
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            await reset(session_factory)
            await assert_database_restored(session_factory)
        except BaseException:
            cleanup_failed = True
            if not original_exception_pending:
                raise
        finally:
            try:
                await engine.dispose()
            except BaseException:
                if not original_exception_pending and not cleanup_failed:
                    raise


async def exercise_item_pairing_guard(
    database_url: str,
    operation: str,
) -> None:
    """验证 chunk/item 错配时当前事务不得留下任何写入。"""
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=False,
    )
    try:
        await reset(session_factory, [make_item(1), make_item(2)])
        initial_states = await stored_states(session_factory)
        chunk_id = UUID("00000000-0000-0000-0000-000000000001")
        wrong_item_id = UUID("10000000-0000-0000-0000-000000000002")
        wrong_candidate = ReindexCandidate(
            chunk_id=chunk_id,
            item_id=wrong_item_id,
            retrieval_text="重建检索文本 1",
        )
        wrong_update = EmbeddingUpdate(
            chunk_id=chunk_id,
            item_id=wrong_item_id,
            expected_retrieval_text="重建检索文本 1",
            vector=tuple(vector(0)),
        )

        async with session_factory() as session:
            repository = KnowledgeRepository(session)
            with pytest.raises(KnowledgeSearchError, match="数量"):
                async with session.begin():
                    if operation == "write_ready_embeddings":
                        await repository.write_ready_embeddings(
                            [wrong_update],
                            MODEL_V1,
                        )
                    else:
                        await getattr(repository, operation)([wrong_candidate])

        assert await stored_states(session_factory) == initial_states
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            await reset(session_factory)
            await assert_database_restored(session_factory)
        except BaseException:
            cleanup_failed = True
            if not original_exception_pending:
                raise
        finally:
            try:
                await engine.dispose()
            except BaseException:
                if not original_exception_pending and not cleanup_failed:
                    raise


@pytest.mark.parametrize(
    "operation",
    [
        "write_ready_embeddings",
        "mark_candidates_indexing",
        "mark_chunks_failed",
    ],
)
def test_repository_rejects_mismatched_chunk_item_pairs_and_rolls_back(
    operation: str,
) -> None:
    """身份守卫后在专用 PG 上验证 DTO 复合身份 CAS。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(
        database_url,
        os.getenv("DATABASE_URL"),
    )

    asyncio.run(exercise_item_pairing_guard(database_url, operation))


def test_reindex_service_commits_batches_idempotently_and_preserves_failure_boundaries() -> None:
    """在身份守卫后的专用 PG 上验证完整重建事务契约。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_reindex(database_url))
