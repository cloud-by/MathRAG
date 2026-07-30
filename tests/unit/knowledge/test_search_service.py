"""知识向量检索编排服务的单元测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from uuid import UUID

import pytest

import app.modules.knowledge.search_service as search_service_module
from app.modules.knowledge.errors import EmbeddingResponseError, KnowledgeSearchError
from app.modules.knowledge.search import KnowledgeSearchHit, merge_search_hits
from app.modules.knowledge.search_service import (
    KnowledgeSearchService,
    build_knowledge_search_service,
)


def make_hit(chunk_number: int, *, distance: float) -> KnowledgeSearchHit:
    """构造可参与真实跨查询合并的稳定命中。"""
    return KnowledgeSearchHit(
        database_chunk_id=UUID(int=chunk_number),
        legacy_chunk_id=f"chunk-{chunk_number}",
        legacy_source_id=f"source-{chunk_number}",
        category="algebra",
        title=f"知识 {chunk_number}",
        keywords=("方程",),
        content="知识正文",
        example="2x + 1 = 5",
        steps=("移项", "求解"),
        difficulty="easy",
        answer_context="回答上下文",
        retrieval_text="检索文本",
        source_line=chunk_number,
        metadata={"origin": "unit-test"},
        distance=distance,
    )


class FakeProvider:
    """记录批量调用边界与时间顺序的 Embedding Provider。"""

    model = "embedding-test"
    dimensions = 1024

    def __init__(
        self,
        events: list[str],
        vectors: list[list[float]] | None = None,
    ) -> None:
        self._events = events
        self._vectors = vectors
        self.calls: list[list[str]] = []
        self.close_calls = 0

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self._events.append("embedding:start")
        self.calls.append(list(texts))
        await asyncio.sleep(0)
        self._events.append("embedding:end")
        if self._vectors is not None:
            return [list(vector) for vector in self._vectors]
        return [[float(index + 1)] for index in range(len(texts))]

    async def aclose(self) -> None:
        self.close_calls += 1
        raise AssertionError("单次检索不得关闭共享 Provider")


class FakeSession:
    """承载仓储响应，并检测同一会话是否被并发使用。"""

    def __init__(
        self,
        events: list[str],
        groups: Sequence[Sequence[KnowledgeSearchHit]] = (),
        *,
        sql_error_at: int | None = None,
    ) -> None:
        self.events = events
        self.groups = [list(group) for group in groups]
        self.sql_error_at = sql_error_at
        self.calls: list[dict[str, object]] = []
        self.repositories: list[FakeRepository] = []
        self.active = False
        self.closed = False
        self.exit_exception: type[BaseException] | None = None
        self.commit_calls = 0
        self.rollback_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1


class FakeSessionContext:
    """模拟 async_sessionmaker 返回的自动关闭上下文。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeSession:
        self._session.events.append("session:open")
        return self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc, traceback
        self._session.exit_exception = exc_type
        self._session.closed = True
        self._session.events.append("session:close")
        return False


class FakeSessionFactory:
    """只在服务真正开始数据库阶段时才创建上下文。"""

    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.call_count = 0

    def __call__(self) -> FakeSessionContext:
        self.call_count += 1
        return FakeSessionContext(self.session)


class FakeRepository:
    """按调用顺序返回分组，并在 await 边界检测并发。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session
        session.repositories.append(self)

    async def search_ready_chunks(
        self,
        *,
        query_vector: Sequence[float],
        embedding_model: str,
        limit: int,
    ) -> list[KnowledgeSearchHit]:
        if self._session.active:
            raise AssertionError("同一 AsyncSession 不得并发执行 SQL")
        call_index = len(self._session.calls)
        self._session.calls.append(
            {
                "query_vector": list(query_vector),
                "embedding_model": embedding_model,
                "limit": limit,
            }
        )
        self._session.events.append("sql")
        self._session.active = True
        try:
            await asyncio.sleep(0)
            if self._session.sql_error_at == call_index:
                raise RuntimeError("database unavailable")
            return list(self._session.groups[call_index])
        finally:
            self._session.active = False


def make_service(
    monkeypatch: pytest.MonkeyPatch,
    provider: FakeProvider,
    factory: FakeSessionFactory,
) -> KnowledgeSearchService:
    """把真实服务边界接到可观测的仓储与会话探针。"""
    monkeypatch.setattr(search_service_module, "KnowledgeRepository", FakeRepository)
    return KnowledgeSearchService(factory, provider)  # type: ignore[arg-type]


def test_search_batches_normalized_distinct_queries_then_uses_one_session_sequentially(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量化必须先完成，数据库调用须同会话、顺序执行并在关闭后合并。"""
    events: list[str] = []
    provider = FakeProvider(events, [[1.0], [2.0]])
    session = FakeSession(
        events,
        [
            [make_hit(1, distance=0.4)],
            [make_hit(1, distance=0.1), make_hit(2, distance=0.2)],
        ],
    )
    factory = FakeSessionFactory(session)
    service = make_service(monkeypatch, provider, factory)
    merge_observations: list[bool] = []

    def observed_merge(
        groups: Sequence[Sequence[KnowledgeSearchHit]],
        top_k: int,
    ) -> list[KnowledgeSearchHit]:
        merge_observations.append(session.closed)
        return merge_search_hits(groups, top_k)

    monkeypatch.setattr(search_service_module, "merge_search_hits", observed_merge)

    async def exercise() -> list[KnowledgeSearchHit]:
        return await service.search(
            ["  一元\n 方程 ", " ", "一元 方程", "\t几何  基础\t"],
            top_k=2,
        )

    hits = asyncio.run(exercise())

    assert events == [
        "embedding:start",
        "embedding:end",
        "session:open",
        "sql",
        "sql",
        "session:close",
    ]
    assert provider.calls == [["一元 方程", "几何 基础"]]
    assert provider.close_calls == 0
    assert factory.call_count == 1
    assert len(session.repositories) == 1
    assert session.calls == [
        {
            "query_vector": [1.0],
            "embedding_model": "embedding-test",
            "limit": 2,
        },
        {
            "query_vector": [2.0],
            "embedding_model": "embedding-test",
            "limit": 2,
        },
    ]
    assert merge_observations == [True]
    assert [(hit.database_chunk_id, hit.distance) for hit in hits] == [
        (UUID(int=1), 0.1),
        (UUID(int=2), 0.2),
    ]


@pytest.mark.parametrize("queries", [[], [" ", "\n\t"]])
def test_search_rejects_empty_normalized_queries_before_external_calls(
    monkeypatch: pytest.MonkeyPatch,
    queries: list[str],
) -> None:
    """清洗后没有文本时不得调用 Provider 或创建会话。"""
    events: list[str] = []
    provider = FakeProvider(events)
    factory = FakeSessionFactory(FakeSession(events))
    service = make_service(monkeypatch, provider, factory)

    async def exercise() -> None:
        with pytest.raises(KnowledgeSearchError):
            await service.search(queries, top_k=1)

    asyncio.run(exercise())
    assert provider.calls == []
    assert factory.call_count == 0
    assert events == []


def test_search_rejects_more_than_four_distinct_queries_before_external_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复项不占配额，但第五条 distinct 查询必须在外部调用前失败。"""
    events: list[str] = []
    provider = FakeProvider(events)
    factory = FakeSessionFactory(FakeSession(events))
    service = make_service(monkeypatch, provider, factory)

    async def exercise() -> None:
        with pytest.raises(KnowledgeSearchError):
            await service.search(["一", "一 ", "二", "三", "四", "五"], top_k=1)

    asyncio.run(exercise())
    assert provider.calls == []
    assert factory.call_count == 0
    assert events == []


@pytest.mark.parametrize("top_k", [0, 11, -1, True, 1.5, "1"])
def test_search_rejects_invalid_top_k_before_external_calls(
    monkeypatch: pytest.MonkeyPatch,
    top_k: object,
) -> None:
    """在线检索 Top-K 只接受 1 到 10 的真整数。"""
    events: list[str] = []
    provider = FakeProvider(events)
    factory = FakeSessionFactory(FakeSession(events))
    service = make_service(monkeypatch, provider, factory)

    async def exercise() -> None:
        with pytest.raises(KnowledgeSearchError):
            await service.search(["方程"], top_k=top_k)  # type: ignore[arg-type]

    asyncio.run(exercise())
    assert provider.calls == []
    assert factory.call_count == 0


@pytest.mark.parametrize("top_k", [1, 10])
def test_search_accepts_top_k_boundaries_and_forwards_them(
    monkeypatch: pytest.MonkeyPatch,
    top_k: int,
) -> None:
    """在线检索须接受闭区间两端，并原样传给仓储。"""
    events: list[str] = []
    provider = FakeProvider(events, [[1.0]])
    session = FakeSession(events, [[]])
    factory = FakeSessionFactory(session)
    service = make_service(monkeypatch, provider, factory)

    assert asyncio.run(service.search(["方程"], top_k=top_k)) == []
    assert session.calls[0]["limit"] == top_k


def test_search_rejects_embedding_count_mismatch_without_opening_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 返回数量异常时只暴露安全领域错误，且不得接触数据库。"""
    events: list[str] = []
    provider = FakeProvider(events, [[1.0]])
    factory = FakeSessionFactory(FakeSession(events))
    service = make_service(monkeypatch, provider, factory)

    async def exercise() -> None:
        with pytest.raises(EmbeddingResponseError, match="数量"):
            await service.search(["方程", "几何"], top_k=2)

    asyncio.run(exercise())
    assert provider.calls == [["方程", "几何"]]
    assert factory.call_count == 0
    assert events == ["embedding:start", "embedding:end"]


def test_search_closes_session_and_propagates_sql_error_without_transaction_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQL 异常必须向上传播，同时依赖上下文关闭会话且不手工提交或回滚。"""
    events: list[str] = []
    provider = FakeProvider(events, [[1.0], [2.0]])
    session = FakeSession(events, [[], []], sql_error_at=1)
    factory = FakeSessionFactory(session)
    service = make_service(monkeypatch, provider, factory)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="database unavailable"):
            await service.search(["方程", "几何"], top_k=3)

    asyncio.run(exercise())
    assert events == [
        "embedding:start",
        "embedding:end",
        "session:open",
        "sql",
        "sql",
        "session:close",
    ]
    assert session.closed is True
    assert session.exit_exception is RuntimeError
    assert session.commit_calls == 0
    assert session.rollback_calls == 0


def test_build_service_composes_lazy_dependency_getters_without_closing_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """构建器只组合现有单例 getter，不创建网络客户端或请求级资源。"""
    events: list[str] = []
    provider = FakeProvider(events)
    factory = FakeSessionFactory(FakeSession(events))
    getter_calls: list[str] = []

    def get_factory() -> FakeSessionFactory:
        getter_calls.append("session_factory")
        return factory

    def get_provider() -> FakeProvider:
        getter_calls.append("provider")
        return provider

    monkeypatch.setattr(search_service_module, "get_session_factory", get_factory)
    monkeypatch.setattr(search_service_module, "get_embedding_provider", get_provider)

    service = build_knowledge_search_service()

    assert isinstance(service, KnowledgeSearchService)
    assert service._session_factory is factory
    assert service._provider is provider
    assert getter_calls == ["session_factory", "provider"]
    assert factory.call_count == 0
    assert provider.calls == []
    assert provider.close_calls == 0
