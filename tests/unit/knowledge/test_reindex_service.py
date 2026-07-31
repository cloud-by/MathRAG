"""知识分块离线重建服务的单元测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

import app.modules.knowledge.reindex_service as reindex_module
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    KnowledgeSearchError,
)
from app.modules.knowledge.reindex_service import (
    KnowledgeReindexService,
    ReindexSummary,
    chunked,
)
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.search import EmbeddingUpdate, ReindexCandidate


MODEL = "embedding-reindex-test"


def vector(axis: int) -> list[float]:
    """构造固定 1024 维非零向量。"""
    values = [0.0] * 1024
    values[axis] = 1.0
    return values


def candidate(number: int) -> ReindexCandidate:
    """构造具有稳定 UUID 与文本快照的重建候选。"""
    return ReindexCandidate(
        chunk_id=UUID(f"00000000-0000-0000-0000-{number:012d}"),
        item_id=UUID(f"10000000-0000-0000-0000-{number:012d}"),
        retrieval_text=f"候选文本 {number}",
    )


class FakeProvider:
    """按调用次序返回结果，并记录数据库会话外的网络边界。"""

    model = MODEL
    dimensions = 1024

    def __init__(self, events: list[str], responses: Sequence[object] = ()) -> None:
        self._events = events
        self._responses = list(responses)
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        self._events.append(f"embedding:{','.join(batch)}")
        response = self._responses.pop(0) if self._responses else [
            vector(index) for index in range(len(batch))
        ]
        if isinstance(response, BaseException):
            raise response
        return [list(values) for values in response]  # type: ignore[union-attr]

    async def aclose(self) -> None:
        raise AssertionError("服务不拥有 Provider 生命周期")


class ProbeState:
    """保存仓储探针的输入输出与事务观察结果。"""

    def __init__(
        self,
        events: list[str],
        candidates: Sequence[ReindexCandidate],
        *,
        skipped: int = 0,
        fail_write_at: int | None = None,
    ) -> None:
        self.events = events
        self.candidates = list(candidates)
        self.skipped = skipped
        self.fail_write_at = fail_write_at
        self.marked: list[ReindexCandidate] = []
        self.writes: list[list[EmbeddingUpdate]] = []
        self.failed: list[list[ReindexCandidate]] = []
        self.refreshed: list[tuple[UUID, ...]] = []
        self.sessions: list[FakeSession] = []


class FakeTransaction:
    """模拟 ``AsyncSession.begin`` 的提交与异常回滚语义。"""

    def __init__(self, session: "FakeSession") -> None:
        self._session = session

    async def __aenter__(self) -> "FakeTransaction":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc, traceback
        if exc_type is None:
            self._session.commits += 1
        else:
            self._session.rollbacks += 1
        return False


class FakeSession:
    """每次调用工厂均产生独立且可观察的短会话。"""

    def __init__(self, state: ProbeState, role: str) -> None:
        self.state = state
        self.role = role
        self.closed = False
        self.commits = 0
        self.rollbacks = 0

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)


class FakeSessionContext:
    """模拟 sessionmaker 的自动关闭上下文。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> FakeSession:
        self._session.state.events.append(f"{self._session.role}:open")
        return self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, exc, traceback
        self._session.closed = True
        self._session.state.events.append(f"{self._session.role}:close")
        return False


class FakeSessionFactory:
    """首个会话用于读取，后续每批各自使用一个写会话。"""

    def __init__(self, state: ProbeState) -> None:
        self._state = state

    def __call__(self) -> FakeSessionContext:
        role = "read" if not self._state.sessions else "write"
        session = FakeSession(self._state, role)
        self._state.sessions.append(session)
        return FakeSessionContext(session)


class FakeRepository:
    """保留真实服务调用形状，只隔离 SQL 细节。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self._state = session.state

    async def list_reindex_candidates(self, embedding_model: str) -> list[ReindexCandidate]:
        assert embedding_model == MODEL
        self._state.events.append("candidates")
        return list(self._state.candidates)

    async def mark_candidates_indexing(
        self, candidates: Sequence[ReindexCandidate]
    ) -> int:
        self._state.marked = list(candidates)
        return len(candidates)

    async def count_ready_chunks(self, embedding_model: str) -> int:
        assert embedding_model == MODEL
        return self._state.skipped

    async def write_ready_embeddings(
        self,
        updates: Sequence[EmbeddingUpdate],
        embedding_model: str,
    ) -> int:
        assert embedding_model == MODEL
        batch = list(updates)
        write_index = len(self._state.writes)
        self._state.writes.append(batch)
        self._state.events.append("write")
        if self._state.fail_write_at == write_index:
            raise KnowledgeSearchError("CAS 写回数量不匹配")
        return len(batch)

    async def mark_chunks_failed(
        self, candidates: Sequence[ReindexCandidate]
    ) -> int:
        batch = list(candidates)
        self._state.failed.append(batch)
        self._state.events.append("failed")
        return len(batch)

    async def refresh_item_statuses(self, item_ids: Sequence[UUID]) -> int:
        self._state.refreshed.append(tuple(item_ids))
        self._state.events.append("refresh")
        return len(item_ids)


def service(
    monkeypatch: pytest.MonkeyPatch,
    state: ProbeState,
    provider: FakeProvider,
    *,
    batch_size: int = 2,
) -> KnowledgeReindexService:
    """将真实服务接到会话与仓储探针。"""
    monkeypatch.setattr(reindex_module, "KnowledgeRepository", FakeRepository)
    return KnowledgeReindexService(FakeSessionFactory(state), provider, batch_size=batch_size)  # type: ignore[arg-type]


def test_reindex_dtos_are_frozen_snapshots() -> None:
    """候选和写回 DTO 必须冻结数据库 CAS 所需的文本快照。"""
    selected = candidate(1)
    update = EmbeddingUpdate(
        chunk_id=selected.chunk_id,
        item_id=selected.item_id,
        expected_retrieval_text=selected.retrieval_text,
        vector=tuple(vector(0)),
    )

    with pytest.raises(FrozenInstanceError):
        selected.retrieval_text = "变化"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        update.vector = tuple(vector(1))  # type: ignore[misc]


def test_candidate_query_uses_only_the_three_planned_reindex_conditions() -> None:
    """候选条件不得在计划规定的三类之外扩大选择集。"""

    class EmptyResult:
        def all(self) -> list[object]:
            return []

    class CapturingSession:
        def __init__(self) -> None:
            self.statement: object | None = None

        async def execute(self, statement: object) -> EmptyResult:
            self.statement = statement
            return EmptyResult()

    session = CapturingSession()
    assert asyncio.run(
        KnowledgeRepository(session).list_reindex_candidates(MODEL)  # type: ignore[arg-type]
    ) == []

    sql = str(session.statement)
    assert "knowledge_chunks.embedding IS NULL" in sql
    assert "knowledge_chunks.status IN" in sql
    assert "knowledge_chunks.embedding_model IS DISTINCT FROM" in sql
    assert "vector_norm" not in sql


def test_repository_embedding_model_accepts_128_and_safely_rejects_longer() -> None:
    """Repository 必须在 SQL 前守住 embedding_model 列宽。"""

    class EmptyResult:
        def all(self) -> list[object]:
            return []

    class CountingSession:
        def __init__(self) -> None:
            self.execute_calls = 0

        async def execute(self, statement: object) -> EmptyResult:
            del statement
            self.execute_calls += 1
            return EmptyResult()

    session = CountingSession()
    repository = KnowledgeRepository(session)  # type: ignore[arg-type]
    assert asyncio.run(repository.list_reindex_candidates("m" * 128)) == []
    assert session.execute_calls == 1

    oversized_model = "  " + ("s" * 129) + "  "
    with pytest.raises(KnowledgeSearchError) as captured:
        asyncio.run(repository.list_reindex_candidates(oversized_model))

    assert "128" in str(captured.value)
    assert oversized_model.strip() not in str(captured.value)
    assert session.execute_calls == 1


@pytest.mark.parametrize(
    "method_name",
    ["mark_candidates_indexing", "mark_chunks_failed"],
)
def test_repository_item_updates_deduplicate_and_sort_item_ids(
    method_name: str,
) -> None:
    """批量写入必须先按 UUID 顺序锁 item，再更新 item 和 chunk。"""

    class RowCountResult:
        def __init__(self, rowcount: int) -> None:
            self.rowcount = rowcount

        def scalars(self) -> RowCountResult:
            return self

        def all(self) -> list[UUID]:
            return [lower_item_id, higher_item_id]

    class CapturingSession:
        def __init__(self) -> None:
            self.statements: list[object] = []

        async def execute(self, statement: object) -> RowCountResult:
            self.statements.append(statement)
            sql = str(statement)
            if sql.startswith("UPDATE knowledge_chunks"):
                return RowCountResult(3)
            return RowCountResult(2)

    lower_item_id = UUID("10000000-0000-0000-0000-000000000001")
    higher_item_id = UUID("f0000000-0000-0000-0000-000000000002")
    candidates = [
        ReindexCandidate(
            chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
            item_id=higher_item_id,
            retrieval_text="高位 item 的首个 chunk",
        ),
        ReindexCandidate(
            chunk_id=UUID("00000000-0000-0000-0000-000000000002"),
            item_id=lower_item_id,
            retrieval_text="低位 item 的 chunk",
        ),
        ReindexCandidate(
            chunk_id=UUID("00000000-0000-0000-0000-000000000003"),
            item_id=higher_item_id,
            retrieval_text="高位 item 的重复 chunk",
        ),
    ]
    session = CapturingSession()
    repository = KnowledgeRepository(session)  # type: ignore[arg-type]

    affected = asyncio.run(getattr(repository, method_name)(candidates))

    assert affected == 3
    assert str(session.statements[0]).startswith("SELECT knowledge_items.id")
    assert "FOR UPDATE" in str(session.statements[0])
    assert str(session.statements[1]).startswith("UPDATE knowledge_items")
    assert str(session.statements[2]).startswith("UPDATE knowledge_chunks")
    item_update_parameters = session.statements[1].compile().params  # type: ignore[union-attr]
    item_id_lists = [
        value
        for value in item_update_parameters.values()
        if isinstance(value, list)
        and value
        and all(isinstance(item_id, UUID) for item_id in value)
    ]
    assert item_id_lists == [[lower_item_id, higher_item_id]]


def test_repository_embedding_write_locks_items_before_chunks() -> None:
    """ready 写回也必须沿用 item→chunk 锁序。"""

    class Result:
        rowcount = 1

        def scalars(self) -> Result:
            return self

        def all(self) -> list[UUID]:
            return [item_id]

    class CapturingSession:
        def __init__(self) -> None:
            self.statements: list[object] = []

        async def execute(self, statement: object) -> Result:
            self.statements.append(statement)
            return Result()

    item_id = UUID("10000000-0000-0000-0000-000000000001")
    selected = candidate(1)
    embedding_update = EmbeddingUpdate(
        chunk_id=selected.chunk_id,
        item_id=item_id,
        expected_retrieval_text=selected.retrieval_text,
        vector=tuple(vector(0)),
    )
    session = CapturingSession()

    assert (
        asyncio.run(
            KnowledgeRepository(session).write_ready_embeddings(  # type: ignore[arg-type]
                [embedding_update],
                MODEL,
            )
        )
        == 1
    )

    assert str(session.statements[0]).startswith("SELECT knowledge_items.id")
    assert "FOR UPDATE" in str(session.statements[0])
    assert str(session.statements[1]).startswith("UPDATE knowledge_chunks")


def test_chunked_preserves_order_and_rejects_non_positive_true_integer_sizes() -> None:
    """分批保持原始顺序，批大小只接受正真整数。"""
    assert list(chunked([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError):
            list(chunked([1], invalid))  # type: ignore[arg-type]


def test_reindex_closes_read_session_before_embedding_and_uses_one_write_session_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读取事务不得跨网络调用，每个成功批次必须独立提交并刷新条目。"""
    events: list[str] = []
    selected = [candidate(number) for number in range(1, 6)]
    state = ProbeState(events, selected, skipped=2)
    provider = FakeProvider(events)

    summary = asyncio.run(service(monkeypatch, state, provider).reindex())

    assert summary == ReindexSummary(
        selected=5,
        ready=5,
        skipped=2,
        failed=0,
        embedding_model=MODEL,
        dimensions=1024,
    )
    assert events == [
        "read:open",
        "candidates",
        "read:close",
        "embedding:候选文本 1,候选文本 2",
        "write:open",
        "write",
        "refresh",
        "write:close",
        "embedding:候选文本 3,候选文本 4",
        "write:open",
        "write",
        "refresh",
        "write:close",
        "embedding:候选文本 5",
        "write:open",
        "write",
        "refresh",
        "write:close",
    ]
    assert provider.calls == [
        ["候选文本 1", "候选文本 2"],
        ["候选文本 3", "候选文本 4"],
        ["候选文本 5"],
    ]
    assert state.marked == selected
    assert [[update.chunk_id for update in batch] for batch in state.writes] == [
        [selected[0].chunk_id, selected[1].chunk_id],
        [selected[2].chunk_id, selected[3].chunk_id],
        [selected[4].chunk_id],
    ]
    assert all(session.closed for session in state.sessions)
    assert [session.commits for session in state.sessions] == [1, 1, 1, 1]


def test_reindex_empty_selection_is_idempotent_without_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当前模型全部就绪时只读一次并返回 skipped，不调用 Provider。"""
    events: list[str] = []
    state = ProbeState(events, [], skipped=4)
    provider = FakeProvider(events)

    summary = asyncio.run(service(monkeypatch, state, provider).reindex())

    assert summary == ReindexSummary(0, 0, 4, 0, MODEL, 1024)
    assert provider.calls == []
    assert events == ["read:open", "candidates", "read:close"]
    assert len(state.sessions) == 1
    assert state.sessions[0].closed is True


def test_second_batch_provider_failure_marks_only_current_batch_and_sanitizes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已提交批次保留，失败批次置 failed，后续候选仍保持初始 pending。"""
    events: list[str] = []
    selected = [candidate(number) for number in range(1, 6)]
    secret = "正文 https://private.example/v1 sk-private-key raw-body"
    provider = FakeProvider(
        events,
        [
            [vector(0), vector(1)],
            EmbeddingUnavailableError(secret),
        ],
    )
    state = ProbeState(events, selected)

    with pytest.raises(EmbeddingUnavailableError) as captured:
        asyncio.run(service(monkeypatch, state, provider).reindex())

    assert secret not in str(captured.value)
    assert provider.calls == [
        ["候选文本 1", "候选文本 2"],
        ["候选文本 3", "候选文本 4"],
    ]
    assert [update.chunk_id for update in state.writes[0]] == [
        selected[0].chunk_id,
        selected[1].chunk_id,
    ]
    assert state.failed == [selected[2:4]]
    assert selected[4] in state.marked
    assert all(selected[4] not in batch for batch in state.failed)
    assert events[-4:] == ["write:open", "failed", "refresh", "write:close"]


def test_response_count_mismatch_marks_batch_failed_with_safe_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 数量异常在写入前转为脱敏响应错误并标记当前批次。"""
    events: list[str] = []
    selected = [candidate(1), candidate(2)]
    state = ProbeState(events, selected)
    provider = FakeProvider(events, [[vector(0)]])

    with pytest.raises(EmbeddingResponseError) as captured:
        asyncio.run(service(monkeypatch, state, provider).reindex())

    assert str(captured.value) == "Embedding Provider 返回无效结果"
    assert state.writes == []
    assert state.failed == [selected]


@pytest.mark.parametrize(
    "malformation",
    ["none", "no_protocol", "length_raises", "iteration_raises"],
)
def test_malformed_response_is_sanitized_and_marks_current_batch_failed(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    """所有 response 结构异常都必须在单一可信边界内脱敏失败。"""
    events: list[str] = []
    selected = [candidate(1), candidate(2)]
    state = ProbeState(events, selected)
    secret = "raw-response sk-malformed-secret 正文"

    class LengthRaises:
        def __len__(self) -> int:
            raise RuntimeError(secret)

        def __iter__(self) -> object:
            return iter([vector(0), vector(1)])

    class IterationRaises:
        def __len__(self) -> int:
            return 2

        def __iter__(self) -> object:
            raise RuntimeError(secret)

    payloads: dict[str, object] = {
        "none": None,
        "no_protocol": object(),
        "length_raises": LengthRaises(),
        "iteration_raises": IterationRaises(),
    }

    class MalformedProvider(FakeProvider):
        async def embed_texts(self, texts: Sequence[str]) -> object:
            batch = list(texts)
            self.calls.append(batch)
            self._events.append(f"embedding:{','.join(batch)}")
            return payloads[malformation]

    provider = MalformedProvider(events)

    with pytest.raises(EmbeddingResponseError) as captured:
        asyncio.run(
            service(
                monkeypatch,
                state,
                provider,  # type: ignore[arg-type]
            ).reindex()
        )

    assert str(captured.value) == "Embedding Provider 返回无效结果"
    assert secret not in str(captured.value)
    assert state.writes == []
    assert state.failed == [selected]


def test_write_count_mismatch_rolls_back_current_batch_without_reclassifying_it_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAS 数量不匹配由当前事务整体回滚，不能另开事务覆盖并发状态。"""
    events: list[str] = []
    selected = [candidate(1), candidate(2)]
    state = ProbeState(events, selected, fail_write_at=0)
    provider = FakeProvider(events)

    with pytest.raises(KnowledgeSearchError, match="CAS"):
        asyncio.run(service(monkeypatch, state, provider).reindex())

    assert state.failed == []
    assert state.sessions[1].commits == 0
    assert state.sessions[1].rollbacks == 1
    assert state.sessions[1].closed is True


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_service_rejects_invalid_batch_size_before_opening_session(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: object,
) -> None:
    """服务构造阶段即拒绝非正真整数批大小。"""
    events: list[str] = []
    state = ProbeState(events, [])
    provider = FakeProvider(events)

    with pytest.raises(ValueError):
        service(monkeypatch, state, provider, batch_size=batch_size)  # type: ignore[arg-type]
    assert state.sessions == []


def test_service_model_length_is_validated_before_session_or_provider_calls() -> None:
    """Service 接受 128 字符 model，并在构造期拒绝清洗后超长 model。"""
    events: list[str] = []
    state = ProbeState(events, [])
    session_factory = FakeSessionFactory(state)
    provider = FakeProvider(events)
    provider.model = "m" * 128

    KnowledgeReindexService(
        session_factory,  # type: ignore[arg-type]
        provider,
        batch_size=2,
    )
    assert state.sessions == []
    assert provider.calls == []

    oversized_model = "  " + ("s" * 129) + "  "
    provider.model = oversized_model
    with pytest.raises(EmbeddingInputError) as captured:
        KnowledgeReindexService(
            session_factory,  # type: ignore[arg-type]
            provider,
            batch_size=2,
        )

    assert "128" in str(captured.value)
    assert oversized_model.strip() not in str(captured.value)
    assert state.sessions == []
    assert provider.calls == []
