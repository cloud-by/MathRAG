"""旧知识导入服务的单元测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from app.modules.knowledge.errors import (
    DuplicateLegacyIdError,
    LegacyKnowledgeConflictError,
    LegacyKnowledgeImportError,
)
from app.modules.knowledge.models import KnowledgeItem
from app.modules.knowledge.schemas import (
    LegacyKnowledgeBundle,
    LegacyKnowledgeChunkInput,
    LegacyKnowledgeItemInput,
)
from app.modules.knowledge.service import (
    LegacyKnowledgeImportService,
    bundle_from_model,
    model_from_bundle,
)


def make_bundle(legacy_id: str = "k0001", **chunk_changes: object) -> LegacyKnowledgeBundle:
    """构造一条可导入的旧知识记录。"""
    item = LegacyKnowledgeItemInput(
        id=legacy_id,
        category="algebra",
        title=f"条目 {legacy_id}",
        keywords=["代数", "方程"],
        content=f"{legacy_id} 的知识内容",
        example="2x + 1 = 5",
        steps=["移项", "求解"],
        difficulty="easy",
    )
    chunk_data: dict[str, object] = {
        "chunk_id": f"{legacy_id}-chunk-0",
        "source_id": legacy_id,
        "category": item.category,
        "title": item.title,
        "keywords": item.keywords,
        "content": item.content,
        "example": item.example,
        "steps": item.steps,
        "difficulty": item.difficulty,
        "source_line": 7,
        "retrieval_text": f"{legacy_id} 检索文本",
        "answer_context": f"{legacy_id} 回答上下文",
        "metadata": {"origin": "legacy", "nested": {"values": ["初始"]}},
    }
    chunk_data.update(chunk_changes)
    return LegacyKnowledgeBundle(item=item, chunk=LegacyKnowledgeChunkInput(**chunk_data))


class FakeTransaction:
    """精确记录提交和回滚的轻量事务探针。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.entered = False
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> FakeTransaction:
        self.entered = True
        self._session.transactions.append(self)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if exc_type is None:
            self.committed = True
            self._session.persisted.extend(self._session.pending)
        else:
            self.rolled_back = True
        self._session.pending.clear()
        return False


class FakeSession:
    """保留 pending 与 committed 两层状态，避免伪造回滚结论。"""

    def __init__(self, persisted: Sequence[KnowledgeItem] = ()) -> None:
        self.persisted = list(persisted)
        self.pending: list[KnowledgeItem] = []
        self.transactions: list[FakeTransaction] = []
        self.flush_pending_counts: list[int] = []

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def flush(self) -> None:
        self.flush_pending_counts.append(len(self.pending))


class FakeRepository:
    """与真实仓储保持相同公开方法签名的内存实现。"""

    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_by_legacy_id(self, legacy_id: str) -> KnowledgeItem | None:
        return next(
            (
                item
                for item in [*self._session.pending, *self._session.persisted]
                if item.legacy_id == legacy_id
            ),
            None,
        )

    def add(self, item: KnowledgeItem) -> None:
        self._session.pending.append(item)

    async def count_legacy_items(self) -> int:
        return sum(
            item.legacy_id is not None
            for item in [*self._session.pending, *self._session.persisted]
        )

    async def count_legacy_chunks(self) -> int:
        return sum(
            len(item.chunks)
            for item in [*self._session.pending, *self._session.persisted]
            if item.legacy_id is not None
        )

    async def list_legacy_items_ordered(self) -> list[KnowledgeItem]:
        return sorted(
            (
                item
                for item in [*self._session.pending, *self._session.persisted]
                if item.legacy_id is not None
            ),
            key=lambda item: item.legacy_id or "",
        )


def test_model_and_bundle_round_trip_preserves_payload_and_copies_containers() -> None:
    """ORM 映射完整保留持久化载荷，且不与输入共享可变 JSON 容器。"""
    bundle = make_bundle()
    model = model_from_bundle(bundle)
    model.keywords.append("篡改")
    model.steps.append("额外步骤")
    model.chunks[0].metadata_["nested"]["values"].append("篡改")

    assert bundle.item.keywords == ["代数", "方程"]
    assert bundle.item.steps == ["移项", "求解"]
    assert bundle.chunk.metadata["nested"] == {"values": ["初始"]}
    assert model.visibility == "public"
    assert model.status == "indexing"
    assert model.revision == 1
    assert model.chunks[0].embedding is None
    assert model.chunks[0].embedding_model is None
    assert model.chunks[0].status == "pending"
    assert bundle_from_model(model_from_bundle(bundle)).persistent_payload() == bundle.persistent_payload()
    assert bundle_from_model(model_from_bundle(bundle)).sha256() == bundle.sha256()


def test_first_import_creates_and_second_identical_import_skips() -> None:
    """同一输入第二次导入应幂等跳过，且创建后立即 flush。"""
    async def exercise() -> None:
        bundle = make_bundle()
        session = FakeSession()
        repository = FakeRepository(session)
        service = LegacyKnowledgeImportService(session, repository)  # type: ignore[arg-type]

        first = await service.import_bundles([bundle])
        second = await service.import_bundles([bundle])

        assert (first.created, first.skipped) == (1, 0)
        assert (second.created, second.skipped) == (0, 1)
        assert (second.database_items, second.database_chunks) == (1, 1)
        assert second.input_sha256 == second.database_sha256
        assert session.flush_pending_counts[0] == 1
        assert [transaction.committed for transaction in session.transactions] == [True, True]

    asyncio.run(exercise())


def test_duplicate_legacy_ids_fail_before_entering_transaction() -> None:
    """重复旧 ID 必须在开启事务之前被拒绝。"""
    async def exercise() -> None:
        session = FakeSession()
        service = LegacyKnowledgeImportService(session, FakeRepository(session))  # type: ignore[arg-type]

        with pytest.raises(DuplicateLegacyIdError, match="k0001"):
            await service.import_bundles([make_bundle(), make_bundle()])

        assert session.transactions == []

    asyncio.run(exercise())


def test_conflicting_existing_legacy_item_rolls_back_transaction() -> None:
    """同 ID 的不同持久化载荷必须触发冲突并让事务回滚。"""
    async def exercise() -> None:
        existing = model_from_bundle(make_bundle())
        session = FakeSession([existing])
        service = LegacyKnowledgeImportService(session, FakeRepository(session))  # type: ignore[arg-type]

        with pytest.raises(LegacyKnowledgeConflictError, match="k0001"):
            await service.import_bundles([make_bundle(retrieval_text="冲突的检索文本")])

        assert session.transactions[0].entered is True
        assert session.transactions[0].rolled_back is True
        assert session.transactions[0].committed is False

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: setattr(item, "legacy_id", None),
        lambda item: item.chunks.clear(),
        lambda item: item.chunks.append(item.chunks[0]),
        lambda item: item.chunks[0].metadata_.pop("legacy_chunk_id"),
        lambda item: item.chunks[0].metadata_.update({"source_line": "7"}),
    ],
)
def test_bundle_from_malformed_model_raises_domain_error(mutate: object) -> None:
    """损坏 ORM 数据不得被静默转换，异常须包含历史 ID 上下文。"""
    item = model_from_bundle(make_bundle())
    mutate(item)  # type: ignore[operator]

    with pytest.raises(LegacyKnowledgeImportError, match="k0001|None"):
        bundle_from_model(item)
