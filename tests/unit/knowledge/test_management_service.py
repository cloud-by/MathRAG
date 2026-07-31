"""权限感知知识读取服务测试。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.core.errors import AppError, ConfigurationError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
    KnowledgeNotFoundError,
    KnowledgeRevisionConflictError,
)
from app.modules.knowledge.management_repository import IndexingSnapshot
from app.modules.knowledge.management_schemas import (
    KnowledgeItemCreate,
    KnowledgeItemUpdate,
)
from app.modules.knowledge.management_service import KnowledgeManagementService
from app.modules.knowledge.models import KnowledgeItem


class AsyncContext:
    async def __aenter__(self) -> object:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeSessionFactory:
    def __call__(self) -> AsyncContext:
        return AsyncContext()


class FakeManagementRepository:
    """保留真实可见性和排序语义的内存仓储。"""

    def __init__(self, items: list[KnowledgeItem]) -> None:
        self.items = items

    @staticmethod
    def _visible(
        item: KnowledgeItem,
        principal: AuthenticatedPrincipal,
    ) -> bool:
        return principal.role == "admin" or (
            item.visibility == "public" and item.status == "ready"
        )

    async def get_visible(
        self,
        item_id: UUID,
        principal: AuthenticatedPrincipal,
    ) -> KnowledgeItem | None:
        return next(
            (
                item
                for item in self.items
                if item.id == item_id and self._visible(item, principal)
            ),
            None,
        )

    async def list_visible(
        self,
        principal: AuthenticatedPrincipal,
        *,
        status: str | None,
        visibility: str | None,
        category: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[KnowledgeItem], int]:
        selected = [
            item
            for item in self.items
            if self._visible(item, principal)
            and (status is None or item.status == status)
            and (visibility is None or item.visibility == visibility)
            and (category is None or item.category == category)
        ]
        selected.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        return selected[offset : offset + limit], len(selected)


def _principal(role: str) -> AuthenticatedPrincipal:
    return AuthenticatedPrincipal(
        user_id=uuid4(),
        session_id=uuid4(),
        username=f"{role}-reader",
        role=role,  # type: ignore[arg-type]
        session_token_hash=b"session-token-hash",
    )


def _item(
    *,
    title: str,
    visibility: str = "public",
    status: str = "ready",
    updated_at: datetime | None = None,
) -> KnowledgeItem:
    now = updated_at or datetime.now(UTC)
    return KnowledgeItem(
        id=uuid4(),
        legacy_id=None,
        owner_id=None,
        category="代数",
        title=title,
        keywords=["测试"],
        content=f"{title}内容",
        example="",
        steps=["步骤一"],
        difficulty="easy",
        visibility=visibility,
        status=status,
        revision=1,
        created_at=now,
        updated_at=now,
    )


def _service(items: list[KnowledgeItem]) -> KnowledgeManagementService:
    repository = FakeManagementRepository(items)
    return KnowledgeManagementService(
        FakeSessionFactory(),  # type: ignore[arg-type]
        repository_factory=lambda _session: repository,
    )


@pytest.mark.parametrize(
    "hidden_item",
    [
        _item(title="私有条目", visibility="private"),
        _item(title="失败条目", status="failed"),
        _item(title="归档条目", status="archived"),
    ],
)
def test_user_cannot_observe_hidden_item(hidden_item: KnowledgeItem) -> None:
    service = _service([hidden_item])

    with pytest.raises(KnowledgeNotFoundError) as exc_info:
        asyncio.run(service.get(hidden_item.id, _principal("user")))

    assert exc_info.value.code == "KNOWLEDGE_NOT_FOUND"
    assert exc_info.value.status_code == 404


def test_missing_and_hidden_items_have_the_same_public_error() -> None:
    private_item = _item(title="私有条目", visibility="private")
    service = _service([private_item])
    principal = _principal("user")

    errors: list[KnowledgeNotFoundError] = []
    for item_id in (private_item.id, uuid4()):
        with pytest.raises(KnowledgeNotFoundError) as exc_info:
            asyncio.run(service.get(item_id, principal))
        errors.append(exc_info.value)

    assert [(error.code, error.message, error.status_code) for error in errors] == [
        ("KNOWLEDGE_NOT_FOUND", "知识条目不存在。", 404),
        ("KNOWLEDGE_NOT_FOUND", "知识条目不存在。", 404),
    ]


def test_admin_can_read_private_and_unready_items() -> None:
    private_item = _item(title="私有草稿", visibility="private", status="draft")
    service = _service([private_item])

    result = asyncio.run(service.get(private_item.id, _principal("admin")))

    assert result.id == private_item.id
    assert result.visibility == "private"
    assert result.status == "draft"
    assert "ingestion_job_id" not in result.model_dump()


def test_list_applies_filters_and_returns_safe_page() -> None:
    now = datetime.now(UTC)
    algebra = _item(title="代数条目", updated_at=now)
    geometry = _item(title="几何条目", updated_at=now.replace(microsecond=1))
    geometry.category = "几何"
    service = _service([algebra, geometry, _item(title="私有", visibility="private")])

    page = asyncio.run(
        service.list(
            _principal("user"),
            status="ready",
            visibility="public",
            category="代数",
            page=1,
            page_size=100,
        )
    )

    assert [item.id for item in page.items] == [algebra.id]
    assert (page.page, page.page_size, page.total) == (1, 100, 1)
    assert "ingestion_job_id" not in page.items[0].model_dump()


def test_list_uses_public_api_pagination_and_filter_defaults() -> None:
    item = _item(title="默认列表条目")
    service = _service([item])

    page = asyncio.run(service.list(_principal("user")))

    assert [listed.id for listed in page.items] == [item.id]
    assert (page.page, page.page_size, page.total) == (1, 20, 1)


@pytest.mark.parametrize(
    ("page", "page_size"),
    [(0, 20), (1, 0), (1, 101)],
)
def test_list_rejects_invalid_pagination(page: int, page_size: int) -> None:
    service = _service([])

    with pytest.raises(AppError) as exc_info:
        asyncio.run(
            service.list(
                _principal("user"),
                status=None,
                visibility=None,
                category=None,
                page=page,
                page_size=page_size,
            )
        )

    assert exc_info.value.code == "REQUEST_VALIDATION_FAILED"
    assert exc_info.value.status_code == 422


class RecordingTransaction:
    """记录短事务边界，异常退出时保留 rollback 证据。"""

    def __init__(self, events: list[str], label: str) -> None:
        self.events = events
        self.label = label

    async def __aenter__(self) -> RecordingTransaction:
        self.events.append(f"{self.label}.begin")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: object,
    ) -> None:
        suffix = "commit" if exc_type is None else "rollback"
        self.events.append(f"{self.label}.{suffix}")


class RecordingSession:
    def __init__(self, events: list[str], label: str) -> None:
        self.events = events
        self.label = label

    async def __aenter__(self) -> RecordingSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.events.append(f"{self.label}.close")

    def begin(self) -> RecordingTransaction:
        return RecordingTransaction(self.events, self.label)


class RecordingSessionFactory:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls = 0

    def __call__(self) -> RecordingSession:
        self.calls += 1
        return RecordingSession(self.events, f"tx{self.calls}")


class RecordingProvider:
    model = "management-embedding-v1"
    dimensions = 1024

    def __init__(self, events: list[str], *, failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.events.append("embedding.start")
        self.calls.append(list(texts))
        if self.failure is not None:
            raise self.failure
        self.events.append("embedding.end")
        return [[1.0, *([0.0] * 1023)]]


class WriteState:
    def __init__(self, item: KnowledgeItem | None = None) -> None:
        self.item = item
        self.chunk_status: str | None = None
        self.complete_conflict = False


class RecordingWriteRepository:
    """只模拟 service 编排，SQL CAS 由集成测试覆盖。"""

    def __init__(self, session: RecordingSession, state: WriteState) -> None:
        self.session = session
        self.state = state

    async def create_indexing(
        self,
        *,
        owner_id: UUID,
        values: Mapping[str, object],
    ) -> IndexingSnapshot:
        self.session.events.append("item.indexing")
        now = datetime.now(UTC)
        item = KnowledgeItem(
            id=uuid4(),
            legacy_id=None,
            owner_id=owner_id,
            category=values["category"],
            title=values["title"],
            keywords=values["keywords"],
            content=values["content"],
            example=values["example"],
            steps=values["steps"],
            difficulty=values["difficulty"],
            visibility=values["visibility"],
            status="indexing",
            revision=1,
            created_at=now,
            updated_at=now,
        )
        self.state.item = item
        self.state.chunk_status = "pending"
        return IndexingSnapshot(
            item_id=item.id,
            revision=1,
            chunk_id=uuid4(),
            retrieval_text=f"检索：{item.title}",
            answer_context=f"回答：{item.title}",
        )

    async def update_with_revision(
        self,
        item_id: UUID,
        *,
        expected_revision: int,
        values: Mapping[str, object],
        reindex: bool,
    ) -> IndexingSnapshot | KnowledgeItem | None:
        item = self.state.item
        if item is None or item.id != item_id or item.status == "archived":
            return None
        if item.revision != expected_revision:
            raise KnowledgeRevisionConflictError()
        for name, value in values.items():
            setattr(item, name, value)
        item.revision += 1
        if not reindex:
            self.session.events.append("item.visibility")
            return item
        item.status = "indexing"
        self.state.chunk_status = "pending"
        self.session.events.append("item.indexing")
        return IndexingSnapshot(
            item_id=item.id,
            revision=item.revision,
            chunk_id=uuid4(),
            retrieval_text=f"检索：{item.title}",
            answer_context=f"回答：{item.title}",
        )

    async def archive_with_revision(
        self,
        item_id: UUID,
        expected_revision: int,
    ) -> bool:
        item = self.state.item
        if item is None or item.id != item_id:
            return False
        if item.status == "archived" or item.revision != expected_revision:
            raise KnowledgeRevisionConflictError()
        item.status = "archived"
        item.revision += 1
        self.session.events.append("item.archived")
        return True

    async def complete_indexing(
        self,
        snapshot: IndexingSnapshot,
        vector: Sequence[float],
        model: str,
    ) -> KnowledgeItem | None:
        assert len(vector) == 1024
        assert model == RecordingProvider.model
        if self.state.complete_conflict:
            return None
        item = self.state.item
        assert item is not None
        assert (item.id, item.revision) == (snapshot.item_id, snapshot.revision)
        item.status = "ready"
        self.state.chunk_status = "ready"
        self.session.events.append("item.ready")
        return item

    async def fail_indexing(self, snapshot: IndexingSnapshot) -> None:
        item = self.state.item
        if item is None or (item.id, item.revision) != (
            snapshot.item_id,
            snapshot.revision,
        ):
            return
        item.status = "failed"
        self.state.chunk_status = "failed"
        self.session.events.append("item.failed")


def _create_request() -> KnowledgeItemCreate:
    return KnowledgeItemCreate(
        category="代数",
        title="一元一次方程",
        keywords=["方程", "移项"],
        content="含有一个未知数且次数为一的等式。",
        example="2x+1=5",
        steps=["移项", "合并同类项"],
        difficulty="easy",
        visibility="public",
    )


def _write_service(
    *,
    events: list[str],
    state: WriteState | None = None,
    provider: RecordingProvider | None = None,
) -> tuple[KnowledgeManagementService, WriteState, RecordingProvider]:
    current_state = state or WriteState()
    current_provider = provider or RecordingProvider(events)
    return (
        KnowledgeManagementService(
            RecordingSessionFactory(events),  # type: ignore[arg-type]
            current_provider,
            repository_factory=lambda session: RecordingWriteRepository(
                session, current_state  # type: ignore[arg-type]
            ),
        ),
        current_state,
        current_provider,
    )


def test_create_embeds_between_two_closed_short_transactions() -> None:
    events: list[str] = []
    service, state, provider = _write_service(events=events)

    result = asyncio.run(service.create(uuid4(), _create_request()))

    assert result.status == "ready"
    assert state.chunk_status == "ready"
    assert provider.calls and provider.calls[0][0].startswith("检索：")
    assert events == [
        "tx1.begin",
        "item.indexing",
        "tx1.commit",
        "tx1.close",
        "embedding.start",
        "embedding.end",
        "tx2.begin",
        "item.ready",
        "tx2.commit",
        "tx2.close",
    ]


def test_visibility_only_update_increments_revision_without_embedding() -> None:
    item = _item(title="只改可见性")
    item.revision = 3
    events: list[str] = []
    state = WriteState(item)
    service, _state, provider = _write_service(events=events, state=state)

    result = asyncio.run(
        service.update(
            item.id,
            KnowledgeItemUpdate(revision=3, visibility="private"),
        )
    )

    assert (result.revision, result.visibility, result.status) == (4, "private", "ready")
    assert provider.calls == []
    assert events == [
        "tx1.begin",
        "item.visibility",
        "tx1.commit",
        "tx1.close",
    ]


def test_stale_update_returns_stable_revision_conflict_without_embedding() -> None:
    item = _item(title="并发条目")
    item.revision = 4
    events: list[str] = []
    service, _state, provider = _write_service(events=events, state=WriteState(item))

    with pytest.raises(KnowledgeRevisionConflictError) as exc_info:
        asyncio.run(
            service.update(
                item.id,
                KnowledgeItemUpdate(revision=3, content="过期编辑"),
            )
        )

    assert (exc_info.value.code, exc_info.value.status_code) == (
        "KNOWLEDGE_REVISION_CONFLICT",
        409,
    )
    assert provider.calls == []
    assert events == ["tx1.begin", "tx1.rollback", "tx1.close"]


@pytest.mark.parametrize(
    "failure",
    [
        EmbeddingInputError("secret-input-body"),
        EmbeddingResponseError("secret-response-body"),
        EmbeddingUnavailableError("secret-provider-body"),
    ],
)
def test_known_embedding_failure_marks_failed_and_hides_details(
    failure: Exception,
) -> None:
    events: list[str] = []
    provider = RecordingProvider(events, failure=failure)
    service, state, _provider = _write_service(events=events, provider=provider)

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.create(uuid4(), _create_request()))

    assert exc_info.value.code == "EMBEDDING_UNAVAILABLE"
    assert exc_info.value.status_code == 502
    assert "secret" not in exc_info.value.message
    assert exc_info.value.__cause__ is None
    assert state.item is not None and state.item.status == "failed"
    assert state.chunk_status == "failed"
    assert events == [
        "tx1.begin",
        "item.indexing",
        "tx1.commit",
        "tx1.close",
        "embedding.start",
        "tx2.begin",
        "item.failed",
        "tx2.commit",
        "tx2.close",
    ]


def test_unknown_provider_programming_error_propagates_without_marking_failed() -> None:
    events: list[str] = []
    provider = RecordingProvider(events, failure=RuntimeError("programming-bug"))
    service, state, _provider = _write_service(events=events, provider=provider)

    with pytest.raises(RuntimeError, match="programming-bug"):
        asyncio.run(service.create(uuid4(), _create_request()))

    assert state.item is not None and state.item.status == "indexing"
    assert state.chunk_status == "pending"
    assert events == [
        "tx1.begin",
        "item.indexing",
        "tx1.commit",
        "tx1.close",
        "embedding.start",
    ]


def test_missing_provider_raises_configuration_error_without_marking_failed() -> None:
    events: list[str] = []
    state = WriteState()
    service = KnowledgeManagementService(
        RecordingSessionFactory(events),  # type: ignore[arg-type]
        repository_factory=lambda session: RecordingWriteRepository(
            session, state  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(ConfigurationError, match="Provider"):
        asyncio.run(service.create(uuid4(), _create_request()))

    assert state.item is not None and state.item.status == "indexing"
    assert state.chunk_status == "pending"
    assert events == [
        "tx1.begin",
        "item.indexing",
        "tx1.commit",
        "tx1.close",
    ]


def test_concurrent_change_before_embedding_completion_returns_409() -> None:
    events: list[str] = []
    state = WriteState()
    state.complete_conflict = True
    service, _state, _provider = _write_service(events=events, state=state)

    with pytest.raises(KnowledgeRevisionConflictError):
        asyncio.run(service.create(uuid4(), _create_request()))

    assert events[-3:] == ["tx2.begin", "tx2.commit", "tx2.close"]


def test_archive_increments_revision_conflicts_when_repeated_and_missing_is_404() -> None:
    item = _item(title="待归档")
    item.revision = 7
    events: list[str] = []
    service, _state, provider = _write_service(events=events, state=WriteState(item))

    asyncio.run(service.archive(item.id, 7))

    assert (item.status, item.revision) == ("archived", 8)
    assert provider.calls == []
    with pytest.raises(KnowledgeRevisionConflictError):
        asyncio.run(service.archive(item.id, 8))

    missing_service, _missing_state, _missing_provider = _write_service(
        events=[],
        state=WriteState(),
    )
    with pytest.raises(KnowledgeNotFoundError):
        asyncio.run(missing_service.archive(uuid4(), 1))


def test_write_payload_rejects_internal_status_and_revision_fields() -> None:
    events: list[str] = []
    service, _state, provider = _write_service(events=events)
    invalid_create = {**_create_request().model_dump(), "status": "ready"}

    with pytest.raises(Exception):
        asyncio.run(service.create(uuid4(), invalid_create))
    with pytest.raises(Exception):
        asyncio.run(
            service.update(
                uuid4(),
                {"revision": 1, "visibility": "public", "status": "ready"},
            )
        )

    assert events == []
    assert provider.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {"revision": 1, "content": None},
        {"revision": 1},
    ],
)
def test_update_rejects_invalid_payload_before_opening_transaction(
    payload: dict[str, object],
) -> None:
    events: list[str] = []
    service, _state, provider = _write_service(events=events)

    with pytest.raises(ValidationError):
        asyncio.run(service.update(uuid4(), payload))

    assert events == []
    assert provider.calls == []
