"""权限感知知识读取服务测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from app.core.errors import AppError
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.errors import KnowledgeNotFoundError
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
