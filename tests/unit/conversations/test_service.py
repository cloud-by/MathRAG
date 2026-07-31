"""会话领域服务测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.core.errors import AppError
from app.modules.conversations.models import Conversation
from app.modules.conversations.service import ConversationService


class AsyncContext:
    async def __aenter__(self) -> object:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> "AsyncContext":
        return self


class FakeSessionFactory:
    def __call__(self) -> AsyncContext:
        return AsyncContext()


class FakeRepository:
    def __init__(self) -> None:
        self.conversation: Conversation | None = None

    def add(self, conversation: Conversation) -> None:
        self.conversation = conversation

    async def get_owned(self, conversation_id, user_id):
        if (
            self.conversation is not None
            and self.conversation.id == conversation_id
            and self.conversation.user_id == user_id
        ):
            return self.conversation
        return None

    async def list_owned(self, user_id, *, status, offset, limit):
        return ([], 0)

    async def update_owned(self, conversation_id, user_id, *, values):
        conversation = await self.get_owned(conversation_id, user_id)
        if conversation is None:
            return None
        for key, value in values.items():
            setattr(conversation, key, value)
        return conversation

    async def list_owned_messages(self, conversation_id, user_id, *, offset, limit):
        if await self.get_owned(conversation_id, user_id) is None:
            return None
        return ([], 0)


def test_empty_update_is_rejected_without_repository_update() -> None:
    repository = FakeRepository()
    service = ConversationService(
        FakeSessionFactory(),  # type: ignore[arg-type]
        repository_factory=lambda _session: repository,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.update(uuid4(), uuid4(), {}))

    assert exc_info.value.code == "REQUEST_VALIDATION_FAILED"
    assert exc_info.value.status_code == 422


def test_archive_is_idempotent_and_cross_owner_is_not_found() -> None:
    repository = FakeRepository()
    owner_id = uuid4()
    conversation_id = uuid4()
    now = datetime.now(UTC)
    repository.conversation = Conversation(
        id=conversation_id,
        user_id=owner_id,
        title="会话",
        status="active",
        created_at=now,
        updated_at=now,
    )
    service = ConversationService(
        FakeSessionFactory(),  # type: ignore[arg-type]
        repository_factory=lambda _session: repository,
    )

    asyncio.run(service.archive(conversation_id, owner_id))
    asyncio.run(service.archive(conversation_id, owner_id))
    assert repository.conversation.status == "archived"
    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.get(conversation_id, uuid4()))
    assert exc_info.value.code == "CONVERSATION_NOT_FOUND"
