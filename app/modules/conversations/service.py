"""会话 CRUD、归档和消息读取用例。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.modules.conversations.errors import ConversationNotFoundError
from app.modules.conversations.models import Conversation, Message
from app.modules.conversations.repository import ConversationRepository
from app.modules.conversations.schemas import (
    ConversationPage,
    ConversationRead,
    MessagePage,
    MessageRead,
)


class ConversationRepositoryProtocol(Protocol):
    def add(self, conversation: Conversation) -> None: ...

    async def get_owned(self, conversation_id: UUID, user_id: UUID) -> Conversation | None: ...

    async def list_owned(
        self,
        user_id: UUID,
        *,
        status: str,
        offset: int,
        limit: int,
    ) -> tuple[list[Conversation], int]: ...

    async def update_owned(
        self,
        conversation_id: UUID,
        user_id: UUID,
        *,
        values: Mapping[str, object],
    ) -> Conversation | None: ...

    async def list_owned_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[Message], int] | None: ...


class ConversationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        repository_factory: Callable[[AsyncSession], ConversationRepositoryProtocol] = (
            ConversationRepository
        ),
    ) -> None:
        self._session_factory = session_factory
        self._repository_factory = repository_factory

    async def create(self, user_id: UUID, title: str) -> ConversationRead:
        now = datetime.now(UTC)
        conversation = Conversation(
            id=uuid4(),
            user_id=user_id,
            title=title,
            status="active",
            created_at=now,
            updated_at=now,
        )
        async with self._session_factory() as session:
            async with session.begin():
                self._repository_factory(session).add(conversation)
        return ConversationRead.model_validate(conversation)

    async def get(self, conversation_id: UUID, user_id: UUID) -> ConversationRead:
        async with self._session_factory() as session:
            conversation = await self._repository_factory(session).get_owned(
                conversation_id,
                user_id,
            )
        if conversation is None:
            raise ConversationNotFoundError()
        return ConversationRead.model_validate(conversation)

    async def list(
        self,
        user_id: UUID,
        *,
        status: str,
        page: int,
        page_size: int,
    ) -> ConversationPage:
        async with self._session_factory() as session:
            items, total = await self._repository_factory(session).list_owned(
                user_id,
                status=status,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
        return ConversationPage(
            items=[ConversationRead.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
            total=total,
        )

    async def update(
        self,
        conversation_id: UUID,
        user_id: UUID,
        values: Mapping[str, object],
    ) -> ConversationRead:
        effective_values = {key: value for key, value in values.items() if value is not None}
        if not effective_values:
            raise AppError(
                code="REQUEST_VALIDATION_FAILED",
                message="至少提供一个需要更新的字段。",
                status_code=422,
            )
        effective_values["updated_at"] = datetime.now(UTC)
        async with self._session_factory() as session:
            async with session.begin():
                conversation = await self._repository_factory(session).update_owned(
                    conversation_id,
                    user_id,
                    values=effective_values,
                )
        if conversation is None:
            raise ConversationNotFoundError()
        return ConversationRead.model_validate(conversation)

    async def archive(self, conversation_id: UUID, user_id: UUID) -> None:
        await self.update(
            conversation_id,
            user_id,
            {"status": "archived"},
        )

    async def list_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        *,
        page: int,
        page_size: int,
    ) -> MessagePage:
        async with self._session_factory() as session:
            result = await self._repository_factory(session).list_owned_messages(
                conversation_id,
                user_id,
                offset=(page - 1) * page_size,
                limit=page_size,
            )
        if result is None:
            raise ConversationNotFoundError()
        items, total = result
        return MessagePage(
            items=[MessageRead.model_validate(item) for item in items],
            page=page,
            page_size=page_size,
            total=total,
        )
