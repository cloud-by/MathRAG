"""按所有者隔离的会话与消息查询。"""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.conversations.models import Conversation, Message


class ConversationRepository:
    """所有资源访问都在 SQL 中包含 user_id 归属条件。"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def add(self, conversation: Conversation) -> None:
        self._session.add(conversation)

    async def get_owned(
        self,
        conversation_id: UUID,
        user_id: UUID,
    ) -> Conversation | None:
        return await self._session.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )

    async def list_owned(
        self,
        user_id: UUID,
        *,
        status: str,
        offset: int,
        limit: int,
    ) -> tuple[list[Conversation], int]:
        filters = (
            Conversation.user_id == user_id,
            Conversation.status == status,
        )
        items = list(
            (
                await self._session.scalars(
                    select(Conversation)
                    .where(*filters)
                    .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        )
        total = int(
            await self._session.scalar(
                select(func.count()).select_from(Conversation).where(*filters)
            )
            or 0
        )
        return items, total

    async def update_owned(
        self,
        conversation_id: UUID,
        user_id: UUID,
        *,
        values: Mapping[str, object],
    ) -> Conversation | None:
        statement = (
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
            .values(**values)
            .returning(Conversation)
        )
        return (await self._session.scalars(statement)).one_or_none()

    async def list_owned_messages(
        self,
        conversation_id: UUID,
        user_id: UUID,
        *,
        offset: int,
        limit: int,
    ) -> tuple[list[Message], int] | None:
        owned = await self._session.scalar(
            select(Conversation.id).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
            )
        )
        if owned is None:
            return None
        filters = (
            Message.conversation_id == Conversation.id,
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
        items = list(
            (
                await self._session.scalars(
                    select(Message)
                    .join(Conversation, Message.conversation_id == Conversation.id)
                    .where(*filters)
                    .order_by(Message.created_at.asc(), Message.id.asc())
                    .offset(offset)
                    .limit(limit)
                )
            ).all()
        )
        total = int(
            await self._session.scalar(
                select(func.count())
                .select_from(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(*filters)
            )
            or 0
        )
        return items, total
