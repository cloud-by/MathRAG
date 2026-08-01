"""会话与消息 REST API。"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status

from app.infrastructure.database.session import get_session_factory
from app.modules.auth.dependencies import (
    AuthenticatedPrincipal,
    require_password_ready,
    require_ready_csrf,
)
from app.modules.conversations.schemas import (
    ConversationCreate,
    ConversationPage,
    ConversationRead,
    ConversationUpdate,
    MessagePage,
)
from app.modules.conversations.service import ConversationService


router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def get_conversation_service() -> ConversationService:
    return ConversationService(get_session_factory())


@router.get("", response_model=ConversationPage)
async def list_conversations(
    status_filter: Literal["active", "archived"] = Query(default="active", alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationPage:
    return await service.list(
        principal.user_id,
        status=status_filter,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    request: ConversationCreate,
    principal: AuthenticatedPrincipal = Depends(require_ready_csrf),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    return await service.create(principal.user_id, request.title)


@router.get("/{conversation_id}/messages", response_model=MessagePage)
async def list_messages(
    conversation_id: UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
    service: ConversationService = Depends(get_conversation_service),
) -> MessagePage:
    return await service.list_messages(
        conversation_id,
        principal.user_id,
        page=page,
        page_size=page_size,
    )


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: UUID,
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    return await service.get(conversation_id, principal.user_id)


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: UUID,
    request: ConversationUpdate,
    principal: AuthenticatedPrincipal = Depends(require_ready_csrf),
    service: ConversationService = Depends(get_conversation_service),
) -> ConversationRead:
    return await service.update(
        conversation_id,
        principal.user_id,
        request.model_dump(exclude_unset=True, exclude_none=True),
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: UUID,
    principal: AuthenticatedPrincipal = Depends(require_ready_csrf),
    service: ConversationService = Depends(get_conversation_service),
) -> Response:
    await service.archive(conversation_id, principal.user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
