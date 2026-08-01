"""受认证和 CSRF 保护的持久化聊天路由。"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.infrastructure.database.session import get_session_factory
from app.modules.auth.dependencies import require_ready_csrf
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.rag.schemas import ChatV1Request, ChatV1Response
from app.modules.rag.service import ChatPersistenceService
from app.services.rag_pipeline import get_rag_pipeline


router = APIRouter(prefix="/api/v1", tags=["chat-v1"])


def get_chat_persistence_service() -> ChatPersistenceService:
    return ChatPersistenceService(
        get_session_factory(),
        get_rag_pipeline(),
        lambda: datetime.now(UTC),
    )


@router.post(
    "/chat",
    response_model=ChatV1Response,
    summary="持久化数学 RAG 问答",
    responses={
        401: {"description": "Session 无效"},
        403: {"description": "CSRF 或来源校验失败"},
        404: {"description": "会话不存在"},
        409: {"description": "会话或幂等请求状态冲突"},
        422: {"description": "请求校验失败"},
    },
)
async def chat_v1(
    request: ChatV1Request,
    principal: AuthenticatedPrincipal = Depends(require_ready_csrf),
    service: ChatPersistenceService = Depends(get_chat_persistence_service),
) -> ChatV1Response:
    result = await service.chat(
        principal=principal,
        conversation_id=request.conversation_id,
        client_request_id=request.client_request_id,
        question=request.question,
        top_k=request.top_k,
    )
    return ChatV1Response.model_validate(result.to_public_response())

