"""知识条目管理 API。"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status

from app.infrastructure.database.session import get_session_factory
from app.infrastructure.embedding.provider import get_embedding_provider
from app.modules.auth.dependencies import require_admin_csrf, require_password_ready
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.knowledge.management_schemas import (
    KnowledgeItemCreate,
    KnowledgeItemPage,
    KnowledgeItemRead,
    KnowledgeItemUpdate,
)
from app.modules.knowledge.management_service import KnowledgeManagementService


router = APIRouter(prefix="/api/v1/knowledge-items", tags=["knowledge-items"])


def get_knowledge_read_service() -> KnowledgeManagementService:
    """读取路径只取得数据库会话，不初始化向量 Provider。"""
    return KnowledgeManagementService(get_session_factory())


def get_knowledge_management_service() -> KnowledgeManagementService:
    """在请求依赖解析阶段惰性取得数据库和向量 Provider。"""
    return KnowledgeManagementService(
        get_session_factory(),
        get_embedding_provider(),
    )


@router.get("", response_model=KnowledgeItemPage)
async def list_knowledge_items(
    status_filter: Literal[
        "draft", "indexing", "ready", "failed", "archived"
    ]
    | None = Query(default=None, alias="status"),
    visibility: Literal["public", "private"] | None = Query(default=None),
    category: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
    service: KnowledgeManagementService = Depends(get_knowledge_read_service),
) -> KnowledgeItemPage:
    return await service.list(
        principal,
        status=status_filter,
        visibility=visibility,
        category=category,
        page=page,
        page_size=page_size,
    )


@router.post(
    "",
    response_model=KnowledgeItemRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_knowledge_item(
    request: KnowledgeItemCreate,
    principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: KnowledgeManagementService = Depends(
        get_knowledge_management_service
    ),
) -> KnowledgeItemRead:
    return await service.create(principal.user_id, request)


@router.get("/{item_id}", response_model=KnowledgeItemRead)
async def get_knowledge_item(
    item_id: UUID,
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
    service: KnowledgeManagementService = Depends(get_knowledge_read_service),
) -> KnowledgeItemRead:
    return await service.get(item_id, principal)


@router.patch("/{item_id}", response_model=KnowledgeItemRead)
async def update_knowledge_item(
    item_id: UUID,
    request: KnowledgeItemUpdate,
    _principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: KnowledgeManagementService = Depends(
        get_knowledge_management_service
    ),
) -> KnowledgeItemRead:
    return await service.update(item_id, request)


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_item(
    item_id: UUID,
    revision: int = Query(ge=1),
    _principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: KnowledgeManagementService = Depends(
        get_knowledge_management_service
    ),
) -> Response:
    await service.archive(item_id, revision)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
