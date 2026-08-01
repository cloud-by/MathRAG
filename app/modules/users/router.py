"""教师与管理员的用户管理 HTTP API。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status

from app.modules.auth.dependencies import (
    require_user_manager,
    require_user_manager_csrf,
)
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.users.dependencies import get_user_service
from app.modules.users.schemas import (
    ManagedUserRead,
    UserCreate,
    UserPage,
    UserPasswordReset,
    UserUpdate,
)
from app.modules.users.service import UserService
from app.modules.users.types import UserActor, UserRole, UserStatus


router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("", response_model=UserPage)
async def list_users(
    q: str | None = Query(default=None, min_length=1, max_length=320),
    role: UserRole | None = Query(default=None),
    status_filter: UserStatus | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    principal: AuthenticatedPrincipal = Depends(require_user_manager),
    service: UserService = Depends(get_user_service),
) -> UserPage:
    return await service.list_managed_users(
        UserActor(principal.user_id, principal.role),
        query=q,
        role=role,
        status=status_filter,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=ManagedUserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: UserCreate,
    principal: AuthenticatedPrincipal = Depends(require_user_manager_csrf),
    service: UserService = Depends(get_user_service),
) -> ManagedUserRead:
    return await service.create_managed_user(
        UserActor(principal.user_id, principal.role),
        request,
    )


@router.get("/{user_id}", response_model=ManagedUserRead)
async def get_user(
    user_id: UUID,
    principal: AuthenticatedPrincipal = Depends(require_user_manager),
    service: UserService = Depends(get_user_service),
) -> ManagedUserRead:
    return await service.get_managed_user(
        UserActor(principal.user_id, principal.role),
        user_id,
    )


@router.patch("/{user_id}", response_model=ManagedUserRead)
async def update_user(
    user_id: UUID,
    request: UserUpdate,
    principal: AuthenticatedPrincipal = Depends(require_user_manager_csrf),
    service: UserService = Depends(get_user_service),
) -> ManagedUserRead:
    return await service.update_managed_user(
        UserActor(principal.user_id, principal.role),
        user_id,
        request,
        datetime.now(UTC),
    )


@router.post("/{user_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(
    user_id: UUID,
    request: UserPasswordReset,
    principal: AuthenticatedPrincipal = Depends(require_user_manager_csrf),
    service: UserService = Depends(get_user_service),
) -> Response:
    await service.reset_managed_password(
        UserActor(principal.user_id, principal.role),
        user_id,
        request.password,
        datetime.now(UTC),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
