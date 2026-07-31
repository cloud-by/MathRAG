"""认证 API 与 Cookie 生命周期。"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, settings
from app.modules.auth.dependencies import (
    AuthenticatedPrincipal,
    get_auth_service,
    require_logout_csrf,
    require_origin,
    get_current_principal,
)
from app.modules.auth.schemas import AuthUserRead, LoginRequest
from app.modules.auth.service import AuthService, IssuedSession


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=AuthUserRead)
async def login(
    request: LoginRequest,
    response: Response,
    _origin: None = Depends(require_origin),
    service: AuthService = Depends(get_auth_service),
) -> AuthUserRead:
    issued = await service.login(request.username, request.password, datetime.now(UTC))
    set_auth_cookies(response, issued, settings)
    return AuthUserRead.model_validate(issued.user)


@router.get("/me", response_model=AuthUserRead)
async def me(
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
    service: AuthService = Depends(get_auth_service),
) -> AuthUserRead:
    return AuthUserRead.model_validate(await service.get_user(principal.user_id))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    principal: AuthenticatedPrincipal = Depends(require_logout_csrf),
    service: AuthService = Depends(get_auth_service),
) -> None:
    await service.logout(principal.session_id, datetime.now(UTC))
    delete_auth_cookies(response, settings)


def set_auth_cookies(
    response: Response,
    issued: IssuedSession,
    config: Settings,
) -> None:
    secure = config.APP_ENV != "development"
    response.set_cookie(
        key=config.session_cookie_name,
        value=issued.raw_token,
        max_age=config.SESSION_TTL_SECONDS,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        key=config.csrf_cookie_name,
        value=issued.csrf_token,
        max_age=config.SESSION_TTL_SECONDS,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )


def delete_auth_cookies(response: Response, config: Settings) -> None:
    secure = config.APP_ENV != "development"
    response.delete_cookie(
        key=config.session_cookie_name,
        path="/",
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key=config.csrf_cookie_name,
        path="/",
        secure=secure,
        httponly=False,
        samesite="lax",
    )
