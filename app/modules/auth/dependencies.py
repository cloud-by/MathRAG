"""认证 principal、来源、CSRF 和角色依赖。"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from urllib.parse import urlsplit

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyCookie

from app.core.config import Settings, settings
from app.core.errors import AppError
from app.infrastructure.database.session import get_session_factory
from app.modules.auth.security import verify_csrf_token
from app.modules.auth.service import AuthService, AuthenticatedPrincipal


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
SESSION_COOKIE_SCHEME = APIKeyCookie(
    name=settings.session_cookie_name,
    scheme_name="SessionCookie",
    auto_error=False,
)


def get_auth_service() -> AuthService:
    return AuthService(
        get_session_factory(),
        session_ttl_seconds=settings.SESSION_TTL_SECONDS,
        csrf_secret=settings.SESSION_SECRET,
    )


async def get_current_principal(
    request: Request,
    raw_token: str | None = Security(SESSION_COOKIE_SCHEME),
    service: AuthService = Depends(get_auth_service),
) -> AuthenticatedPrincipal:
    if not raw_token:
        raise _invalid_session()
    return await service.resolve(raw_token, datetime.now(UTC))


async def get_logout_principal(
    request: Request,
    raw_token: str | None = Security(SESSION_COOKIE_SCHEME),
    service: AuthService = Depends(get_auth_service),
) -> AuthenticatedPrincipal:
    if not raw_token:
        raise _invalid_session()
    return await service.resolve_for_logout(raw_token)


def validate_request_origin(request: Request, config: Settings = settings) -> None:
    supplied_origin = request.headers.get("Origin")
    if supplied_origin:
        candidate = supplied_origin.strip()
    else:
        referer = request.headers.get("Referer", "").strip()
        parsed = urlsplit(referer)
        candidate = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""
    if candidate not in config.ALLOWED_ORIGINS:
        raise AppError(
            code="AUTH_ORIGIN_INVALID",
            message="请求来源不受信任。",
            status_code=403,
        )


async def require_origin(request: Request) -> None:
    validate_request_origin(request, settings)


def validate_csrf_request(
    request: Request,
    principal: AuthenticatedPrincipal,
    config: Settings = settings,
) -> None:
    if request.method.upper() not in UNSAFE_METHODS:
        return
    cookie_token = request.cookies.get(config.csrf_cookie_name, "")
    header_token = request.headers.get("X-CSRF-Token", "")
    if (
        not cookie_token
        or not header_token
        or not hmac.compare_digest(cookie_token, header_token)
        or not verify_csrf_token(
            cookie_token,
            principal.session_token_hash,
            config.SESSION_SECRET,
        )
    ):
        raise AppError(
            code="AUTH_CSRF_INVALID",
            message="CSRF 校验失败。",
            status_code=403,
        )
    validate_request_origin(request, config)


async def require_csrf(
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
) -> AuthenticatedPrincipal:
    validate_csrf_request(request, principal, settings)
    return principal


async def require_logout_csrf(
    request: Request,
    principal: AuthenticatedPrincipal = Depends(get_logout_principal),
) -> AuthenticatedPrincipal:
    validate_csrf_request(request, principal, settings)
    return principal


async def require_admin(
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
) -> AuthenticatedPrincipal:
    if principal.role != "admin":
        raise AppError(
            code="AUTH_FORBIDDEN",
            message="权限不足。",
            status_code=403,
        )
    return principal


async def require_admin_csrf(
    principal: AuthenticatedPrincipal = Depends(require_csrf),
) -> AuthenticatedPrincipal:
    return await require_admin(principal)


def _invalid_session() -> AppError:
    return AppError(
        code="AUTH_SESSION_INVALID",
        message="登录状态无效或已过期。",
        status_code=401,
    )
