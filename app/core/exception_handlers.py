from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.core.errors import AppError


def _is_v1_request(request: Request) -> bool:
    path = request.url.path
    return path == "/api/v1" or path.startswith("/api/v1/")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _error_envelope(
    request: Request,
    *,
    code: str,
    message: str,
    details: object,
) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": _request_id(request),
            "details": details,
        }
    }


def install_exception_handlers(app: FastAPI) -> None:
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if not _is_v1_request(request):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.message},
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(
                request,
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ),
        )

    async def handle_http_exception(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        if not _is_v1_request(request):
            return await http_exception_handler(request, exc)
        message = exc.detail if isinstance(exc.detail, str) else "请求失败。"
        return JSONResponse(
            status_code=exc.status_code,
            headers=exc.headers,
            content=_error_envelope(
                request,
                code="HTTP_ERROR",
                message=message,
                details={},
            ),
        )

    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        if not _is_v1_request(request):
            return await request_validation_exception_handler(request, exc)
        try:
            validation_errors = exc.errors(include_input=False)
        except TypeError:
            # FastAPI 包装器可能不透传 Pydantic 的 include_input 参数。
            validation_errors = exc.errors()
        details = [
            {
                "loc": error["loc"],
                "type": error["type"],
                "msg": error["msg"],
            }
            for error in validation_errors
        ]
        return JSONResponse(
            status_code=422,
            content=_error_envelope(
                request,
                code="REQUEST_VALIDATION_FAILED",
                message="请求参数校验失败。",
                details=details,
            ),
        )

    async def handle_unknown_exception(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del exc
        if not _is_v1_request(request):
            return JSONResponse(
                status_code=500,
                headers={"X-Request-ID": _request_id(request)},
                content={"detail": "Internal Server Error"},
            )
        return JSONResponse(
            status_code=500,
            headers={"X-Request-ID": _request_id(request)},
            content=_error_envelope(
                request,
                code="INTERNAL_ERROR",
                message="服务器内部错误。",
                details={},
            ),
        )

    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(HTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unknown_exception)
