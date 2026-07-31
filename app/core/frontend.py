from __future__ import annotations

from pathlib import Path, PurePosixPath

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


RESERVED_PREFIXES = ("api/", "health", "docs", "redoc", "openapi.json")
INDEX_CACHE_CONTROL = "no-cache"
ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


def _accepts_html(scope: dict[str, object]) -> bool:
    accept = Headers(scope=scope).get("accept", "*/*")
    for raw_item in accept.split(","):
        parts = [part.strip() for part in raw_item.split(";")]
        media_type = parts[0].lower()
        quality = 1.0
        for parameter in parts[1:]:
            if parameter.lower().startswith("q="):
                try:
                    quality = float(parameter[2:])
                except ValueError:
                    quality = 0.0
        if quality > 0 and media_type in {
            "text/html",
            "application/xhtml+xml",
            "*/*",
        }:
            return True
    return False


def _is_reserved(path: str) -> bool:
    return path.startswith(RESERVED_PREFIXES)


def _may_fall_back(path: str, scope: dict[str, object]) -> bool:
    method = str(scope.get("method", "")).upper()
    return (
        method in {"GET", "HEAD"}
        and not _is_reserved(path)
        and PurePosixPath(path).suffix == ""
        and _accepts_html(scope)
    )


class VueStaticFiles(StaticFiles):
    """提供 Vite 产物，并严格限制客户端路由回退。"""

    def __init__(self, directory: Path) -> None:
        self.dist_dir = directory
        super().__init__(directory=str(directory), html=False, check_dir=False)

    async def check_config(self) -> None:
        # 构建产物缺失不应阻止 API 启动；请求首页时返回明确的 503。
        return None

    async def get_response(
        self,
        path: str,
        scope: dict[str, object],
    ):
        normalized = path.lstrip("/")
        method = str(scope.get("method", "")).upper()
        if method not in {"GET", "HEAD"} or _is_reserved(normalized):
            raise HTTPException(status_code=404)

        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or not _may_fall_back(normalized, scope):
                raise
            index_file = self.dist_dir / "index.html"
            if not index_file.is_file():
                return JSONResponse(
                    status_code=503,
                    content={"detail": "前端构建产物不可用。"},
                )
            response = FileResponse(index_file, media_type="text/html")
            response.headers["Cache-Control"] = INDEX_CACHE_CONTROL
            return response

        response.headers["Cache-Control"] = (
            INDEX_CACHE_CONTROL
            if normalized == "index.html"
            else ASSET_CACHE_CONTROL
        )
        return response


def mount_vue_frontend(app: FastAPI, dist_dir: Path) -> None:
    """必须在所有 API 与系统路由之后调用。"""
    app.mount("/", VueStaticFiles(dist_dir), name="frontend")
