from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.core.config import settings
from app.core.middleware import RequestIdMiddleware
from app.infrastructure.database.session import dispose_engine
from app.modules.system.router import router as system_router
from app.schemas.chat import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_runtime()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="基于 FastAPI + FAISS + 大模型 API 的数学 RAG 问答原型系统",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="兼容健康检查")
    def health() -> HealthResponse:
        return HealthResponse(app_name=settings.APP_NAME)

    @app.get("/", tags=["system"], include_in_schema=False, response_model=None)
    def root() -> JSONResponse | HTMLResponse:
        frontend_dir = Path(__file__).resolve().parent / "frontend"
        index_file = frontend_dir / "index.html"
        if not index_file.exists():
            return JSONResponse(
                {
                    "message": f"{settings.APP_NAME} 已启动。",
                    "docs": "/docs",
                    "chat_api": "/api/chat",
                    "health": "/health",
                }
            )
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(knowledge_router)

    frontend_dir = Path(__file__).resolve().parent / "frontend"
    if frontend_dir.exists():
        # API 路由先注册，根路径静态挂载不会吞掉 /api/* 和 /health/*。
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    return app


app = create_app()
