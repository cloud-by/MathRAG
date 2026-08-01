from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.knowledge import router as knowledge_router
from app.core.config import settings
from app.core.exception_handlers import install_exception_handlers
from app.core.frontend import mount_vue_frontend
from app.core.middleware import RequestIdMiddleware
from app.infrastructure.database.session import dispose_engine
from app.infrastructure.embedding.provider import dispose_embedding_provider
from app.modules.auth.router import router as auth_router
from app.modules.conversations.router import router as conversations_router
from app.modules.ingestion.router import router as ingestion_router
from app.modules.knowledge.router import router as knowledge_management_router
from app.modules.rag.router import router as rag_router
from app.modules.system.router import router as system_router
from app.modules.users.router import router as users_router
from app.schemas.chat import HealthResponse
from app.services.rag_pipeline import reset_rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    application_error: BaseException | None = None
    try:
        settings.validate_runtime()
        yield
    except BaseException as exc:
        application_error = exc
        raise
    finally:
        reset_rag_pipeline()
        cleanup_error: BaseException | None = None
        try:
            await dispose_embedding_provider()
        except BaseException as exc:
            cleanup_error = exc
        try:
            await dispose_engine()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if application_error is None and cleanup_error is not None:
            raise cleanup_error


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="基于 FastAPI + PostgreSQL/pgvector + 大模型 API 的数学 RAG 问答原型系统",
        lifespan=lifespan,
    )
    install_exception_handlers(app)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.ALLOWED_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="兼容健康检查")
    def health() -> HealthResponse:
        return HealthResponse(app_name=settings.APP_NAME)

    app.include_router(system_router)
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(conversations_router)
    app.include_router(rag_router)
    app.include_router(knowledge_management_router)
    app.include_router(ingestion_router)
    app.include_router(knowledge_router)
    mount_vue_frontend(app, settings.FRONTEND_DIST_DIR)
    return app


app = create_app()
