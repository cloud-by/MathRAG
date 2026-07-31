"""CLI 使用的 ingestion 运行时组装与管理员解析。"""

from __future__ import annotations

from uuid import UUID

from app.core.config import settings
from app.infrastructure.database.session import dispose_engine, get_session_factory
from app.infrastructure.embedding.provider import (
    dispose_embedding_provider,
    get_embedding_provider,
)
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import UploadStorage
from app.modules.users.repository import UserRepository


def build_ingestion_service() -> IngestionService:
    """组装 CLI 所需的数据库、存储、抽取器和共享 Provider。"""
    storage = UploadStorage(
        root=settings.UPLOAD_DIR,
        max_bytes=settings.MAX_UPLOAD_BYTES,
        max_pages=settings.MAX_PDF_PAGES,
    )
    return IngestionService(
        get_session_factory(),
        storage,
        embedding_provider=get_embedding_provider(),
        upload_root=settings.UPLOAD_DIR,
        max_pdf_pages=settings.MAX_PDF_PAGES,
        max_ingestion_text_chars=settings.MAX_INGESTION_TEXT_CHARS,
        embedding_batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


async def resolve_active_admin(username: str) -> UUID:
    """解析 active admin，普通或停用用户不能触发批量导入。"""
    normalized = str(username or "").strip().lower()
    if not normalized:
        raise ValueError("requested_by 必须是管理员用户名")
    async with get_session_factory()() as session:
        user = await UserRepository(session).get_by_username(normalized)
    if user is None or user.role != "admin" or user.status != "active":
        raise ValueError("requested_by 必须是 active admin")
    return user.id


async def dispose_ingestion_resources() -> None:
    """依次释放 Provider 和数据库资源，并保留第一个清理异常。"""
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
    if cleanup_error is not None:
        raise cleanup_error
