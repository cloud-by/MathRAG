"""文档管理和导入任务 API。"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Query,
    UploadFile,
    status,
)

from app.core.config import settings
from app.infrastructure.database.session import get_session_factory
from app.modules.auth.dependencies import require_admin, require_admin_csrf
from app.modules.auth.service import AuthenticatedPrincipal
from app.modules.ingestion.schemas import (
    DocumentAccepted,
    DocumentPage,
    IngestionJobPage,
    IngestionJobRead,
)
from app.modules.ingestion.service import IngestionService
from app.modules.ingestion.storage import UploadStorage


router = APIRouter(prefix="/api/v1")


def get_ingestion_service() -> IngestionService:
    """为请求组装摄取服务；Embedding Provider 在后台执行时才获取。"""
    storage = UploadStorage(
        root=settings.UPLOAD_DIR,
        max_bytes=settings.MAX_UPLOAD_BYTES,
        max_pages=settings.MAX_PDF_PAGES,
    )
    return IngestionService(
        get_session_factory(),
        storage,
        upload_root=settings.UPLOAD_DIR,
        max_pdf_pages=settings.MAX_PDF_PAGES,
        max_ingestion_text_chars=settings.MAX_INGESTION_TEXT_CHARS,
        embedding_batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


@router.post(
    "/documents",
    tags=["documents"],
    response_model=DocumentAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentAccepted:
    accepted = await service.accept_pdf(
        file,
        owner_id=principal.user_id,
        category=category,
    )
    background_tasks.add_task(service.run_pending, accepted.job.id)
    return accepted


@router.get(
    "/documents",
    tags=["documents"],
    response_model=DocumentPage,
)
async def list_documents(
    status_filter: Literal[
        "pending", "processing", "ready", "failed", "archived"
    ]
    | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentPage:
    return await service.list_documents(
        status=status_filter,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/ingestion-jobs",
    tags=["ingestion"],
    response_model=IngestionJobPage,
)
async def list_ingestion_jobs(
    status_filter: Literal[
        "pending", "running", "completed", "failed", "cancelled"
    ]
    | None = Query(default=None, alias="status"),
    job_type: Literal["text", "pdf", "web", "reindex"] | None = Query(
        default=None
    ),
    document_id: UUID | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobPage:
    return await service.list_jobs(
        status=status_filter,
        job_type=job_type,
        document_id=document_id,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/ingestion-jobs/{job_id}",
    tags=["ingestion"],
    response_model=IngestionJobRead,
)
async def get_ingestion_job(
    job_id: UUID,
    _principal: AuthenticatedPrincipal = Depends(require_admin),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobRead:
    return await service.get_job(job_id)


@router.post(
    "/ingestion-jobs/{job_id}/cancel",
    tags=["ingestion"],
    response_model=IngestionJobRead,
)
async def cancel_ingestion_job(
    job_id: UUID,
    _principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobRead:
    return await service.cancel(job_id)


@router.post(
    "/ingestion-jobs/{job_id}/retry",
    tags=["ingestion"],
    response_model=IngestionJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ingestion_job(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    _principal: AuthenticatedPrincipal = Depends(require_admin_csrf),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobRead:
    snapshot = await service.claim_retry(job_id)
    job = await service.get_job(job_id)
    background_tasks.add_task(service.resume_retry, snapshot)
    return job
