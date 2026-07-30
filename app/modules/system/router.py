from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.modules.system.service import ReadinessService


router = APIRouter(tags=["system"])


def get_readiness_service() -> ReadinessService:
    return ReadinessService()


@router.get("/health/live", summary="进程存活检查")
async def live() -> dict[str, str]:
    return {"status": "ok", "app_name": settings.APP_NAME}


@router.get("/health/ready", summary="应用就绪检查", response_model=None)
async def ready(
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> dict[str, object] | JSONResponse:
    result = await service.check()
    payload = result.as_payload()
    if not result.ready:
        return JSONResponse(status_code=503, content=payload)
    return payload
