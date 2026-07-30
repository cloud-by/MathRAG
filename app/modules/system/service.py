from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.core.errors import ConfigurationError
from app.infrastructure.database.session import get_engine


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    checks: dict[str, str]

    def as_payload(self) -> dict[str, object]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": self.checks,
        }


class ReadinessService:
    def __init__(self, engine_provider: Callable[[], AsyncEngine] = get_engine) -> None:
        self.engine_provider = engine_provider

    async def check(self) -> ReadinessResult:
        checks = {"config": "ok", "database": "unknown", "pgvector": "unknown"}
        try:
            settings.require_database_url()
        except ConfigurationError:
            checks["config"] = "invalid"
            return ReadinessResult(False, checks)

        try:
            async with self.engine_provider().connect() as connection:
                version = (
                    await connection.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    )
                ).scalar_one_or_none()
        except (SQLAlchemyError, OSError):
            checks["database"] = "unavailable"
            return ReadinessResult(False, checks)

        checks["database"] = "ok"
        if version is None:
            checks["pgvector"] = "missing"
            return ReadinessResult(False, checks)
        checks["pgvector"] = str(version)
        return ReadinessResult(True, checks)
