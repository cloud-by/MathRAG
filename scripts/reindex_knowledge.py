"""将 PostgreSQL 知识分块可重入地重建为 pgvector 向量。"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.infrastructure.database.session import dispose_engine, get_session_factory
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    get_embedding_provider,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from app.modules.knowledge.reindex_service import (
    KnowledgeReindexService,
    ReindexSummary,
)


async def run_reindex(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    provider: EmbeddingProvider | None = None,
    batch_size: int | None = None,
) -> ReindexSummary:
    """执行一次重建，并始终尝试释放 Provider 和数据库引擎。"""
    active_provider = provider
    business_error: BaseException | None = None
    try:
        if active_provider is None:
            active_provider = get_embedding_provider()
        active_session_factory = (
            session_factory if session_factory is not None else get_session_factory()
        )
        service = KnowledgeReindexService(
            active_session_factory,
            active_provider,
            batch_size=(
                settings.EMBEDDING_BATCH_SIZE
                if batch_size is None
                else batch_size
            ),
        )
        return await service.reindex()
    except BaseException as exc:
        business_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        if active_provider is not None:
            try:
                await active_provider.aclose()
            except BaseException as exc:
                cleanup_error = exc
        try:
            await dispose_engine()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if business_error is None and cleanup_error is not None:
            raise cleanup_error


def write_error(error: str, detail: str) -> None:
    """向 stderr 写入不包含异常消息或堆栈的单行 JSON。"""
    sys.stderr.write(
        json.dumps(
            {"detail": detail, "error": error},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    """运行重建并返回稳定退出码。"""
    try:
        summary = asyncio.run(run_reindex())
    except EmbeddingInputError as exc:
        write_error("invalid_embedding_config", type(exc).__name__)
        return 2
    except (EmbeddingUnavailableError, EmbeddingResponseError) as exc:
        write_error("embedding_unavailable", type(exc).__name__)
        return 3
    except Exception as exc:
        write_error("database_error", type(exc).__name__)
        return 1

    sys.stdout.write(
        json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
