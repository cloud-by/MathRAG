"""通过统一 ingestion service 导入公开网页数学知识。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from uuid import UUID

from app.modules.ingestion.factory import (
    build_ingestion_service,
    dispose_ingestion_resources,
    resolve_active_admin,
)
from app.modules.ingestion.service import IngestionService
from app.services.math_knowledge_importer import SOURCE_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="抓取公开数学来源并通过统一摄取任务写入 PostgreSQL/pgvector。"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["proofwiki", "wikibooks", "wikipedia"],
        choices=sorted(SOURCE_REGISTRY),
    )
    parser.add_argument("--keywords", nargs="+", required=True)
    parser.add_argument("--limit-per-source", type=int, default=3)
    parser.add_argument("--category", default=None)
    parser.add_argument("--max-chunk-chars", type=int, default=6000)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument(
        "--requested-by",
        required=True,
        help="发起导入的 active admin 用户名。",
    )
    return parser


async def run_import(
    args: argparse.Namespace,
    *,
    service: IngestionService | None = None,
    resolve_admin: Callable[[str], Awaitable[UUID]] = resolve_active_admin,
) -> int:
    requested_by = await resolve_admin(args.requested_by)
    active_service = service or build_ingestion_service()
    job = await active_service.accept_web(
        requested_by=requested_by,
        sources=list(args.sources),
        keywords=list(args.keywords),
        limit_per_source=args.limit_per_source,
        category=args.category,
        delay_seconds=args.delay_seconds,
        max_chunk_chars=args.max_chunk_chars,
    )
    await active_service.run_pending(job.id)
    completed_job = await active_service.get_job(job.id)
    if completed_job.status != "completed":
        raise RuntimeError("网页导入任务未完成")
    print("Completed jobs: 1")
    return 1


async def _async_main(args: argparse.Namespace) -> int:
    business_error: BaseException | None = None
    try:
        await run_import(args)
        return 0
    except BaseException as exc:
        business_error = exc
        raise
    finally:
        try:
            await dispose_ingestion_resources()
        except BaseException:
            if business_error is None:
                raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except Exception as exc:
        print(f"INGESTION_IMPORT_FAILED: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
