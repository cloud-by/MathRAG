"""通过统一 ingestion service 导入本地 PDF。"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from uuid import UUID

from app.modules.ingestion.factory import (
    build_ingestion_service,
    dispose_ingestion_resources,
    resolve_active_admin,
)
from app.modules.ingestion.service import IngestionService
from app.services.pdf_knowledge_importer import DEFAULT_DATA_LAKE_DIR, iter_pdf_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本地 PDF 通过统一摄取任务导入 PostgreSQL/pgvector。"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_LAKE_DIR,
        help="PDF 文件目录，默认 data/data_lake。",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不扫描子目录。",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="本次最多创建的 PDF 导入任务数。",
    )
    parser.add_argument("--category", default=None, help="可选知识分类提示。")
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
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("max_chunks 必须大于 0")
    owner_id = await resolve_admin(args.requested_by)
    active_service = service or build_ingestion_service()
    paths = iter_pdf_paths(
        args.data_dir,
        recursive=not args.no_recursive,
    )
    if args.max_chunks is not None:
        paths = paths[: args.max_chunks]

    completed = 0
    for path in paths:
        accepted = await active_service.accept_local_pdf(
            path,
            owner_id=owner_id,
            category=args.category,
        )
        await active_service.run_pending(accepted.job.id)
        job = await active_service.get_job(accepted.job.id)
        if job.status != "completed":
            raise RuntimeError("PDF 导入任务未完成")
        completed += 1
    print(f"Completed jobs: {completed}")
    return completed


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
