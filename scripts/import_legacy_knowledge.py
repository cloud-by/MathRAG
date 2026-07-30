"""将既有 UTF-8 JSONL 知识离线导入 PostgreSQL。"""

from __future__ import annotations

import asyncio
import json
import sys

from app.core.config import settings
from app.infrastructure.database.session import dispose_engine, get_session_factory
from app.modules.knowledge.errors import LegacyKnowledgeConflictError, LegacyKnowledgeInputError
from app.modules.knowledge.legacy_loader import load_legacy_bundles
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.schemas import LegacyImportSummary
from app.modules.knowledge.service import LegacyKnowledgeImportService


async def run_import() -> LegacyImportSummary:
    """读取现有知识文件并在单个数据库会话中完成可重入导入。"""
    try:
        bundles = load_legacy_bundles(settings.RAW_KB_PATH, settings.PROCESSED_KB_PATH)
        async with get_session_factory()() as session:
            repository = KnowledgeRepository(session)
            return await LegacyKnowledgeImportService(session, repository).import_bundles(bundles)
    finally:
        await dispose_engine()


def _write_error(error: str, detail: str) -> None:
    """向 stderr 写入严格单行 JSON 错误，不混入日志或异常栈。"""
    sys.stderr.write(json.dumps({"error": error, "detail": detail}, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    """运行导入并返回可供自动化调用的稳定退出码。"""
    try:
        summary = asyncio.run(run_import())
    except LegacyKnowledgeInputError as exc:
        _write_error("invalid_input", str(exc))
        return 2
    except LegacyKnowledgeConflictError as exc:
        _write_error("conflict", str(exc))
        return 3
    except Exception as exc:
        _write_error("database_error", type(exc).__name__)
        return 1

    sys.stdout.write(
        json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
