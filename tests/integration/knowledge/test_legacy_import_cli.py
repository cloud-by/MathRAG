"""旧知识离线导入 CLI 的集成与退出码测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import settings
from app.modules.knowledge.errors import LegacyKnowledgeConflictError, LegacyKnowledgeInputError
from app.modules.knowledge.schemas import LegacyImportSummary


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_SHA256 = "2593f45081b11ab4ae280d1a7fb107791b3099c364f3813f215a73fa7369d062"
PROCESSED_SHA256 = "a0334a626d7e54ce04a447861af1616da26ad8b012d81f6720aa1d404539e5aa"


def _summary() -> LegacyImportSummary:
    """构造 CLI 成功输出的最小真实摘要。"""
    return LegacyImportSummary(
        input_items=26,
        input_chunks=26,
        created=26,
        skipped=0,
        conflicts=0,
        failed=0,
        database_items=26,
        database_chunks=26,
        input_sha256="0" * 64,
        database_sha256="0" * 64,
    )


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_error"),
    [
        (LegacyKnowledgeInputError("raw.jsonl:1: invalid"), 2, "invalid_input"),
        (LegacyKnowledgeConflictError("conflict"), 3, "conflict"),
        (RuntimeError("假密码=top-secret；知识正文"), 1, "database_error"),
    ],
)
def test_main_outputs_single_safe_json_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected_code: int,
    expected_error: str,
) -> None:
    """失败路径只向 stderr 输出单行安全 JSON，且通用异常不泄露消息。"""
    import scripts.import_legacy_knowledge as command

    async def fail() -> LegacyImportSummary:
        raise failure

    monkeypatch.setattr(command, "run_import", fail)

    assert command.main() == expected_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    payload = json.loads(captured.err)
    assert payload["error"] == expected_error
    if expected_error == "database_error":
        assert payload["detail"] == "RuntimeError"
        assert "top-secret" not in captured.err
        assert "知识正文" not in captured.err


def test_main_outputs_one_json_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """成功时 stdout 仅含一行确定性 JSON，stderr 为空。"""
    import scripts.import_legacy_knowledge as command

    async def succeed() -> LegacyImportSummary:
        return _summary()

    monkeypatch.setattr(command, "run_import", succeed)

    assert command.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == _summary().model_dump(mode="json")


def test_legacy_import_cli_is_idempotent_and_lossless() -> None:
    """在专用测试库中连续运行两次 CLI，验证幂等、状态与字段级无损。"""
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("未配置 TEST_DATABASE_URL")

    from app.infrastructure.database.session import dispose_engine
    from app.modules.knowledge.legacy_loader import load_legacy_bundles
    from app.modules.knowledge.repository import KnowledgeRepository
    from app.modules.knowledge.service import bundle_from_model

    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "DATABASE_URL": test_database_url},
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    try:
        async def clear() -> None:
            engine = create_async_engine(test_database_url)
            try:
                async with engine.begin() as connection:
                    await connection.execute(text("TRUNCATE knowledge_chunks, knowledge_items RESTART IDENTITY CASCADE"))
                    result = await connection.execute(text("SELECT (SELECT count(*) FROM knowledge_items), (SELECT count(*) FROM knowledge_chunks)"))
                    assert result.one() == (0, 0)
            finally:
                await engine.dispose()

        asyncio.run(clear())
        before_raw = hashlib.sha256(settings.RAW_KB_PATH.read_bytes()).hexdigest()
        before_processed = hashlib.sha256(settings.PROCESSED_KB_PATH.read_bytes()).hexdigest()
        environment = {**os.environ, "DATABASE_URL": test_database_url}
        first = subprocess.run([sys.executable, "-m", "scripts.import_legacy_knowledge"], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, encoding="utf-8")
        second = subprocess.run([sys.executable, "-m", "scripts.import_legacy_knowledge"], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, encoding="utf-8")

        assert (first.returncode, second.returncode) == (0, 0)
        assert first.stderr == second.stderr == ""
        assert first.stdout.count("\n") == second.stdout.count("\n") == 1
        first_summary, second_summary = json.loads(first.stdout), json.loads(second.stdout)
        assert (first_summary["created"], first_summary["skipped"], first_summary["conflicts"], first_summary["failed"]) == (26, 0, 0, 0)
        assert (second_summary["created"], second_summary["skipped"], second_summary["conflicts"], second_summary["failed"]) == (0, 26, 0, 0)
        assert all(summary["input_items"] == summary["input_chunks"] == 26 for summary in (first_summary, second_summary))
        assert first_summary["input_sha256"] == first_summary["database_sha256"] == second_summary["input_sha256"] == second_summary["database_sha256"]
        assert before_raw == hashlib.sha256(settings.RAW_KB_PATH.read_bytes()).hexdigest()
        assert before_processed == hashlib.sha256(settings.PROCESSED_KB_PATH.read_bytes()).hexdigest()
        assert before_raw == RAW_SHA256
        assert before_processed == PROCESSED_SHA256

        async def verify() -> None:
            engine = create_async_engine(test_database_url)
            try:
                async with engine.connect() as connection:
                    counts = await connection.execute(text("SELECT (SELECT count(*) FROM knowledge_items), (SELECT count(*) FROM knowledge_chunks), (SELECT count(DISTINCT legacy_id) FROM knowledge_items), (SELECT count(*) FROM knowledge_items WHERE legacy_id IS NOT NULL), (SELECT count(*) FROM knowledge_items WHERE status = 'indexing'), (SELECT count(*) FROM knowledge_chunks WHERE status = 'pending'), (SELECT count(*) FROM knowledge_chunks WHERE embedding IS NULL)"))
                    assert counts.one() == (26, 26, 26, 26, 26, 26, 26)
                async with async_sessionmaker(engine, expire_on_commit=False)() as session:
                    persisted = await KnowledgeRepository(session).list_legacy_items_ordered()
                    expected = load_legacy_bundles(settings.RAW_KB_PATH, settings.PROCESSED_KB_PATH)
                    assert [bundle_from_model(item).persistent_payload() for item in persisted] == [bundle.persistent_payload() for bundle in expected]
            finally:
                await engine.dispose()

        asyncio.run(verify())
    finally:
        async def cleanup() -> None:
            engine = create_async_engine(test_database_url)
            try:
                async with engine.begin() as connection:
                    await connection.execute(text("TRUNCATE knowledge_chunks, knowledge_items RESTART IDENTITY CASCADE"))
                    result = await connection.execute(text("SELECT (SELECT count(*) FROM knowledge_items), (SELECT count(*) FROM knowledge_chunks)"))
                    assert result.one() == (0, 0)
            finally:
                await engine.dispose()

        asyncio.run(cleanup())
        asyncio.run(dispose_engine())
