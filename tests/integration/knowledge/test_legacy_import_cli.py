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
from tests.integration.database_safety import require_test_database_url


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_CONTENT_SHA256 = "b87355849f828ae219ba4e03315436d65a1fce749db96740ae645a74c231e4b0"
PROCESSED_CONTENT_SHA256 = "f723c518f13c4a747b515785979d613139e9c6ec3e037a9210b0ba79c94032ad"
COLLECTION_SHA256 = "82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5"


def normalized_utf8_sha256(path: Path) -> str:
    """按 UTF-8/LF 规范化内容计算跨平台稳定摘要。"""
    content = path.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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


def test_main_preserves_input_exit_code_when_engine_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """业务输入错误传播期间，dispose 失败不得覆盖既有退出码。"""
    import scripts.import_legacy_knowledge as command

    def fail_loader(*args: object, **kwargs: object) -> list[object]:
        raise LegacyKnowledgeInputError("raw.jsonl:1: invalid")

    async def fail_dispose() -> None:
        raise RuntimeError("dispose failed")

    monkeypatch.setattr(command, "load_legacy_bundles", fail_loader)
    monkeypatch.setattr(command, "dispose_engine", fail_dispose)

    assert command.main() == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload == {"detail": "raw.jsonl:1: invalid", "error": "invalid_input"}


def test_run_import_propagates_engine_cleanup_failure_without_prior_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有业务异常时，dispose 失败必须正常传播，不能被静默吞掉。"""
    import scripts.import_legacy_knowledge as command

    class FakeSessionContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> bool:
            return False

    class FakeService:
        def __init__(self, *args: object) -> None:
            pass

        async def import_bundles(self, bundles: object) -> LegacyImportSummary:
            return _summary()

    def fake_factory() -> FakeSessionContext:
        return FakeSessionContext()

    async def fail_dispose() -> None:
        raise RuntimeError("dispose failed")

    monkeypatch.setattr(command, "load_legacy_bundles", lambda *args: [])
    monkeypatch.setattr(command, "get_session_factory", lambda: fake_factory)
    monkeypatch.setattr(command, "LegacyKnowledgeImportService", FakeService)
    monkeypatch.setattr(command, "dispose_engine", fail_dispose)

    with pytest.raises(RuntimeError, match="dispose failed"):
        asyncio.run(command.run_import())


def test_legacy_import_cli_is_idempotent_and_lossless() -> None:
    """在专用测试库中连续运行两次 CLI，验证幂等、状态与字段级无损。"""
    test_database_url = os.getenv("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.skip("未配置 TEST_DATABASE_URL")
    test_database_url = require_test_database_url(test_database_url, os.getenv("DATABASE_URL"))

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
        timeout=60,
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
        first = subprocess.run([sys.executable, "-m", "scripts.import_legacy_knowledge"], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
        second = subprocess.run([sys.executable, "-m", "scripts.import_legacy_knowledge"], cwd=PROJECT_ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)

        assert (first.returncode, second.returncode) == (0, 0)
        assert first.stderr == second.stderr == ""
        assert first.stdout.count("\n") == second.stdout.count("\n") == 1
        first_summary, second_summary = json.loads(first.stdout), json.loads(second.stdout)
        assert first_summary == {
            "input_items": 26, "input_chunks": 26, "created": 26, "skipped": 0, "conflicts": 0, "failed": 0,
            "database_items": 26, "database_chunks": 26, "input_sha256": COLLECTION_SHA256, "database_sha256": COLLECTION_SHA256,
        }
        assert second_summary == {
            "input_items": 26, "input_chunks": 26, "created": 0, "skipped": 26, "conflicts": 0, "failed": 0,
            "database_items": 26, "database_chunks": 26, "input_sha256": COLLECTION_SHA256, "database_sha256": COLLECTION_SHA256,
        }
        assert before_raw == hashlib.sha256(settings.RAW_KB_PATH.read_bytes()).hexdigest()
        assert before_processed == hashlib.sha256(settings.PROCESSED_KB_PATH.read_bytes()).hexdigest()
        assert normalized_utf8_sha256(settings.RAW_KB_PATH) == RAW_CONTENT_SHA256
        assert normalized_utf8_sha256(settings.PROCESSED_KB_PATH) == PROCESSED_CONTENT_SHA256

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
        original_exception_pending = sys.exc_info()[0] is not None
        async def cleanup() -> None:
            engine = create_async_engine(test_database_url)
            try:
                async with engine.begin() as connection:
                    await connection.execute(text("TRUNCATE knowledge_chunks, knowledge_items RESTART IDENTITY CASCADE"))
                    result = await connection.execute(text("SELECT (SELECT count(*) FROM knowledge_items), (SELECT count(*) FROM knowledge_chunks)"))
                    assert result.one() == (0, 0)
            finally:
                await engine.dispose()

        try:
            asyncio.run(cleanup())
        except BaseException:
            if not original_exception_pending:
                raise
        try:
            asyncio.run(dispose_engine())
        except BaseException:
            if not original_exception_pending:
                raise
