"""知识向量重建 CLI 的退出码、清理与幂等测试。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Sequence
from uuid import UUID

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem
from app.modules.knowledge.reindex_service import ReindexSummary
from tests.integration.database_safety import require_test_database_url


MODEL = "integration-reindex-cli"


def summary() -> ReindexSummary:
    """构造 CLI 成功输出的稳定摘要。"""
    return ReindexSummary(2, 2, 1, 0, MODEL, 1024)


@pytest.mark.parametrize(
    ("failure", "expected_code", "expected_error"),
    [
        (
            EmbeddingInputError("sk-secret 正文"),
            2,
            "invalid_embedding_config",
        ),
        (
            EmbeddingUnavailableError("https://private.example/v1 sk-secret 正文"),
            3,
            "embedding_unavailable",
        ),
        (
            EmbeddingResponseError("raw-response-body sk-secret 正文"),
            3,
            "embedding_unavailable",
        ),
        (RuntimeError("数据库密码 sk-secret 正文"), 1, "database_error"),
    ],
)
def test_main_maps_failures_to_single_safe_json_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
    expected_code: int,
    expected_error: str,
) -> None:
    """四类失败使用稳定退出码，stderr 不回显异常消息或栈。"""
    import scripts.reindex_knowledge as command

    async def fail() -> ReindexSummary:
        raise failure

    monkeypatch.setattr(command, "run_reindex", fail)

    assert command.main() == expected_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    payload = json.loads(captured.err)
    assert payload == {"detail": type(failure).__name__, "error": expected_error}
    assert "sk-secret" not in captured.err
    assert "private.example" not in captured.err
    assert "raw-response-body" not in captured.err
    assert "正文" not in captured.err


def test_main_outputs_exactly_one_json_summary_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """成功时 stdout 只输出一行 JSON，stderr 为空。"""
    import scripts.reindex_knowledge as command

    async def succeed() -> ReindexSummary:
        return summary()

    monkeypatch.setattr(command, "run_reindex", succeed)

    assert command.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {
        "dimensions": 1024,
        "embedding_model": MODEL,
        "failed": 0,
        "ready": 2,
        "selected": 2,
        "skipped": 1,
    }


def test_run_reindex_closes_injected_provider_and_engine_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """注入依赖也必须在成功后依次关闭 Provider 与全局数据库引擎。"""
    import scripts.reindex_knowledge as command

    events: list[str] = []

    class Provider:
        model = MODEL
        dimensions = 1024

        async def aclose(self) -> None:
            events.append("provider:close")

    class Service:
        def __init__(self, *args: object, **kwargs: object) -> None:
            events.append("service:init")

        async def reindex(self) -> ReindexSummary:
            events.append("service:reindex")
            return summary()

    async def dispose_database() -> None:
        events.append("database:dispose")

    monkeypatch.setattr(command, "KnowledgeReindexService", Service)
    monkeypatch.setattr(command, "dispose_engine", dispose_database)

    result = asyncio.run(
        command.run_reindex(
            session_factory=object(),  # type: ignore[arg-type]
            provider=Provider(),  # type: ignore[arg-type]
            batch_size=2,
        )
    )

    assert result == summary()
    assert events == [
        "service:init",
        "service:reindex",
        "provider:close",
        "database:dispose",
    ]


def test_run_reindex_recreates_and_disposes_global_provider_on_each_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一进程内每次全局运行都必须使用新 Provider 并清空缓存。"""
    import app.infrastructure.embedding.provider as provider_module
    import scripts.reindex_knowledge as command

    instances: list[Provider] = []
    disposed_engines = 0

    class Provider:
        model = MODEL
        dimensions = 1024

        def __init__(self) -> None:
            self.close_calls = 0
            instances.append(self)

        async def aclose(self) -> None:
            self.close_calls += 1

    class Service:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def reindex(self) -> ReindexSummary:
            return summary()

    async def dispose_database() -> None:
        nonlocal disposed_engines
        disposed_engines += 1

    monkeypatch.setattr(provider_module, "_embedding_provider", None)
    monkeypatch.setattr(provider_module, "OpenAIEmbeddingProvider", Provider)
    monkeypatch.setattr(
        command,
        "get_embedding_provider",
        provider_module.get_embedding_provider,
    )
    monkeypatch.setattr(command, "KnowledgeReindexService", Service)
    monkeypatch.setattr(command, "dispose_engine", dispose_database)

    for _ in range(2):
        assert asyncio.run(
            command.run_reindex(
                session_factory=object(),  # type: ignore[arg-type]
                batch_size=2,
            )
        ) == summary()
        assert provider_module._embedding_provider is None

    assert len(instances) == 2
    assert [instance.close_calls for instance in instances] == [1, 1]
    assert disposed_engines == 2


def test_cleanup_failures_do_not_override_provider_business_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """两项清理都执行，其故障不得覆盖已有 Provider 退出分类或泄密。"""
    import scripts.reindex_knowledge as command

    events: list[str] = []

    class Provider:
        model = MODEL
        dimensions = 1024

        async def aclose(self) -> None:
            events.append("provider:close")
            raise RuntimeError("cleanup sk-cleanup-secret")

    class Service:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def reindex(self) -> ReindexSummary:
            raise EmbeddingUnavailableError("provider sk-provider-secret 正文")

    async def dispose_database() -> None:
        events.append("database:dispose")
        raise RuntimeError("database sk-database-secret")

    provider = Provider()

    async def dispose_provider() -> None:
        events.append("provider:dispose")
        await provider.aclose()

    monkeypatch.setattr(command, "get_session_factory", lambda: object())
    monkeypatch.setattr(command, "get_embedding_provider", lambda: provider)
    monkeypatch.setattr(
        command,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(command, "KnowledgeReindexService", Service)
    monkeypatch.setattr(command, "dispose_engine", dispose_database)

    assert command.main() == 3
    captured = capsys.readouterr()
    assert events == [
        "provider:dispose",
        "provider:close",
        "database:dispose",
    ]
    assert json.loads(captured.err) == {
        "detail": "EmbeddingUnavailableError",
        "error": "embedding_unavailable",
    }
    assert "secret" not in captured.err
    assert "正文" not in captured.err


def test_cleanup_failure_without_business_error_maps_to_database_exit_and_still_disposes_engine(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """清理本身失败时仍运行后续清理，并以通用 exit 1 脱敏报告。"""
    import scripts.reindex_knowledge as command

    events: list[str] = []

    class Provider:
        model = MODEL
        dimensions = 1024

        async def aclose(self) -> None:
            events.append("provider:close")
            raise RuntimeError("cleanup sk-cleanup-secret")

    class Service:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def reindex(self) -> ReindexSummary:
            return summary()

    async def dispose_database() -> None:
        events.append("database:dispose")

    provider = Provider()

    async def dispose_provider() -> None:
        events.append("provider:dispose")
        await provider.aclose()

    monkeypatch.setattr(command, "get_session_factory", lambda: object())
    monkeypatch.setattr(command, "get_embedding_provider", lambda: provider)
    monkeypatch.setattr(
        command,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(command, "KnowledgeReindexService", Service)
    monkeypatch.setattr(command, "dispose_engine", dispose_database)

    assert command.main() == 1
    captured = capsys.readouterr()
    assert events == [
        "provider:dispose",
        "provider:close",
        "database:dispose",
    ]
    assert json.loads(captured.err) == {
        "detail": "RuntimeError",
        "error": "database_error",
    }
    assert "sk-cleanup-secret" not in captured.err


def vector(axis: int) -> list[float]:
    """构造真实 pgvector 可写入的固定维度单位向量。"""
    values = [0.0] * 1024
    values[axis] = 1.0
    return values


class DatabaseProvider:
    """用于真实双跑的无网络 Provider。"""

    model = MODEL
    dimensions = 1024

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.close_calls = 0

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        return [vector(index) for index in range(len(batch))]

    async def aclose(self) -> None:
        self.close_calls += 1


def make_item(number: int) -> KnowledgeItem:
    """构造 CLI 双跑使用的真实待重建条目。"""
    item = KnowledgeItem(
        id=UUID(f"20000000-0000-0000-0000-{number:012d}"),
        legacy_id=f"reindex-cli-{number}",
        category="reindex-cli",
        title=f"CLI 条目 {number}",
        keywords=["CLI"],
        content="正文",
        example="示例",
        steps=["步骤"],
        difficulty="easy",
        status="indexing",
    )
    item.chunks.append(
        KnowledgeChunk(
            id=UUID(f"30000000-0000-0000-0000-{number:012d}"),
            chunk_index=0,
            retrieval_text=f"CLI 检索文本 {number}",
            answer_context="回答上下文",
            metadata_={},
            status="pending",
        )
    )
    return item


async def cleanup(session: AsyncSession) -> None:
    """清空专用测试库知识表。"""
    await session.execute(delete(KnowledgeChunk))
    await session.execute(delete(KnowledgeItem))


async def assert_restored(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """确认数据库回到当前迁移 head 且知识表为空。"""
    async with session_factory() as session:
        assert (
            await session.scalar(text("SELECT version_num FROM alembic_version")),
            await session.scalar(select(func.count()).select_from(KnowledgeItem)),
            await session.scalar(select(func.count()).select_from(KnowledgeChunk)),
        ) == ("0005_create_documents_ingestion_jobs", 0, 0)


async def exercise_cli_double_run(database_url: str) -> None:
    """通过可注入 run_reindex 验证真实数据库双跑与资源释放。"""
    import scripts.reindex_knowledge as command

    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    try:
        async with session_factory() as session:
            async with session.begin():
                await cleanup(session)
                session.add_all([make_item(2), make_item(1)])

        first_provider = DatabaseProvider()
        first = await command.run_reindex(
            session_factory=session_factory,
            provider=first_provider,
            batch_size=2,
        )
        assert first == ReindexSummary(2, 2, 0, 0, MODEL, 1024)
        assert first_provider.calls == [["CLI 检索文本 1", "CLI 检索文本 2"]]
        assert first_provider.close_calls == 1

        second_provider = DatabaseProvider()
        second = await command.run_reindex(
            session_factory=session_factory,
            provider=second_provider,
            batch_size=2,
        )
        assert second == ReindexSummary(0, 0, 2, 0, MODEL, 1024)
        assert second_provider.calls == []
        assert second_provider.close_calls == 1
    finally:
        original_exception_pending = sys.exc_info()[0] is not None
        cleanup_failed = False
        try:
            async with session_factory() as session:
                async with session.begin():
                    await cleanup(session)
            await assert_restored(session_factory)
        except BaseException:
            cleanup_failed = True
            if not original_exception_pending:
                raise
        finally:
            try:
                await engine.dispose()
            except BaseException:
                if not original_exception_pending and not cleanup_failed:
                    raise


def test_run_reindex_is_idempotent_on_real_postgres_and_closes_each_provider() -> None:
    """身份守卫后在专用 PG 连跑两次，第二次不得调用 Provider。"""
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))

    asyncio.run(exercise_cli_double_run(database_url))
