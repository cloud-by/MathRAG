from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def test_lifespan_preserves_app_error_while_attempting_both_cleanups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    events: list[str] = []

    async def dispose_provider() -> None:
        events.append("embedding")
        raise RuntimeError("embedding cleanup failed")

    async def dispose_database() -> None:
        events.append("database")
        raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(validate_runtime=lambda: events.append("validate")),
    )
    monkeypatch.setattr(main, "dispose_embedding_provider", dispose_provider)
    monkeypatch.setattr(main, "dispose_engine", dispose_database)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="application failed"):
            async with main.lifespan(main.app):
                events.append("body")
                raise RuntimeError("application failed")

    asyncio.run(exercise())

    assert events == ["validate", "body", "embedding", "database"]


def test_lifespan_rebuilds_rag_dependencies_and_preserves_app_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main
    from app.services import rag_pipeline as rag_pipeline_module

    events: list[str] = []
    built_searches: list[object] = []
    provider_calls = 0
    database_calls = 0

    def build_search() -> object:
        search = object()
        built_searches.append(search)
        events.append("build")
        return search

    def reset_pipeline() -> None:
        events.append("reset")
        rag_pipeline_module._rag_pipeline = None

    async def dispose_provider() -> None:
        nonlocal provider_calls
        provider_calls += 1
        events.append(f"embedding-{provider_calls}")
        if provider_calls == 2:
            raise RuntimeError("embedding cleanup failed")

    async def dispose_database() -> None:
        nonlocal database_calls
        database_calls += 1
        events.append(f"database-{database_calls}")
        if database_calls == 2:
            raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(rag_pipeline_module, "_rag_pipeline", None)
    monkeypatch.setattr(
        rag_pipeline_module,
        "build_knowledge_search_service",
        build_search,
    )
    monkeypatch.setattr(
        rag_pipeline_module,
        "get_query_planner",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(validate_runtime=lambda: events.append("validate")),
    )
    monkeypatch.setattr(main, "reset_rag_pipeline", reset_pipeline)
    monkeypatch.setattr(main, "dispose_embedding_provider", dispose_provider)
    monkeypatch.setattr(main, "dispose_engine", dispose_database)

    async def exercise() -> tuple[object, object]:
        async with main.lifespan(main.app):
            first = rag_pipeline_module.get_rag_pipeline()
            events.append("body-1")

        with pytest.raises(RuntimeError, match="application failed"):
            async with main.lifespan(main.app):
                second = rag_pipeline_module.get_rag_pipeline()
                events.append("body-2")
                raise RuntimeError("application failed")

        return first, second

    first, second = asyncio.run(exercise())

    assert first is not second
    assert len(built_searches) == 2
    assert events == [
        "validate",
        "build",
        "body-1",
        "reset",
        "embedding-1",
        "database-1",
        "validate",
        "build",
        "body-2",
        "reset",
        "embedding-2",
        "database-2",
    ]
