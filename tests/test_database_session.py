from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def test_engine_is_created_lazily_and_reused(monkeypatch) -> None:
    from app.infrastructure.database import session

    created: list[tuple[str, dict]] = []
    fake_engine = object()
    fake_settings = SimpleNamespace(
        DB_POOL_SIZE=5,
        DB_MAX_OVERFLOW=5,
        DB_POOL_TIMEOUT=30,
        require_database_url=lambda: "postgresql+asyncpg://u:p@db/test",
    )
    monkeypatch.setattr(session, "settings", fake_settings)
    monkeypatch.setattr(
        session,
        "create_async_engine",
        lambda url, **kwargs: created.append((url, kwargs)) or fake_engine,
    )
    session.reset_database_state_for_tests()

    assert created == []
    assert session.get_engine() is fake_engine
    assert session.get_engine() is fake_engine
    assert len(created) == 1
    assert created[0][1]["pool_pre_ping"] is True


def test_session_dependency_closes_and_rolls_back(monkeypatch) -> None:
    from app.infrastructure.database import session

    class FakeSession:
        rolled_back = False

        async def rollback(self) -> None:
            self.rolled_back = True

    class FakeContext:
        def __init__(self) -> None:
            self.value = FakeSession()
            self.closed = False

        async def __aenter__(self) -> FakeSession:
            return self.value

        async def __aexit__(self, *args) -> None:
            self.closed = True

    async def exercise_dependency() -> None:
        context = FakeContext()
        monkeypatch.setattr(session, "get_session_factory", lambda: lambda: context)
        dependency = session.get_db_session()
        yielded = await anext(dependency)

        assert yielded is context.value
        with pytest.raises(RuntimeError, match="boom"):
            await dependency.athrow(RuntimeError("boom"))
        assert context.value.rolled_back is True
        assert context.closed is True

    asyncio.run(exercise_dependency())
