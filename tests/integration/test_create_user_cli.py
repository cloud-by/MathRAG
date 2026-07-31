"""安全创建用户 CLI 集成测试。"""

from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.modules.auth.security import verify_password
from app.modules.users.models import User
from scripts.create_user import main
from tests.integration.database_safety import require_test_database_url


def test_cli_reads_password_twice_and_outputs_only_public_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")
    database_url = require_test_database_url(database_url, os.getenv("DATABASE_URL"))
    # main() 使用 asyncio.run；NullPool 避免测试在多个短事件循环间复用连接。
    engine = create_async_engine(database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    password = "cli-private-password"
    prompts: list[str] = []

    def fake_getpass(prompt: str) -> str:
        prompts.append(prompt)
        return password

    monkeypatch.setattr("scripts.create_user.getpass.getpass", fake_getpass)
    try:
        async def cleanup() -> None:
            async with session_factory() as session:
                async with session.begin():
                    await session.execute(delete(User).where(User.username == "cli-admin"))

        asyncio.run(cleanup())
        exit_code = main(
            ["--username", "CLI-Admin", "--role", "admin", "--email", "CLI@Example.Local"],
            session_factory=session_factory,
        )

        async def fetch_user() -> User | None:
            async with session_factory() as session:
                return await session.scalar(select(User).where(User.username == "cli-admin"))

        user = asyncio.run(fetch_user())
        output = capsys.readouterr().out
        assert exit_code == 0
        assert len(prompts) == 2
        assert user is not None
        assert asyncio.run(verify_password(password, user.password_hash)) is True
        assert str(user.id) in output
        assert "cli-admin" in output
        assert "admin" in output
        assert "USER_CREATED" in output
        assert password not in output
        assert user.password_hash not in output
        assert database_url not in output
    finally:
        asyncio.run(engine.dispose())


def test_cli_rejects_password_argument_without_reading_it() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--username", "cli-user", "--password", "not-allowed"])

    assert exc_info.value.code == 2


def test_cli_password_mismatch_returns_input_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    passwords = iter(["first-password", "second-password"])
    monkeypatch.setattr(
        "scripts.create_user.getpass.getpass",
        lambda _prompt: next(passwords),
    )

    exit_code = main(["--username", "cli-user"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "USER_INPUT_INVALID" in captured.err
    assert "first-password" not in captured.err
    assert "second-password" not in captured.err


def test_cli_database_failure_returns_stable_generic_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "scripts.create_user.getpass.getpass",
        lambda _prompt: "valid-password",
    )

    def broken_session_factory() -> None:
        raise RuntimeError("postgresql://private-database")

    exit_code = main(
        ["--username", "cli-user"],
        session_factory=broken_session_factory,  # type: ignore[arg-type]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err.strip() == "USER_CREATE_FAILED"
    assert "postgresql" not in captured.err
