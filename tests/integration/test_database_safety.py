"""破坏性集成测试的数据库身份保护测试。"""

from __future__ import annotations

import pytest

from tests.integration.database_safety import require_test_database_url


TEST_DATABASE_URL = "postgresql+asyncpg://test_user:test_password@localhost:5432/mathrag_test"


def test_accepts_dedicated_mathrag_test_database() -> None:
    """专用测试库 URL 可用于破坏性集成测试。"""
    assert require_test_database_url(TEST_DATABASE_URL) == TEST_DATABASE_URL


def test_rejects_non_test_database_without_revealing_url_or_password() -> None:
    """主库误配必须在连接前被拒绝，且异常不得泄露凭据。"""
    main_url = "postgresql+asyncpg://app_user:fake-password@db.example:5432/mathrag"

    with pytest.raises(RuntimeError) as error:
        require_test_database_url(main_url)

    message = str(error.value)
    assert "mathrag_test" in message
    assert "mathrag" in message
    assert "fake-password" not in message
    assert main_url not in message


def test_rejects_when_database_and_test_urls_identify_same_database() -> None:
    """测试库与应用库同址时也不能执行破坏性测试。"""
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        require_test_database_url(TEST_DATABASE_URL, TEST_DATABASE_URL)


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql+asyncpg://other_user:other-password@localhost:5432/mathrag_test",
        "postgresql+asyncpg://test_user:other-password@localhost:5432/mathrag_test?sslmode=require",
        "postgresql://test_user:other-password@localhost/mathrag_test",
    ],
)
def test_rejects_same_physical_database_despite_connection_option_differences(
    database_url: str,
) -> None:
    """用户名、非路由 query 与 PostgreSQL 默认端口都不得绕过同库保护。"""
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        require_test_database_url(TEST_DATABASE_URL, database_url)


@pytest.mark.parametrize(
    "test_database_url",
    [
        "not a database url",
        "postgresql+asyncpg://test_user:fake-password@localhost:5432",
    ],
)
def test_rejects_invalid_or_database_less_url_without_revealing_secrets(
    test_database_url: str,
) -> None:
    """无效 URL 与缺少数据库名都不能绕过守卫。"""
    with pytest.raises(RuntimeError) as error:
        require_test_database_url(test_database_url)

    message = str(error.value)
    assert "fake-password" not in message
    assert test_database_url not in message


def test_rejects_invalid_test_port_without_revealing_url_or_password() -> None:
    """延迟解析的无效端口也必须在破坏操作前失败。"""
    invalid_url = "postgresql+asyncpg://test_user:fake-password@localhost:not-a-port/mathrag_test"

    with pytest.raises(RuntimeError) as error:
        require_test_database_url(invalid_url)

    message = str(error.value)
    assert "fake-password" not in message
    assert invalid_url not in message


def test_migration_guard_runs_before_any_alembic_downgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """主库误配在调用迁移破坏性辅助函数之前失败。"""
    from tests.integration import test_migrations

    destructive_calls: list[tuple[str, ...]] = []

    def destructive_probe(database_url: str, *args: str) -> None:
        del database_url
        destructive_calls.append(args)
        raise AssertionError("不应调用 Alembic")

    monkeypatch.setenv(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://test_user:fake-password@localhost:5432/mathrag",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(test_migrations, "run_alembic", destructive_probe)

    with pytest.raises(RuntimeError):
        test_migrations.test_migration_upgrade_downgrade_upgrade_round_trip()

    assert destructive_calls == []
