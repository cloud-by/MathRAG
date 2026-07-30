from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError


def test_db_pool_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="DB_POOL_SIZE"):
        Settings(DB_POOL_SIZE=0)


@pytest.mark.parametrize("app_env", ["staging", "production"])
def test_deployed_environments_require_database_url_at_runtime(app_env: str) -> None:
    settings = Settings(APP_ENV=app_env, DATABASE_URL="")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.validate_runtime()


def test_development_allows_empty_database_url_at_runtime() -> None:
    settings = Settings(APP_ENV="development", DATABASE_URL="")

    settings.validate_runtime()


@pytest.mark.parametrize("app_env", ["development", "staging", "production", "test"])
def test_require_database_url_rejects_empty_value_in_every_environment(app_env: str) -> None:
    settings = Settings(APP_ENV=app_env, DATABASE_URL="")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.require_database_url()


@pytest.mark.parametrize(
    ("invalid_setting", "field_name"),
    [
        ({"APP_PORT": 0}, "APP_PORT"),
        ({"APP_WORKERS": 0}, "APP_WORKERS"),
        ({"DB_POOL_TIMEOUT": 0}, "DB_POOL_TIMEOUT"),
        ({"DB_MAX_OVERFLOW": -1}, "DB_MAX_OVERFLOW"),
    ],
)
def test_numeric_database_and_runtime_settings_are_validated(
    invalid_setting: dict[str, object],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        Settings(**invalid_setting)


def test_settings_normalize_app_env_and_database_url() -> None:
    settings = Settings(
        APP_ENV=" Production ",
        DATABASE_URL=" postgresql+asyncpg://user:password@localhost/database ",
    )

    assert settings.APP_ENV == "production"
    assert settings.DATABASE_URL == "postgresql+asyncpg://user:password@localhost/database"


def test_require_database_url_rejects_non_asyncpg_scheme() -> None:
    settings = Settings(DATABASE_URL="postgresql://user:password@localhost/database")

    with pytest.raises(ConfigurationError, match=r"postgresql\+asyncpg"):
        settings.require_database_url()


def test_require_database_url_returns_valid_url() -> None:
    database_url = "postgresql+asyncpg://user:password@localhost/database"
    settings = Settings(DATABASE_URL=database_url)

    assert settings.require_database_url() == database_url
