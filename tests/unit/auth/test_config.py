from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError


DEPLOYED_DATABASE_URL = "postgresql+asyncpg://user:password@localhost/mathrag"
DEPLOYED_ORIGINS = ("https://mathrag.example",)


def _deployed_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "APP_ENV": "production",
        "DATABASE_URL": DEPLOYED_DATABASE_URL,
        "SESSION_SECRET": "x" * 32,
        "ALLOWED_ORIGINS": DEPLOYED_ORIGINS,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize("session_secret", ["", " ", "x" * 31])
def test_deployed_environments_reject_empty_or_short_session_secret(
    app_env: str,
    session_secret: str,
) -> None:
    with pytest.raises(ConfigurationError, match="SESSION_SECRET"):
        _deployed_settings(
            APP_ENV=app_env,
            SESSION_SECRET=session_secret,
        )


def test_session_secret_minimum_is_measured_in_utf8_bytes() -> None:
    configured = _deployed_settings(SESSION_SECRET="密" * 11)

    configured.validate_runtime()


@pytest.mark.parametrize("app_env", ["staging", "production"])
@pytest.mark.parametrize("allowed_origins", [(), ("",), ("*",), ("https://safe.example", "*")])
def test_deployed_environments_require_explicit_non_wildcard_origins(
    app_env: str,
    allowed_origins: tuple[str, ...],
) -> None:
    with pytest.raises(ConfigurationError, match="ALLOWED_ORIGINS"):
        _deployed_settings(
            APP_ENV=app_env,
            ALLOWED_ORIGINS=allowed_origins,
        )


def test_session_ttl_must_be_positive() -> None:
    with pytest.raises(ValueError, match="SESSION_TTL_SECONDS"):
        Settings(SESSION_TTL_SECONDS=0)


def test_development_uses_local_origins_by_default() -> None:
    configured = Settings(
        APP_ENV="development",
        SESSION_SECRET="",
        ALLOWED_ORIGINS=(),
    )

    assert configured.ALLOWED_ORIGINS == (
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    )


@pytest.mark.parametrize("app_env", ["development", "test"])
def test_wildcard_origin_is_rejected_in_every_environment(app_env: str) -> None:
    with pytest.raises(ConfigurationError, match="ALLOWED_ORIGINS"):
        Settings(APP_ENV=app_env, ALLOWED_ORIGINS=("*",))


@pytest.mark.parametrize(
    ("app_env", "session_cookie_name", "csrf_cookie_name"),
    [
        ("development", "mathrag_session", "mathrag_csrf"),
        ("staging", "__Host-mathrag_session", "__Host-mathrag_csrf"),
        ("production", "__Host-mathrag_session", "__Host-mathrag_csrf"),
        ("test", "__Host-mathrag_session", "__Host-mathrag_csrf"),
    ],
)
def test_cookie_names_are_derived_from_environment(
    app_env: str,
    session_cookie_name: str,
    csrf_cookie_name: str,
) -> None:
    configured = Settings(
        APP_ENV=app_env,
        SESSION_SECRET="x" * 32,
        ALLOWED_ORIGINS=DEPLOYED_ORIGINS,
    )

    assert configured.session_cookie_name == session_cookie_name
    assert configured.csrf_cookie_name == csrf_cookie_name
