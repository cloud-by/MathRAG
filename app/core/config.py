from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.core.errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
DEVELOPMENT_ALLOWED_ORIGINS = (
    "http://127.0.0.1:8000",
    "http://localhost:8000",
)
DEVELOPMENT_SESSION_SECRET = "mathrag-development-only-session-secret"

# 允许在项目根目录存在 .env 时自动加载
load_dotenv(ENV_PATH if ENV_PATH.exists() else None)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_origins(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "MathRAG MVP")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    APP_ENV: str = os.getenv("APP_ENV", "development").strip().lower()
    APP_WORKERS: int = int(os.getenv("APP_WORKERS", "1"))
    DEBUG: bool = _to_bool(os.getenv("DEBUG"), True)

    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "604800"))
    ALLOWED_ORIGINS: tuple[str, ...] = field(
        default_factory=lambda: _to_origins(os.getenv("ALLOWED_ORIGINS"))
    )

    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "5"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"
    UPLOAD_DIR: Path = Path(
        os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "data" / "uploads"))
    )
    MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "200"))
    MAX_INGESTION_TEXT_CHARS: int = int(
        os.getenv("MAX_INGESTION_TEXT_CHARS", "200000")
    )
    INGESTION_CHUNK_CHARS: int = int(os.getenv("INGESTION_CHUNK_CHARS", "4000"))

    RAW_KB_PATH: Path = PROJECT_ROOT / "data" / "raw" / "math_knowledge_seed.jsonl"
    PROCESSED_KB_PATH: Path = PROJECT_ROOT / "data" / "processed" / "kb_chunks.jsonl"

    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    EMBEDDING_NORMALIZE: bool = _to_bool(os.getenv("EMBEDDING_NORMALIZE"), True)

    TOP_K: int = int(os.getenv("TOP_K", "3"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "APP_ENV", self.APP_ENV.strip().lower())
        object.__setattr__(self, "DATABASE_URL", self.DATABASE_URL.strip())
        session_secret = self.SESSION_SECRET.strip()
        origins = tuple(
            origin.strip()
            for origin in self.ALLOWED_ORIGINS
            if origin.strip()
        )
        if self.APP_ENV == "development":
            # 仅供本地开发启动；部署环境必须显式提供强密钥。
            session_secret = session_secret or DEVELOPMENT_SESSION_SECRET
            origins = origins or DEVELOPMENT_ALLOWED_ORIGINS
        object.__setattr__(self, "SESSION_SECRET", session_secret)
        object.__setattr__(self, "ALLOWED_ORIGINS", origins)

        positive_settings = (
            ("APP_PORT", self.APP_PORT),
            ("APP_WORKERS", self.APP_WORKERS),
            ("DB_POOL_SIZE", self.DB_POOL_SIZE),
            ("DB_POOL_TIMEOUT", self.DB_POOL_TIMEOUT),
            ("SESSION_TTL_SECONDS", self.SESSION_TTL_SECONDS),
            ("MAX_UPLOAD_BYTES", self.MAX_UPLOAD_BYTES),
            ("MAX_PDF_PAGES", self.MAX_PDF_PAGES),
            ("MAX_INGESTION_TEXT_CHARS", self.MAX_INGESTION_TEXT_CHARS),
            ("INGESTION_CHUNK_CHARS", self.INGESTION_CHUNK_CHARS),
        )
        for field_name, value in positive_settings:
            if value <= 0:
                raise ValueError(f"{field_name} 必须大于 0")

        if self.DB_MAX_OVERFLOW < 0:
            raise ValueError("DB_MAX_OVERFLOW 必须大于等于 0")

        self._validate_security_config()

    def _validate_security_config(self) -> None:
        if "*" in self.ALLOWED_ORIGINS:
            raise ConfigurationError("ALLOWED_ORIGINS 不得包含通配符 *")
        if self.APP_ENV in {"staging", "production"}:
            if len(self.SESSION_SECRET.encode("utf-8")) < 32:
                raise ConfigurationError(
                    f"{self.APP_ENV} 环境的 SESSION_SECRET 必须至少包含 32 个 UTF-8 字节"
                )
            if not self.ALLOWED_ORIGINS:
                raise ConfigurationError(
                    f"{self.APP_ENV} 环境必须配置 ALLOWED_ORIGINS"
                )

    def validate_runtime(self) -> None:
        if self.APP_ENV in {"staging", "production"} and not self.DATABASE_URL:
            raise ConfigurationError(
                f"{self.APP_ENV} 环境必须配置 DATABASE_URL"
            )
        self._validate_security_config()

    @property
    def session_cookie_name(self) -> str:
        if self.APP_ENV == "development":
            return "mathrag_session"
        return "__Host-mathrag_session"

    @property
    def csrf_cookie_name(self) -> str:
        if self.APP_ENV == "development":
            return "mathrag_csrf"
        return "__Host-mathrag_csrf"

    def require_database_url(self) -> str:
        if not self.DATABASE_URL:
            raise ConfigurationError("DATABASE_URL 不能为空")
        if not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            raise ConfigurationError(
                "DATABASE_URL 必须使用 postgresql+asyncpg:// 前缀"
            )
        return self.DATABASE_URL


settings = Settings()
