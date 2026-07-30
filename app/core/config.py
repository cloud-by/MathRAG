from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from app.core.errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 允许在项目根目录存在 .env 时自动加载
load_dotenv(ENV_PATH if ENV_PATH.exists() else None)


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "MathRAG MVP")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
    APP_ENV: str = os.getenv("APP_ENV", "development").strip().lower()
    APP_WORKERS: int = int(os.getenv("APP_WORKERS", "1"))
    DEBUG: bool = _to_bool(os.getenv("DEBUG"), True)

    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "5"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"

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

        positive_settings = (
            ("APP_PORT", self.APP_PORT),
            ("APP_WORKERS", self.APP_WORKERS),
            ("DB_POOL_SIZE", self.DB_POOL_SIZE),
            ("DB_POOL_TIMEOUT", self.DB_POOL_TIMEOUT),
        )
        for field_name, value in positive_settings:
            if value <= 0:
                raise ValueError(f"{field_name} 必须大于 0")

        if self.DB_MAX_OVERFLOW < 0:
            raise ValueError("DB_MAX_OVERFLOW 必须大于等于 0")

    def validate_runtime(self) -> None:
        if self.APP_ENV in {"staging", "production"} and not self.DATABASE_URL:
            raise ConfigurationError(
                f"{self.APP_ENV} 环境必须配置 DATABASE_URL"
            )

    def require_database_url(self) -> str:
        if not self.DATABASE_URL:
            raise ConfigurationError("DATABASE_URL 不能为空")
        if not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            raise ConfigurationError(
                "DATABASE_URL 必须使用 postgresql+asyncpg:// 前缀"
            )
        return self.DATABASE_URL


settings = Settings()
