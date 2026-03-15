from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
    DEBUG: bool = _to_bool(os.getenv("DEBUG"), True)

    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"

    RAW_KB_PATH: Path = PROJECT_ROOT / "data" / "raw" / "math_knowledge_seed.jsonl"
    PROCESSED_KB_PATH: Path = PROJECT_ROOT / "data" / "processed" / "kb_chunks.jsonl"
    FAISS_INDEX_PATH: Path = PROJECT_ROOT / "data" / "index" / "faiss.index"
    ID_MAP_PATH: Path = PROJECT_ROOT / "data" / "index" / "id_map.json"

    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    EMBEDDING_NORMALIZE: bool = _to_bool(os.getenv("EMBEDDING_NORMALIZE"), True)

    TOP_K: int = int(os.getenv("TOP_K", "3"))
    USE_INNER_PRODUCT: bool = _to_bool(os.getenv("USE_INNER_PRODUCT"), True)


settings = Settings()
