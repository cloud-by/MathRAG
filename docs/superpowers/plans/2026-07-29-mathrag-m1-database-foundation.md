# MathRAG M1 Database Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有 `/api/chat` 和 FAISS 检索行为的前提下，建立锁定版本的 PostgreSQL + pgvector、异步 SQLAlchemy 会话、Alembic 迁移以及可区分存活与就绪的健康检查。

**Architecture:** 应用保持模块化单体和单异步 Worker；进程内只共享一个延迟创建的 `AsyncEngine`/`async_sessionmaker`，每个请求获得独立 `AsyncSession`。Alembic 是唯一 schema 写入口，首个迁移只启用 `vector` 扩展；数据库故障只使 `/health/ready` 返回 503，不影响 `/health/live` 和既有 FAISS 聊天路径。

**Tech Stack:** Python 3.11.9、FastAPI 0.140.13、SQLAlchemy 2.0.51、asyncpg 0.31.0、Alembic 1.18.5、PostgreSQL 18.4、pgvector 0.8.5、pgvector Python 0.5.0、pytest、Docker Compose。

---

## 实施边界与文件职责

- `app/core/config.py`：环境变量解析、数值约束和生产必需配置校验。
- `app/core/errors.py`：基础设施稳定异常类型。
- `app/core/middleware.py`：请求 ID 接收、生成和响应回传。
- `app/infrastructure/database/base.py`：SQLAlchemy Base 与约束命名规则。
- `app/infrastructure/database/session.py`：延迟创建 Engine、Session factory、请求 dependency 和关闭逻辑。
- `app/infrastructure/database/types.py`：强制带时区 UTC 的数据库时间类型。
- `app/modules/system/service.py`：数据库、pgvector 和配置就绪检查。
- `app/modules/system/router.py`：`/health/live` 与 `/health/ready` HTTP 协议。
- `alembic/`：异步迁移环境与 `vector` 扩展基线 migration。
- `docker-compose.yml`：本地源码应用、PostgreSQL + pgvector、持久卷和健康依赖。
- `tests/`：配置/会话单元测试、健康 API 测试、真实数据库迁移回环测试。

M1 明确不创建知识、用户、会话等业务表，不导入 26 条知识，不切换 Retriever，不修改 `/api/chat` 响应结构。

## Task 1: 锁定数据库依赖并扩展配置契约

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements.lock.txt`
- Modify: `.env.example`
- Modify: `app/core/config.py`
- Create: `app/core/errors.py`
- Create: `tests/test_database_config.py`

- [ ] **Step 1: 写配置失败测试**

创建 `tests/test_database_config.py`：

```python
from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError


def test_database_pool_values_must_be_positive() -> None:
    with pytest.raises(ValueError, match="DB_POOL_SIZE"):
        Settings(DB_POOL_SIZE=0)


def test_production_requires_database_url() -> None:
    production = Settings(APP_ENV="production", DATABASE_URL="")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        production.validate_runtime()


def test_development_can_start_without_database_url() -> None:
    development = Settings(APP_ENV="development", DATABASE_URL="")

    development.validate_runtime()


def test_require_database_url_rejects_missing_value_in_every_environment() -> None:
    settings = Settings(APP_ENV="development", DATABASE_URL="")

    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        settings.require_database_url()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_database_config.py -q
```

Expected: collection 失败，提示 `ConfigurationError` 或数据库配置字段尚不存在。

- [ ] **Step 3: 增加异常类型和数据库配置**

创建 `app/core/errors.py`：

```python
from __future__ import annotations


class ConfigurationError(RuntimeError):
    """应用配置缺失或互相矛盾。"""
```

用以下完整内容替换 `app/core/config.py`：

```python
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
    APP_ENV: str = os.getenv("APP_ENV", "development").strip().lower()
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
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

    def __post_init__(self) -> None:
        positive_values = {
            "APP_PORT": self.APP_PORT,
            "APP_WORKERS": self.APP_WORKERS,
            "DB_POOL_SIZE": self.DB_POOL_SIZE,
            "DB_POOL_TIMEOUT": self.DB_POOL_TIMEOUT,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if self.DB_MAX_OVERFLOW < 0:
            raise ValueError("DB_MAX_OVERFLOW 不能小于 0")

    def validate_runtime(self) -> None:
        if self.APP_ENV in {"staging", "production"} and not self.DATABASE_URL:
            raise ConfigurationError("生产环境必须设置 DATABASE_URL")

    def require_database_url(self) -> str:
        if not self.DATABASE_URL:
            raise ConfigurationError("DATABASE_URL 未配置")
        if not self.DATABASE_URL.startswith("postgresql+asyncpg://"):
            raise ConfigurationError("DATABASE_URL 必须使用 postgresql+asyncpg 驱动")
        return self.DATABASE_URL


settings = Settings()
```

在 `requirements.txt` 末尾加入精确依赖：

```text
sqlalchemy==2.0.51
asyncpg==0.31.0
alembic==1.18.5
pgvector==0.5.0
pyyaml==6.0.3
```

在 `.env.example` 的应用配置后加入：

```dotenv
APP_ENV=development
APP_WORKERS=1
DATABASE_URL=postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag
TEST_DATABASE_URL=postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=5
DB_POOL_TIMEOUT=30
```

- [ ] **Step 4: 安装并刷新完整锁文件**

Run:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
uv pip freeze --python .\.venv\Scripts\python.exe
```

将命令输出逐行通过 `apply_patch` 更新 `requirements.lock.txt`。确认文件是 UTF-8、仅包含 `name==version`，且上述四个数据库包版本与 ADR-0001 一致。

- [ ] **Step 5: 运行配置测试和旧回归**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_database_config.py tests\test_runtime_configuration.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: 新配置测试全部 PASS；M0 的 39 项测试无回归，仅保留已记录的 Starlette/httpx 弃用警告。

- [ ] **Step 6: 提交依赖与配置契约**

```powershell
git add requirements.txt requirements.lock.txt .env.example app/core/config.py app/core/errors.py tests/test_database_config.py
git commit -m "feat: add M1 database configuration contract"
```

## Task 2: 建立异步数据库基础设施

**Files:**
- Create: `app/infrastructure/__init__.py`
- Create: `app/infrastructure/database/__init__.py`
- Create: `app/infrastructure/database/base.py`
- Create: `app/infrastructure/database/session.py`
- Create: `app/infrastructure/database/types.py`
- Create: `tests/test_database_session.py`
- Create: `tests/test_database_types.py`

- [ ] **Step 1: 写 Engine、Session 和 UTC 类型失败测试**

创建 `tests/test_database_session.py`：

```python
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
```

创建 `tests/test_database_types.py`：

```python
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.infrastructure.database.types import UTCDateTime


def test_utc_datetime_rejects_naive_values() -> None:
    column_type = UTCDateTime()

    with pytest.raises(ValueError, match="时区"):
        column_type.process_bind_param(datetime(2026, 7, 29, 12, 0), None)


def test_utc_datetime_normalizes_offset_values() -> None:
    column_type = UTCDateTime()
    value = datetime(2026, 7, 29, 16, 0, tzinfo=timezone(timedelta(hours=8)))

    assert column_type.process_bind_param(value, None) == datetime(
        2026, 7, 29, 8, 0, tzinfo=UTC
    )
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_database_session.py tests\test_database_types.py -q
```

Expected: FAIL，提示 `app.infrastructure.database` 尚不存在。

- [ ] **Step 3: 实现 Base、UTC 类型和延迟 Session 工厂**

创建空的 `app/infrastructure/__init__.py` 与 `app/infrastructure/database/__init__.py`。

创建 `app/infrastructure/database/base.py`：

```python
from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
```

创建 `app/infrastructure/database/types.py`：

```python
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("UTCDateTime 只接受带时区的 datetime")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
```

创建 `app/infrastructure/database/session.py`：

```python
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.require_database_url(),
            pool_size=settings.DB_POOL_SIZE,
            max_overflow=settings.DB_MAX_OVERFLOW,
            pool_timeout=settings.DB_POOL_TIMEOUT,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


def reset_database_state_for_tests() -> None:
    global _engine, _session_factory
    _engine = None
    _session_factory = None
```

- [ ] **Step 4: 运行定向测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_database_session.py tests\test_database_types.py -q
```

Expected: 4 tests PASS，导入模块期间没有数据库连接。

- [ ] **Step 5: 刷新锁文件并提交**

```powershell
uv pip freeze --python .\.venv\Scripts\python.exe
git add requirements.txt requirements.lock.txt app/infrastructure tests/test_database_session.py tests/test_database_types.py
git commit -m "feat: add async database infrastructure"
```

将 `uv pip freeze` 的精确输出通过补丁同步进 `requirements.lock.txt` 后再提交。

## Task 3: 建立可验证的本地 Compose 数据库

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`
- Create: `docker/postgres/init-test-db.sql`
- Create: `tests/test_compose_contract.py`

- [ ] **Step 1: 写 Compose 契约失败测试**

创建 `tests/test_compose_contract.py`：

```python
from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_compose_uses_local_app_and_pinned_pgvector() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    app = compose["services"]["mathrag"]
    postgres = compose["services"]["postgres"]

    assert app["build"] == {"context": ".", "dockerfile": "Dockerfile"}
    assert app["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "-w 1" in app["command"]
    assert postgres["image"] == "pgvector/pgvector:0.8.5-pg18-bookworm"
    assert "pg_isready" in postgres["healthcheck"]["test"][1]
    assert "postgres_data" in compose["volumes"]


def test_compose_exposes_database_only_on_loopback() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    assert compose["services"]["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
```

- [ ] **Step 2: 运行测试确认旧 Compose 不满足契约**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_compose_contract.py -q
```

Expected: 2 tests FAIL，分别指出缺少本地 `build` 和 `postgres` 服务。

- [ ] **Step 3: 实现双服务 Compose 与单 Worker 镜像**

用以下内容替换 `docker-compose.yml`：

```yaml
services:
  postgres:
    image: pgvector/pgvector:0.8.5-pg18-bookworm
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-mathrag}
      POSTGRES_USER: ${POSTGRES_USER:-mathrag}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mathrag-dev-only}
    ports:
      - "127.0.0.1:5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./docker/postgres/init-test-db.sql:/docker-entrypoint-initdb.d/10-create-test-db.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mathrag -d mathrag"]
      interval: 5s
      timeout: 3s
      retries: 20
    restart: unless-stopped

  mathrag:
    build:
      context: .
      dockerfile: Dockerfile
    image: mathrag:local
    env_file:
      - .env
    environment:
      APP_ENV: production
      APP_WORKERS: 1
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-mathrag}:${POSTGRES_PASSWORD:-mathrag-dev-only}@postgres:5432/${POSTGRES_DB:-mathrag}
    ports:
      - "127.0.0.1:8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
    command: gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:8000 --timeout 120
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8000/health/ready"]
      interval: 10s
      timeout: 5s
      retries: 12
    restart: unless-stopped

volumes:
  postgres_data:
```

创建 `docker/postgres/init-test-db.sql`：

```sql
CREATE DATABASE mathrag_test;
```

用以下内容替换 `Dockerfile`：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.lock.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120"]
```

- [ ] **Step 4: 验证 Compose 与契约测试**

```powershell
docker compose config --quiet
.\.venv\Scripts\python.exe -m pytest tests\test_compose_contract.py -q
docker compose build mathrag
```

Expected: Compose 配置退出码 0；2 tests PASS；`mathrag:local` 从当前工作区成功构建。

- [ ] **Step 5: 提交容器基础设施**

```powershell
git add docker-compose.yml Dockerfile docker/postgres/init-test-db.sql tests/test_compose_contract.py
git commit -m "feat: add PostgreSQL pgvector compose service"
```

## Task 4: 实现请求 ID 和 live/ready 健康检查

**Files:**
- Create: `app/core/middleware.py`
- Create: `app/modules/__init__.py`
- Create: `app/modules/system/__init__.py`
- Create: `app/modules/system/service.py`
- Create: `app/modules/system/router.py`
- Modify: `app/main.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/test_health.py`

- [ ] **Step 1: 写 API 行为失败测试**

创建空的 `tests/api/__init__.py`，再创建 `tests/api/test_health.py`：

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.modules.system.router import get_readiness_service
from app.modules.system.service import ReadinessResult


class FakeReadinessService:
    def __init__(self, result: ReadinessResult) -> None:
        self.result = result

    async def check(self) -> ReadinessResult:
        return self.result


def build_client(result: ReadinessResult) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_readiness_service] = lambda: FakeReadinessService(result)
    return TestClient(app)


def test_live_does_not_depend_on_database_readiness() -> None:
    client = build_client(
        ReadinessResult(False, {"config": "ok", "database": "unavailable", "pgvector": "unknown"})
    )

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_200_when_all_checks_pass() -> None:
    client = build_client(
        ReadinessResult(True, {"config": "ok", "database": "ok", "pgvector": "0.8.5"})
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"config": "ok", "database": "ok", "pgvector": "0.8.5"},
    }


def test_ready_returns_503_without_leaking_database_error() -> None:
    client = build_client(
        ReadinessResult(False, {"config": "ok", "database": "unavailable", "pgvector": "unknown"})
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "password" not in response.text.lower()


def test_request_id_is_accepted_and_returned() -> None:
    client = build_client(
        ReadinessResult(True, {"config": "ok", "database": "ok", "pgvector": "0.8.5"})
    )

    response = client.get("/health/live", headers={"X-Request-ID": "m1-health-001"})

    assert response.headers["X-Request-ID"] == "m1-health-001"
```

- [ ] **Step 2: 运行测试确认路由尚不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_health.py -q
```

Expected: collection 或请求 FAIL，原因是 `app.modules.system`、`/health/live`、`/health/ready` 尚不存在。

- [ ] **Step 3: 实现请求 ID 中间件**

创建 `app/core/middleware.py`：

```python
from __future__ import annotations

import re
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "").strip()
        request_id = incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

- [ ] **Step 4: 实现就绪 service 与 system router**

创建空的 `app/modules/__init__.py` 和 `app/modules/system/__init__.py`。

创建 `app/modules/system/service.py`：

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import settings
from app.core.errors import ConfigurationError
from app.infrastructure.database.session import get_engine


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    checks: dict[str, str]

    def as_payload(self) -> dict[str, object]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": self.checks,
        }


class ReadinessService:
    def __init__(self, engine_provider: Callable[[], AsyncEngine] = get_engine) -> None:
        self.engine_provider = engine_provider

    async def check(self) -> ReadinessResult:
        checks = {"config": "ok", "database": "unknown", "pgvector": "unknown"}
        try:
            settings.require_database_url()
        except ConfigurationError:
            checks["config"] = "invalid"
            return ReadinessResult(False, checks)

        try:
            async with self.engine_provider().connect() as connection:
                version = (
                    await connection.execute(
                        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    )
                ).scalar_one_or_none()
        except (SQLAlchemyError, OSError):
            checks["database"] = "unavailable"
            return ReadinessResult(False, checks)

        checks["database"] = "ok"
        if version is None:
            checks["pgvector"] = "missing"
            return ReadinessResult(False, checks)
        checks["pgvector"] = str(version)
        return ReadinessResult(True, checks)
```

创建 `app/modules/system/router.py`：

```python
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.modules.system.service import ReadinessService


router = APIRouter(tags=["system"])


def get_readiness_service() -> ReadinessService:
    return ReadinessService()


@router.get("/health/live", summary="进程存活检查")
async def live() -> dict[str, str]:
    return {"status": "ok", "app_name": settings.APP_NAME}


@router.get("/health/ready", summary="应用就绪检查", response_model=None)
async def ready(
    service: Annotated[ReadinessService, Depends(get_readiness_service)],
) -> dict[str, object] | JSONResponse:
    result = await service.check()
    payload = result.as_payload()
    if not result.ready:
        return JSONResponse(status_code=503, content=payload)
    return payload
```

- [ ] **Step 5: 在应用生命周期注册基础设施**

用以下完整内容替换 `app/main.py`：

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.knowledge import router as knowledge_router
from app.core.config import settings
from app.core.middleware import RequestIdMiddleware
from app.infrastructure.database.session import dispose_engine
from app.modules.system.router import router as system_router
from app.schemas.chat import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_runtime()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="基于 FastAPI + FAISS + 大模型 API 的数学 RAG 问答原型系统",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"], summary="兼容健康检查")
    def health() -> HealthResponse:
        return HealthResponse(app_name=settings.APP_NAME)

    @app.get("/", tags=["system"], include_in_schema=False, response_model=None)
    def root() -> JSONResponse | HTMLResponse:
        frontend_dir = Path(__file__).resolve().parent / "frontend"
        index_file = frontend_dir / "index.html"
        if not index_file.exists():
            return JSONResponse(
                {
                    "message": f"{settings.APP_NAME} 已启动。",
                    "docs": "/docs",
                    "chat_api": "/api/chat",
                    "health": "/health",
                }
            )
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(knowledge_router)

    frontend_dir = Path(__file__).resolve().parent / "frontend"
    if frontend_dir.exists():
        # API 路由先注册，根路径静态挂载不会吞掉 /api/* 和 /health/*。
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    return app


app = create_app()
```

M1 保留当前 CORS 行为以避免把安全迁移与数据库底座混在同一阶段；ADR-0001 的显式来源策略在认证 Cookie 引入前落实。

- [ ] **Step 6: 运行健康测试与全量回归**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_health.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: 4 项健康测试 PASS；旧 `/health`、`/api/chat`、知识抽取和 M0 检索测试继续 PASS。

- [ ] **Step 7: 提交健康检查**

```powershell
git add app/core/middleware.py app/modules app/main.py tests/api
git commit -m "feat: distinguish liveness and database readiness"
```

## Task 5: 建立 Alembic 异步迁移与 pgvector 扩展回环

**Files:**
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Create: `alembic/script.py.mako`
- Create: `alembic/versions/0001_enable_vector_extension.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_migrations.py`

- [ ] **Step 1: 写真实数据库迁移失败测试**

创建空的 `tests/integration/__init__.py`，再创建 `tests/integration/test_migrations.py`：

```python
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_alembic(database_url: str, *args: str) -> None:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = database_url
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "alembic.ini", *args],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


async def vector_extension_version(database_url: str) -> str | None:
    connection = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        return await connection.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
    finally:
        await connection.close()


def test_migration_upgrade_downgrade_upgrade_round_trip() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL 未配置")

    run_alembic(database_url, "downgrade", "base")
    assert asyncio.run(vector_extension_version(database_url)) is None

    run_alembic(database_url, "upgrade", "head")
    assert asyncio.run(vector_extension_version(database_url)) == "0.8.5"

    run_alembic(database_url, "downgrade", "base")
    assert asyncio.run(vector_extension_version(database_url)) is None

    run_alembic(database_url, "upgrade", "head")
    assert asyncio.run(vector_extension_version(database_url)) == "0.8.5"
```

- [ ] **Step 2: 启动测试数据库并确认测试失败**

```powershell
docker compose up -d postgres
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\test_migrations.py -q
```

Expected: FAIL，提示缺少 `alembic.ini` 或 revision。

- [ ] **Step 3: 创建 Alembic 配置与异步环境**

创建 `alembic.ini`：

```ini
[alembic]
script_location = %(here)s/alembic
prepend_sys_path = .
path_separator = os
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

创建 `alembic/env.py`：

```python
from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import settings
from app.infrastructure.database.base import Base


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
config.set_main_option("sqlalchemy.url", settings.require_database_url().replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

创建 `alembic/script.py.mako`：

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 4: 创建只管理扩展的首个 migration**

创建 `alembic/versions/0001_enable_vector_extension.py`：

```python
"""启用 pgvector 扩展。"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "0001_enable_vector_extension"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
```

- [ ] **Step 5: 执行迁移回环与就绪检查**

```powershell
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\test_migrations.py -q
$env:DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\alembic.exe current
```

Expected: migration test PASS；`alembic current` 输出 `0001_enable_vector_extension (head)`。

- [ ] **Step 6: 启动完整 Compose 并验证健康语义**

```powershell
docker compose up -d --build
docker compose ps
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Expected: `postgres` 与 `mathrag` 均为 healthy；live 返回 `status=ok`；ready 返回 `status=ready`、`database=ok`、`pgvector=0.8.5`。

停止 PostgreSQL 后验证差异，再恢复：

```powershell
docker compose stop postgres
Invoke-WebRequest http://127.0.0.1:8000/health/live -SkipHttpErrorCheck
Invoke-WebRequest http://127.0.0.1:8000/health/ready -SkipHttpErrorCheck
docker compose start postgres
```

Expected: live 为 200，ready 为 503；数据库恢复后 ready 自动回到 200。

- [ ] **Step 7: 提交迁移基线**

```powershell
git add alembic.ini alembic tests/integration/test_migrations.py
git commit -m "feat: add pgvector migration baseline"
```

## Task 6: 文档、反模式检查与 M1 总验收

**Files:**
- Modify: `README.md`
- Create: `docs/baselines/2026-07-29-m1-database-foundation.md`

- [ ] **Step 1: 更新 README 的 M1 本地流程**

在 README Docker 章节加入以下完整命令块：

````markdown
### 本地数据库开发

```powershell
Copy-Item .env.example .env
docker compose up -d postgres
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\python.exe run.py
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

`/health/live` 只检查应用进程；`/health/ready` 同时检查数据库、关键配置和 pgvector 扩展。
````

- [ ] **Step 2: 执行静态反模式检查**

```powershell
rg -n "create_all\(|AsyncSession\(" app
rg -n "faiss|id_map|kb_chunks" app\modules app\infrastructure
```

Expected: `create_all(` 无结果；`AsyncSession(` 不存在跨请求全局实例；新增模块不引用 FAISS、`id_map` 或 processed JSONL。

- [ ] **Step 3: 执行 M1 完整验收**

```powershell
docker compose config --quiet
docker compose up -d postgres
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\test_migrations.py tests\api\test_health.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.capture_retrieval_baseline --fixture tests\fixtures\retrieval_questions.json --output docs\baselines\artifacts\faiss-top3-m1-regression.json
git diff --check
```

Expected:

- 迁移回环与健康 API 全部 PASS。
- 全量测试不低于 M0 的 39 项，除已记录的 Starlette/httpx 警告外无新警告。
- FAISS 固定题集仍为 26/26 期望命中。
- `git diff --check` 退出码 0。

- [ ] **Step 4: 写 M1 基线报告**

创建 `docs/baselines/2026-07-29-m1-database-foundation.md`，逐项记录：Git commit、Python/数据库/扩展/包版本、Compose 服务状态、migration current、upgrade/downgrade/upgrade 结果、live/ready 正常与断库状态码、全量测试结果、FAISS 回归命中率和未解决警告。报告不得包含 `.env` 值或数据库密码。

- [ ] **Step 5: 停止测试服务并提交 M1 收口**

```powershell
docker compose down
git add README.md docs/baselines/2026-07-29-m1-database-foundation.md
git commit -m "docs: record M1 database foundation verification"
git status --short
```

Expected: Compose 服务停止；工作树干净；M1 的六个提交均存在且未混入 M2 业务表或检索切换。

## M1 完成定义

- 空 `mathrag_test` 数据库能执行 `base -> head -> base -> head`。
- `/health/live` 在数据库断开时仍为 200；`/health/ready` 为 503，恢复后回到 200。
- 应用 import 不连接数据库，启动不修改 schema。
- 一个进程共享一个 Engine/sessionmaker，每个请求独立 Session，异常回滚并关闭。
- Compose 构建当前工作区，数据库镜像固定为 `0.8.5-pg18-bookworm`，应用只有一个 Worker。
- 旧 `/api/chat`、FAISS 和 26 题固定检索基线无回归。
- M1 不包含任何知识、用户、会话或导入业务表。

## 官方实现依据

- [SQLAlchemy 2.0 asyncio](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic async template](https://alembic.sqlalchemy.org/en/latest/cookbook.html#using-asyncio-with-alembic)
- [PostgreSQL CREATE EXTENSION](https://www.postgresql.org/docs/current/sql-createextension.html)
- [pgvector installation](https://github.com/pgvector/pgvector#installation)
- [pgvector Python SQLAlchemy integration](https://github.com/pgvector/pgvector-python#sqlalchemy)
