# MathRAG M2 旧知识数据迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 `knowledge_items` 与 `knowledge_chunks`，并提供可审计、可回滚、可重复执行的 UTF-8 导入命令，将现有 26 条 JSONL 知识和 26 条 chunk 无损迁移到 PostgreSQL。

**Architecture:** M2 只增加离线迁移链路，不修改在线聊天和 FAISS 检索。Loader 读取并配对 raw/processed JSONL；Pydantic schema 负责严格校验和稳定摘要；Service 持有整批事务并决定创建、跳过或冲突；Repository 只访问数据库，不提交事务。相同 `legacy_id` 且持久化内容摘要一致时跳过，相同 ID 内容不一致时拒绝整批导入并回滚。

**Tech Stack:** Python 3.11.9、Pydantic 2.13.4、SQLAlchemy 2.0.51 asyncio、asyncpg 0.31.0、Alembic 1.18.5、PostgreSQL 18.4、pgvector 0.8.5 / pgvector Python 0.5.0、pytest 9.1.1、Docker Compose。

---

## 已冻结的设计决策

- 输入固定为 `data/raw/math_knowledge_seed.jsonl` 与 `data/processed/kb_chunks.jsonl`，均显式按 UTF-8 读取，导入过程不修改源文件。
- raw 的 `id` 与 chunk 的 `source_id` 必须一一对应；M2 固定每个知识点一个 `chunk_index=0` 的 chunk。
- 首次导入创建；同一 `legacy_id` 的规范化持久化摘要相同则跳过；摘要不同则抛出冲突，整批回滚，绝不覆盖。
- `knowledge_items.status="indexing"`、`knowledge_chunks.status="pending"`；`embedding` 与 `embedding_model` 保持空值，M3 向量化完成后才能改为 `ready`。
- `owner_id` 与 `document_id` 不在 M2 提前创建无外键列，待对应实体落库时通过后续 migration 增加。
- `/api/chat`、`app/services/retriever.py`、FAISS、`id_map.json` 和 processed JSONL 的在线读取路径保持不变；M2 不做在线双写。
- Repository 不调用 `commit()` 或 `rollback()`；Service 用 `async with session.begin()` 管理唯一事务边界。
- CLI 只从 `DATABASE_URL` 配置读取连接信息；输出计数、摘要与错误码，不输出完整知识正文或数据库口令。
- 不增加 `pytest-asyncio`；异步测试沿用项目现有做法，由同步测试函数调用 `asyncio.run()`。

## 文件清单

| 文件 | 操作 | 职责 |
|---|---|---|
| `app/modules/knowledge/__init__.py` | Create | knowledge 模块入口 |
| `app/modules/knowledge/errors.py` | Create | 输入、重复 ID、冲突异常 |
| `app/modules/knowledge/schemas.py` | Create | 严格输入、跨文件校验、规范化摘要、结果模型 |
| `app/modules/knowledge/models.py` | Create | 两张知识表的 ORM 映射 |
| `app/modules/knowledge/repository.py` | Create | 查询、添加、计数、有序读取，不提交事务 |
| `app/modules/knowledge/service.py` | Create | 幂等判断、冲突判断、整批事务 |
| `app/modules/knowledge/legacy_loader.py` | Create | UTF-8 JSONL 逐行解析、唯一性和一一配对 |
| `alembic/env.py` | Modify | 注册知识模型 metadata |
| `alembic/versions/0002_create_knowledge_tables.py` | Create | 建表、约束、索引和回滚 |
| `scripts/import_legacy_knowledge.py` | Create | 离线导入 CLI |
| `tests/unit/knowledge/*` | Create | schema、model、loader、service 单元测试 |
| `tests/integration/knowledge/*` | Create | migration、repository、回滚、真实双次导入 |
| `docs/baselines/2026-07-30-m2-legacy-knowledge-migration.md` | Create | M2 验收证据和回滚说明 |

## M2 验收常量

```text
raw 文件 SHA-256       = 2593f45081b11ab4ae280d1a7fb107791b3099c364f3813f215a73fa7369d062
processed 文件 SHA-256 = a0334a626d7e54ce04a447861af1616da26ad8b012d81f6720aa1d404539e5aa
raw 记录数              = 26
processed chunk 数      = 26
knowledge_items         = 26
knowledge_chunks        = 26
唯一且非空 legacy_id    = 26
首次导入                = created 26 / skipped 0 / conflicts 0 / failed 0
第二次导入              = created 0 / skipped 26 / conflicts 0 / failed 0
FAISS 固定题集命中       = 26/26
```

---

## Task 1: 建立旧知识输入契约与稳定摘要

**Files:**

- Create: `app/modules/knowledge/__init__.py`
- Create: `app/modules/knowledge/errors.py`
- Create: `app/modules/knowledge/schemas.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/knowledge/__init__.py`
- Create: `tests/unit/knowledge/test_schemas.py`

- [ ] **Step 1: 写失败测试**

在 `tests/unit/knowledge/test_schemas.py` 固定以下行为：

```python
def test_bundle_rejects_source_id_mismatch() -> None:
    with pytest.raises(ValidationError, match="source_id"):
        LegacyKnowledgeBundle(item=make_item("k0001"), chunk=make_chunk("k0002"))


def test_bundle_rejects_persisted_field_mismatch() -> None:
    chunk = make_chunk("k0001").model_copy(update={"title": "被修改的标题"})
    with pytest.raises(ValidationError, match="title"):
        LegacyKnowledgeBundle(item=make_item("k0001"), chunk=chunk)


def test_digest_ignores_json_key_order_but_not_content() -> None:
    first = make_bundle(metadata={"a": 1, "b": 2})
    reordered = make_bundle(metadata={"b": 2, "a": 1})
    changed = make_bundle(metadata={"a": 1, "b": 3})
    assert first.sha256() == reordered.sha256()
    assert first.sha256() != changed.sha256()
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_schemas.py -q
```

Expected: FAIL，提示 `app.modules.knowledge.schemas` 不存在。

- [ ] **Step 3: 实现异常和 schema**

`errors.py` 定义以下稳定异常层级：

```python
class LegacyKnowledgeImportError(Exception):
    """旧知识导入失败。"""


class LegacyKnowledgeInputError(LegacyKnowledgeImportError):
    """输入文件格式或配对关系无效。"""


class DuplicateLegacyIdError(LegacyKnowledgeInputError):
    """同一输入批次出现重复旧 ID。"""


class LegacyKnowledgeConflictError(LegacyKnowledgeImportError):
    """数据库已有同 ID、不同内容的记录。"""
```

`schemas.py` 使用 `ConfigDict(extra="forbid")` 定义：

```python
class LegacyKnowledgeItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    keywords: list[str]
    content: str = Field(min_length=1)
    example: str
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]


class LegacyKnowledgeChunkInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    chunk_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=64)
    category: str
    title: str
    keywords: list[str]
    content: str
    example: str
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    source_line: int = Field(ge=1)
    retrieval_text: str = Field(min_length=1)
    answer_context: str = Field(min_length=1)
    metadata: dict[str, object]


class LegacyKnowledgeBundle(BaseModel):
    item: LegacyKnowledgeItemInput
    chunk: LegacyKnowledgeChunkInput
    chunk_index: int = 0

    @model_validator(mode="after")
    def validate_pair(self) -> "LegacyKnowledgeBundle":
        if self.item.id != self.chunk.source_id:
            raise ValueError("item.id 与 chunk.source_id 不一致")
        for field in ("category", "title", "keywords", "content", "example", "steps", "difficulty"):
            if getattr(self.item, field) != getattr(self.chunk, field):
                raise ValueError(f"raw/chunk 字段不一致: {field}")
        return self

    def persistent_payload(self) -> dict[str, object]:
        return {
            "item": self.item.model_dump(mode="json"),
            "chunk": {
                "chunk_index": self.chunk_index,
                "retrieval_text": self.chunk.retrieval_text,
                "answer_context": self.chunk.answer_context,
                "metadata": {
                    **self.chunk.metadata,
                    "legacy_chunk_id": self.chunk.chunk_id,
                    "legacy_source_id": self.chunk.source_id,
                    "source_line": self.chunk.source_line,
                },
            },
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.persistent_payload(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
```

同时实现 `collection_sha256(bundles)`：先按 `item.id` 排序，再对每个 bundle 的 `persistent_payload()` 组成列表，用同样的 JSON 参数和 UTF-8 编码计算 SHA-256。定义 `LegacyImportSummary` 字段：`input_items`、`input_chunks`、`created`、`skipped`、`conflicts`、`failed`、`database_items`、`database_chunks`、`input_sha256`、`database_sha256`。

- [ ] **Step 4: 运行测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_schemas.py -q
git add app/modules/knowledge tests/unit
git commit -m "feat: define legacy knowledge import contract"
```

Expected: schema 定向测试全部 PASS，提交不包含数据库实现。

---

## Task 2: 创建知识 ORM 模型和 0002 migration

**Files:**

- Create: `app/modules/knowledge/models.py`
- Create: `alembic/versions/0002_create_knowledge_tables.py`
- Modify: `alembic/env.py`
- Create: `tests/unit/knowledge/test_models.py`
- Create: `tests/integration/knowledge/__init__.py`
- Create: `tests/integration/knowledge/test_migration_schema.py`

- [ ] **Step 1: 写模型和迁移失败测试**

`test_models.py` 直接检查 metadata：表名、`uq_knowledge_items_legacy_id`、`uq_knowledge_chunks_knowledge_item_id_chunk_index`、两个状态约束、CASCADE 外键以及 `VECTOR(1024)`。

`test_migration_schema.py` 沿用 `tests/integration/test_migrations.py` 的 `run_alembic()` 与 `asyncio.run()` 模式，执行：

```python
run_alembic(database_url, "downgrade", "base")
run_alembic(database_url, "upgrade", "head")
assert asyncio.run(read_knowledge_tables(database_url)) == {
    "knowledge_items",
    "knowledge_chunks",
}
assert asyncio.run(read_embedding_type(database_url)) == "vector(1024)"
run_alembic(database_url, "downgrade", "0001_enable_vector_extension")
assert asyncio.run(read_knowledge_tables(database_url)) == set()
run_alembic(database_url, "upgrade", "head")
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_models.py -q
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_migration_schema.py -q
```

Expected: ORM 模块和 0002 尚不存在，测试 FAIL。

- [ ] **Step 3: 实现 ORM 字段与关系**

`KnowledgeItem`：

```text
id UUID PK default uuid4
legacy_id varchar(64) nullable UNIQUE
category varchar(128) NOT NULL indexed
title varchar(255) NOT NULL
keywords JSONB NOT NULL
content text NOT NULL
example text NOT NULL default ""
steps JSONB NOT NULL
difficulty varchar(16) NOT NULL CHECK easy|medium|hard
visibility varchar(16) NOT NULL default public CHECK public|private
status varchar(16) NOT NULL default indexing CHECK draft|indexing|ready|failed|archived
revision integer NOT NULL default 1 CHECK revision > 0
created_at UTCDateTime server default now
updated_at UTCDateTime server default now, onupdate now
```

`KnowledgeChunk`：

```text
id UUID PK default uuid4
knowledge_item_id UUID NOT NULL FK knowledge_items.id ON DELETE CASCADE
chunk_index integer NOT NULL CHECK >= 0
retrieval_text text NOT NULL
answer_context text NOT NULL
embedding vector(1024) nullable
embedding_model varchar(128) nullable
metadata JSONB NOT NULL（ORM 属性名 metadata_）
status varchar(16) NOT NULL default pending CHECK pending|ready|failed
created_at UTCDateTime server default now
UNIQUE(knowledge_item_id, chunk_index)
```

关系使用 `back_populates`、`cascade="all, delete-orphan"` 和 `passive_deletes=True`。

- [ ] **Step 4: 注册模型并实现 migration**

在 `alembic/env.py` 导入模型，使 `Base.metadata` 包含两张表：

```python
from app.modules.knowledge.models import KnowledgeChunk, KnowledgeItem  # noqa: F401
```

`0002_create_knowledge_tables.py`：

```python
"""创建旧知识迁移所需的知识表。"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision: str = "0002_create_knowledge_tables"
down_revision: str | None = "0001_enable_vector_extension"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("legacy_id", sa.String(length=64), nullable=True),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("keywords", postgresql.JSONB(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("example", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("difficulty", sa.String(length=16), nullable=False),
        sa.Column(
            "visibility", sa.String(length=16),
            server_default=sa.text("'public'"), nullable=False,
        ),
        sa.Column(
            "status", sa.String(length=16),
            server_default=sa.text("'indexing'"), nullable=False,
        ),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.CheckConstraint(
            "difficulty IN ('easy', 'medium', 'hard')",
            name="ck_knowledge_items_difficulty",
        ),
        sa.CheckConstraint(
            "visibility IN ('public', 'private')",
            name="ck_knowledge_items_visibility",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'indexing', 'ready', 'failed', 'archived')",
            name="ck_knowledge_items_status",
        ),
        sa.CheckConstraint("revision > 0", name="ck_knowledge_items_revision"),
        sa.PrimaryKeyConstraint("id", name="pk_knowledge_items"),
        sa.UniqueConstraint("legacy_id", name="uq_knowledge_items_legacy_id"),
    )
    op.create_index("ix_knowledge_items_category", "knowledge_items", ["category"])
    op.create_index("ix_knowledge_items_status", "knowledge_items", ["status"])

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("knowledge_item_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("retrieval_text", sa.Text(), nullable=False),
        sa.Column("answer_context", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status", sa.String(length=16),
            server_default=sa.text("'pending'"), nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.text("now()"), nullable=False,
        ),
        sa.CheckConstraint(
            "chunk_index >= 0", name="ck_knowledge_chunks_chunk_index"
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'ready', 'failed')",
            name="ck_knowledge_chunks_status",
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_item_id"], ["knowledge_items.id"],
            name="fk_knowledge_chunks_knowledge_item_id_knowledge_items",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_knowledge_chunks"),
        sa.UniqueConstraint(
            "knowledge_item_id", "chunk_index",
            name="uq_knowledge_chunks_knowledge_item_id_chunk_index",
        ),
    )
    op.create_index("ix_knowledge_chunks_status", "knowledge_chunks", ["status"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_status", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_index("ix_knowledge_items_status", table_name="knowledge_items")
    op.drop_index("ix_knowledge_items_category", table_name="knowledge_items")
    op.drop_table("knowledge_items")
```

迁移必须手写并与 ORM 逐字段核对，不依赖当次环境 autogenerate 的偶然输出。

- [ ] **Step 5: 验证 upgrade/downgrade/upgrade 并提交**

```powershell
docker compose up -d postgres
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_models.py tests\integration\knowledge\test_migration_schema.py -q
$env:DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
.\.venv\Scripts\python.exe -m alembic -c alembic.ini current
git add app/modules/knowledge/models.py alembic/env.py alembic/versions/0002_create_knowledge_tables.py tests/unit/knowledge/test_models.py tests/integration/knowledge
git commit -m "feat: add knowledge persistence schema"
```

Expected: 定向测试 PASS；主库 current 为 `0002_create_knowledge_tables (head)`；测试库可往返迁移。

---

## Task 3: 实现不提交事务的 Knowledge Repository

**Files:**

- Create: `app/modules/knowledge/repository.py`
- Create: `tests/integration/knowledge/test_repository.py`

- [ ] **Step 1: 写真实数据库失败测试**

测试用独立 session 清表，然后在事务中添加一个 item 和一个 chunk；事务结束后换 session 查询并断言字段与计数。测试必须覆盖：

```python
assert await repository.get_by_legacy_id("k9001") is not None
assert await repository.count_legacy_items() == 1
assert await repository.count_legacy_chunks() == 1
assert [item.legacy_id for item in await repository.list_legacy_items_ordered()] == ["k9001"]
```

- [ ] **Step 2: 确认测试先失败**

```powershell
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_repository.py -q
```

Expected: FAIL，提示 Repository 不存在。

- [ ] **Step 3: 实现 Repository**

```python
class KnowledgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_legacy_id(self, legacy_id: str) -> KnowledgeItem | None:
        statement = (
            select(KnowledgeItem)
            .options(selectinload(KnowledgeItem.chunks))
            .where(KnowledgeItem.legacy_id == legacy_id)
        )
        return (await self.session.execute(statement)).scalar_one_or_none()

    def add(self, item: KnowledgeItem) -> None:
        self.session.add(item)

    async def count_legacy_items(self) -> int:
        statement = select(func.count()).select_from(KnowledgeItem).where(
            KnowledgeItem.legacy_id.is_not(None)
        )
        return int((await self.session.execute(statement)).scalar_one())

    async def count_legacy_chunks(self) -> int:
        statement = select(func.count()).select_from(KnowledgeChunk).join(
            KnowledgeItem
        ).where(KnowledgeItem.legacy_id.is_not(None))
        return int((await self.session.execute(statement)).scalar_one())

    async def list_legacy_items_ordered(self) -> list[KnowledgeItem]:
        statement = (
            select(KnowledgeItem)
            .options(selectinload(KnowledgeItem.chunks))
            .where(KnowledgeItem.legacy_id.is_not(None))
            .order_by(KnowledgeItem.legacy_id)
        )
        return list((await self.session.execute(statement)).scalars().unique())
```

- [ ] **Step 4: 验证无事务越权并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_repository.py -q
rg -n "commit\(|rollback\(" app\modules\knowledge\repository.py
git add app/modules/knowledge/repository.py tests/integration/knowledge/test_repository.py
git commit -m "feat: add knowledge repository"
```

Expected: test PASS；`rg` 无匹配；Repository 不关闭外部 session。

---

## Task 4: 实现幂等导入 Service 和整批回滚

**Files:**

- Create: `app/modules/knowledge/service.py`
- Create: `tests/unit/knowledge/test_import_service.py`
- Create: `tests/integration/knowledge/test_import_rollback.py`

- [ ] **Step 1: 写 Service 失败测试**

使用实现相同 Repository 方法签名的 fake，所有异步场景由同步测试调用 `asyncio.run()`：

```python
def test_import_creates_then_skips_identical_bundle() -> None:
    async def exercise() -> None:
        service = make_service()
        first = await service.import_bundles([make_bundle()])
        second = await service.import_bundles([make_bundle()])
        assert (first.created, first.skipped) == (1, 0)
        assert (second.created, second.skipped) == (0, 1)
    asyncio.run(exercise())


def test_import_rejects_duplicate_ids_before_transaction() -> None:
    async def exercise() -> None:
        service, transaction = make_service_with_transaction_probe()
        with pytest.raises(DuplicateLegacyIdError, match="k0001"):
            await service.import_bundles([make_bundle(), make_bundle()])
        assert transaction.entered is False
    asyncio.run(exercise())


def test_conflict_rolls_back_transaction() -> None:
    async def exercise() -> None:
        service, transaction = make_service_with_existing_item()
        with pytest.raises(LegacyKnowledgeConflictError, match="k0001"):
            await service.import_bundles([make_bundle(title="冲突标题")])
        assert transaction.rolled_back is True
    asyncio.run(exercise())
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_import_service.py -q
```

Expected: FAIL，提示 Service 不存在。

- [ ] **Step 3: 实现模型转换和事务算法**

`service.py` 实现三个可独立测试的函数：

```python
def model_from_bundle(bundle: LegacyKnowledgeBundle) -> KnowledgeItem:
    source = bundle.item
    chunk = bundle.chunk
    item = KnowledgeItem(
        legacy_id=source.id,
        category=source.category,
        title=source.title,
        keywords=source.keywords,
        content=source.content,
        example=source.example,
        steps=source.steps,
        difficulty=source.difficulty,
        visibility="public",
        status="indexing",
        revision=1,
    )
    item.chunks.append(
        KnowledgeChunk(
            chunk_index=bundle.chunk_index,
            retrieval_text=chunk.retrieval_text,
            answer_context=chunk.answer_context,
            embedding=None,
            embedding_model=None,
            metadata_={
                **chunk.metadata,
                "legacy_chunk_id": chunk.chunk_id,
                "legacy_source_id": chunk.source_id,
                "source_line": chunk.source_line,
            },
            status="pending",
        )
    )
    return item


def bundle_from_model(item: KnowledgeItem) -> LegacyKnowledgeBundle:
    if item.legacy_id is None:
        raise LegacyKnowledgeImportError("旧知识记录缺少 legacy_id")
    if len(item.chunks) != 1:
        raise LegacyKnowledgeImportError(
            f"legacy_id={item.legacy_id} 的 chunk 数不是 1"
        )
    chunk = item.chunks[0]
    metadata = dict(chunk.metadata_)
    try:
        chunk_id = str(metadata.pop("legacy_chunk_id"))
        source_id = str(metadata.pop("legacy_source_id"))
        source_line = int(metadata.pop("source_line"))
    except (KeyError, TypeError, ValueError) as exc:
        raise LegacyKnowledgeImportError(
            f"legacy_id={item.legacy_id} 的 chunk 元数据不完整"
        ) from exc

    item_input = LegacyKnowledgeItemInput(
        id=item.legacy_id,
        category=item.category,
        title=item.title,
        keywords=item.keywords,
        content=item.content,
        example=item.example,
        steps=item.steps,
        difficulty=item.difficulty,
    )
    chunk_input = LegacyKnowledgeChunkInput(
        chunk_id=chunk_id,
        source_id=source_id,
        category=item.category,
        title=item.title,
        keywords=item.keywords,
        content=item.content,
        example=item.example,
        steps=item.steps,
        difficulty=item.difficulty,
        source_line=source_line,
        retrieval_text=chunk.retrieval_text,
        answer_context=chunk.answer_context,
        metadata=metadata,
    )
    return LegacyKnowledgeBundle(
        item=item_input, chunk=chunk_input, chunk_index=chunk.chunk_index
    )


class LegacyKnowledgeImportService:
    def __init__(self, session: AsyncSession, repository: KnowledgeRepository) -> None:
        self.session = session
        self.repository = repository

    async def import_bundles(
        self, bundles: Sequence[LegacyKnowledgeBundle]
    ) -> LegacyImportSummary:
        ordered = sorted(bundles, key=lambda bundle: bundle.item.id)
        ids = [bundle.item.id for bundle in ordered]
        duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicate_ids:
            raise DuplicateLegacyIdError(
                f"输入批次包含重复 legacy_id: {', '.join(duplicate_ids)}"
            )

        created = 0
        skipped = 0
        async with self.session.begin():
            for bundle in ordered:
                existing = await self.repository.get_by_legacy_id(bundle.item.id)
                if existing is None:
                    self.repository.add(model_from_bundle(bundle))
                    created += 1
                    continue
                if bundle_from_model(existing).sha256() != bundle.sha256():
                    raise LegacyKnowledgeConflictError(
                        f"legacy_id={bundle.item.id} 已存在但内容不同"
                    )
                skipped += 1
            await self.session.flush()
            database_items = await self.repository.count_legacy_items()
            database_chunks = await self.repository.count_legacy_chunks()
            persisted = [
                bundle_from_model(item)
                for item in await self.repository.list_legacy_items_ordered()
            ]

        return LegacyImportSummary(
            input_items=len(ordered), input_chunks=len(ordered),
            created=created, skipped=skipped, conflicts=0, failed=0,
            database_items=database_items, database_chunks=database_chunks,
            input_sha256=collection_sha256(ordered),
            database_sha256=collection_sha256(persisted),
        )
```

重复检测实现时改用 `Counter`，避免上方规格代码中 `ids.count()` 的二次复杂度；行为必须与测试一致。

- [ ] **Step 4: 写真实数据库的部分写入回滚测试**

测试准备数据库中已有 `k0002`。随后同一批先放全新的 `k0001`，再放内容冲突的 `k0002`。断言：

```python
with pytest.raises(LegacyKnowledgeConflictError, match="k0002"):
    asyncio.run(import_new_then_conflicting_existing())
assert asyncio.run(read_legacy_ids()) == ["k0002"]
```

该断言证明冲突前已加入 session 的 `k0001` 没有提交。清理和验证必须使用独立 session。

- [ ] **Step 5: 运行定向测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_import_service.py -q
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_import_rollback.py -q
git add app/modules/knowledge/service.py tests/unit/knowledge/test_import_service.py tests/integration/knowledge/test_import_rollback.py
git commit -m "feat: add idempotent legacy import service"
```

Expected: create、skip、duplicate、conflict 单元测试 PASS；真实数据库回滚测试 PASS。

---

## Task 5: 实现 UTF-8 Loader 和离线导入 CLI

**Files:**

- Create: `app/modules/knowledge/legacy_loader.py`
- Create: `scripts/import_legacy_knowledge.py`
- Create: `tests/unit/knowledge/test_legacy_loader.py`
- Create: `tests/integration/knowledge/test_legacy_import_cli.py`

- [ ] **Step 1: 写 Loader 失败测试**

用 `tmp_path.write_text(..., encoding="utf-8")` 构造最小 JSONL，覆盖：有效配对、空行忽略、坏 JSON 显示文件和行号、重复 raw ID、重复 source ID、集合缺失、跨文件字段不一致。失败断言示例：

```python
with pytest.raises(LegacyKnowledgeInputError, match=r"raw\.jsonl:2"):
    load_legacy_bundles(raw_path, chunk_path)

with pytest.raises(DuplicateLegacyIdError, match="k0001"):
    load_legacy_bundles(duplicate_raw_path, chunk_path)
```

- [ ] **Step 2: 实现 Loader**

```python
from collections import Counter
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LegacyKnowledgeInputError(f"无法按 UTF-8 读取 {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except (ValueError, ValidationError) as exc:
            raise LegacyKnowledgeInputError(
                f"{path}:{line_number} JSONL 校验失败: {exc}"
            ) from exc
    return records


def load_legacy_bundles(raw_path: Path, chunk_path: Path) -> list[LegacyKnowledgeBundle]:
    items = _read_jsonl(raw_path, LegacyKnowledgeItemInput)
    chunks = _read_jsonl(chunk_path, LegacyKnowledgeChunkInput)

    duplicate_item_ids = sorted(
        item_id for item_id, count in Counter(item.id for item in items).items()
        if count > 1
    )
    duplicate_source_ids = sorted(
        source_id
        for source_id, count in Counter(chunk.source_id for chunk in chunks).items()
        if count > 1
    )
    duplicate_chunk_ids = sorted(
        chunk_id
        for chunk_id, count in Counter(chunk.chunk_id for chunk in chunks).items()
        if count > 1
    )
    if duplicate_item_ids:
        raise DuplicateLegacyIdError(
            f"raw 包含重复 id: {', '.join(duplicate_item_ids)}"
        )
    if duplicate_source_ids:
        raise DuplicateLegacyIdError(
            f"processed 包含重复 source_id: {', '.join(duplicate_source_ids)}"
        )
    if duplicate_chunk_ids:
        raise LegacyKnowledgeInputError(
            f"processed 包含重复 chunk_id: {', '.join(duplicate_chunk_ids)}"
        )

    items_by_id = {item.id: item for item in items}
    chunks_by_source = {chunk.source_id: chunk for chunk in chunks}
    missing_chunks = sorted(items_by_id.keys() - chunks_by_source.keys())
    orphan_chunks = sorted(chunks_by_source.keys() - items_by_id.keys())
    if missing_chunks or orphan_chunks:
        raise LegacyKnowledgeInputError(
            "raw/processed 无法一一配对: "
            f"missing_chunks={missing_chunks}, orphan_chunks={orphan_chunks}"
        )

    return [
        LegacyKnowledgeBundle(
            item=items_by_id[legacy_id],
            chunk=chunks_by_source[legacy_id],
            chunk_index=0,
        )
        for legacy_id in sorted(items_by_id)
    ]
```

- [ ] **Step 3: 写 CLI 真实双次导入失败测试**

`test_legacy_import_cli.py` 在已升级的 `mathrag_test` 中清空两表，以 subprocess 执行同一命令两次并解析 stdout 的单行 JSON。断言：

```python
assert first.returncode == second.returncode == 0
assert json.loads(first.stdout)["created"] == 26
assert json.loads(second.stdout)["skipped"] == 26
assert asyncio.run(read_counts(database_url)) == (26, 26, 26)
assert json.loads(first.stdout)["input_sha256"] == json.loads(first.stdout)["database_sha256"]
assert json.loads(second.stdout)["input_sha256"] == json.loads(second.stdout)["database_sha256"]
```

- [ ] **Step 4: 实现 CLI 和稳定退出码**

```python
async def run_import() -> LegacyImportSummary:
    bundles = load_legacy_bundles(settings.RAW_KB_PATH, settings.PROCESSED_KB_PATH)
    async with get_session_factory()() as session:
        repository = KnowledgeRepository(session)
        return await LegacyKnowledgeImportService(session, repository).import_bundles(bundles)


def main() -> int:
    try:
        summary = asyncio.run(run_import())
    except LegacyKnowledgeInputError as exc:
        print(json.dumps({"error": "invalid_input", "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except LegacyKnowledgeConflictError as exc:
        print(json.dumps({"error": "conflict", "detail": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 3
    except Exception as exc:
        print(json.dumps({"error": "database_error", "detail": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(summary.model_dump_json())
    return 0
```

文件末尾使用 `raise SystemExit(main())`。退出码固定为：成功 `0`、数据库/未知异常 `1`、输入无效 `2`、内容冲突 `3`。未知异常只输出异常类型，不打印连接串或知识正文。

- [ ] **Step 5: 运行 Loader 与真实双次导入测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_legacy_loader.py -q
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_legacy_import_cli.py -q
git add app/modules/knowledge/legacy_loader.py scripts/import_legacy_knowledge.py tests/unit/knowledge/test_legacy_loader.py tests/integration/knowledge/test_legacy_import_cli.py
git commit -m "feat: import legacy knowledge into postgres"
```

Expected: Loader 负例全部 PASS；真实 CLI 首次创建 26 条，第二次跳过 26 条，总数不增长。

---

## Task 6: 完成 M2 验收、回归和证据记录

**Files:**

- Create: `docs/baselines/2026-07-30-m2-legacy-knowledge-migration.md`
- Verify unchanged: `app/api/chat.py`
- Verify unchanged: `app/services/retriever.py`
- Verify unchanged: `data/raw/math_knowledge_seed.jsonl`
- Verify unchanged: `data/processed/kb_chunks.jsonl`

- [ ] **Step 1: 校验输入文件未被修改**

```powershell
Get-FileHash data\raw\math_knowledge_seed.jsonl -Algorithm SHA256
Get-FileHash data\processed\kb_chunks.jsonl -Algorithm SHA256
```

Expected: 分别等于验收常量中的两个 SHA-256。

- [ ] **Step 2: 在主库执行可重复导入并核对字段**

```powershell
$env:DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
.\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge
```

使用只读 SQL 核对：item 数 26、chunk 数 26、非空唯一 legacy ID 26、空 embedding 26、item 全部 `indexing`、chunk 全部 `pending`。另外由 `load_legacy_bundles()` 与 `bundle_from_model()` 比较 26 个 bundle 的 `persistent_payload()`，不得只比较行数。

- [ ] **Step 3: 跑完整回归和固定检索基线**

```powershell
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\test_retrieval_baseline.py -q
git diff main -- app/api/chat.py app/services/retriever.py data/raw/math_knowledge_seed.jsonl data/processed/kb_chunks.jsonl
```

Expected: 全量测试 PASS；固定题集 26/26；最后一条命令无差异，证明在线链路和源数据未改动。

- [ ] **Step 4: 记录验收证据**

在 baseline 文档记录：提交 SHA、migration current、两次 CLI 摘要、表计数、集合摘要、输入文件 SHA、回滚测试名、全量测试结果、FAISS 26/26，以及回滚命令：

```powershell
$env:DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
.\.venv\Scripts\python.exe -m alembic -c alembic.ini downgrade 0001_enable_vector_extension
```

注明该回滚会删除两张 M2 表，执行前必须备份业务数据；它不会删除 vector extension 或旧 JSONL/FAISS 文件。

- [ ] **Step 5: 最终提交**

```powershell
git add docs/baselines/2026-07-30-m2-legacy-knowledge-migration.md
git commit -m "docs: record M2 migration acceptance"
git status --short
```

Expected: 工作区干净。M2 完成条件是 migration、双次导入、冲突回滚、字段级一致性、全量测试和旧 FAISS 回归全部通过，而不是仅以“脚本运行成功”为准。
