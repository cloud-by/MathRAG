# MathRAG M3 pgvector 检索与在线切换 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 M2 导入的 26 条知识 chunk 生成 1024 维向量，以 PostgreSQL + pgvector 精确余弦检索替代在线 FAISS，并通过固定 26 题证明召回质量、权限过滤、模型隔离与数据库检索性能满足门槛。

**Architecture:** `OpenAIEmbeddingProvider` 负责异步批量向量化和严格向量校验；`KnowledgeRepository` 只执行带状态、可见性和模型过滤的 pgvector SQL；`KnowledgeSearchService` 在数据库事务外调用 Embedding Provider，在短生命周期 `AsyncSession` 中执行精确检索并合并多查询结果。离线 `KnowledgeReindexService` 先读取候选并关闭会话，再调用外部 API，最后用短事务写回向量和 `ready` 状态。RAG 管道只依赖 Knowledge Service 的公开接口，在线切换验收后删除生产导入图中的 FAISS、`id_map.json` 和 processed JSONL 依赖。

**Tech Stack:** Python 3.11.9、FastAPI 0.140.13、OpenAI Python 2.50.0 `AsyncOpenAI`、SQLAlchemy 2.0.51 asyncio、asyncpg 0.31.0、Alembic 1.18.5、PostgreSQL 18.4、pgvector Server 0.8.5 / pgvector Python 0.5.0、Pydantic 2.13.4、pytest 9.1.1、Docker Compose。

---

## 起点与前置条件

- M3 分支：`codex/m3-pgvector-retrieval-cutover`。
- 分支基点：`main@cd77635`；该提交已包含 M2 全部代码及跨平台 UTF-8/LF 内容哈希修复。
- 合并后全量基线：`144 passed, 1 warning`；唯一警告是既有 Starlette/httpx 弃用提示。
- M2 数据契约：26 个 `knowledge_items`、26 个 `knowledge_chunks`；item 初始为 `indexing`，chunk 初始为 `pending`，`embedding` 与 `embedding_model` 为空。
- 执行 M3 前先对目标数据库运行 `alembic upgrade head` 和 `python -m scripts.import_legacy_knowledge`。若使用新的 Compose volume，不能假定 M2 数据已存在。
- 真实重建向量和质量对账必须配置 Embedding Provider；没有真实凭据时只能完成单元/集成测试，不能宣称 M3 验收完成。
- 所有中文源码、测试、JSON 和文档保持 UTF-8；修改已有文件时保持原换行风格。

## 冻结设计决策

- 在线相似度固定使用 pgvector 余弦距离，公开分数固定为 `1.0 - cosine_distance`，分数越大越相关。
- 每个向量必须恰好 1024 维、全部为有限数值、L2 范数大于 0；Provider 输出顺序必须与输入顺序一致。
- 查询只允许返回同时满足以下条件的行：item `status=ready`、item `visibility=public`、chunk `status=ready`、embedding 非空、`embedding_model` 与当前 Provider 模型完全相同。
- M4 尚未引入用户和 owner，因此 M3 不提供“任意 private 可检索”的参数；private 数据一律不进入在线结果，权限泄漏必须为 0。
- 多查询最多 4 条；Embedding Provider 一次批量生成查询向量；同一 `AsyncSession` 内顺序执行 SQL，不并发使用 Session。
- 多查询结果按数据库 chunk UUID 去重，保留最高分；同分时按 UUID 字符串升序，最终重排并截取 Top-K。
- RAG 响应字段保持兼容；`ReferenceItem.index` 暂保留但 pgvector 路径固定为 `None`，不得伪造 FAISS 序号。
- 不在等待 LLM/Embedding 网络调用时持有数据库事务或连接；Repository 不调用 `commit()`、`rollback()` 或创建 Session。
- 26 条小数据使用精确扫描；M3 不创建 HNSW/IVFFlat。只有实测数据量和 P95 证明需要时，才另立 ADR 和 migration。
- 重建向量可重入：相同模型且已经 ready 的 chunk 跳过；模型变化、空向量、pending 或 failed 的 chunk 重新生成。
- 在线切换是单向收口：验收通过后不保留长期双检索开关。FAISS 只存在于显式 evaluation 工具和只读历史工件中。

## 量化验收门槛

| 指标 | 门槛 |
|---|---:|
| 固定问题集 | 复用 `tests/fixtures/retrieval_questions.json` 的 26 题，`top_k=3` |
| pgvector 期望知识 Top-3 命中 | 至少 24/26，即不低于 90% |
| pgvector 与 FAISS Top-3 集合平均重合率 | 不低于 0.80 |
| 本地精确 SQL 检索 P95 | 不高于 100 ms；不包含 Embedding API 时间 |
| public/private 泄漏 | 0 |
| 非 ready chunk 泄漏 | 0 |
| 不同 embedding_model 混用 | 0 |
| 全量测试 | 0 failed、0 unexpected skipped；只允许已记录的既有 warning |

## 官方实现依据

- pgvector Python 的 SQLAlchemy 官方示例使用 `VECTOR(n)` 映射列，并通过 `Item.embedding.cosine_distance(query_vector)` 排序：[pgvector-python SQLAlchemy](https://github.com/pgvector/pgvector-python#sqlalchemy)。
- pgvector 官方定义余弦相似度为 `1 - cosine distance`，并要求向量元素为有限数值：[pgvector distance functions](https://github.com/pgvector/pgvector#distances)。
- OpenAI Python 官方库要求异步调用实例化 `AsyncOpenAI`，异步客户端与同步客户端使用相同资源接口：[openai-python async usage](https://github.com/openai/openai-python#async-usage)。

## 文件结构

| 文件 | 操作 | 单一职责 |
|---|---|---|
| `app/infrastructure/embedding/__init__.py` | Create | 导出 Embedding Provider 公共接口 |
| `app/infrastructure/embedding/provider.py` | Create | 异步 OpenAI-compatible Provider、批处理、清洗和向量校验 |
| `app/modules/knowledge/errors.py` | Modify | 增加安全的向量化与检索异常层级 |
| `app/modules/knowledge/models.py` | Modify | 注册 ready/embedding 一致性约束与普通检索索引 |
| `app/modules/knowledge/search.py` | Create | 检索 DTO、引用映射、多查询去重与排序 |
| `app/modules/knowledge/repository.py` | Modify | 候选读取、精确余弦查询、向量状态写回；不管理事务 |
| `app/modules/knowledge/search_service.py` | Create | 批量查询向量化、短会话检索、公开接口 |
| `app/modules/knowledge/reindex_service.py` | Create | 事务外批量向量化、短事务写回、可重入状态机 |
| `alembic/versions/0003_enforce_vector_readiness.py` | Create | ready 数据约束及普通过滤索引，不创建向量近似索引 |
| `scripts/reindex_knowledge.py` | Create | 可审计重建 CLI 与稳定退出码 |
| `scripts/evaluate_pgvector_retrieval.py` | Create | 真实 Provider 下 FAISS/pgvector 对账和 P95 artifact |
| `scripts/legacy_faiss_retriever.py` | Create | 仅 evaluation 使用的只读 FAISS 适配器 |
| `scripts/capture_retrieval_baseline.py` | Modify | 改用只读 legacy adapter，不依赖在线 app service |
| `scripts/demo_query.py` | Modify | 改为调用 pgvector Knowledge Search Service |
| `app/services/rag_pipeline.py` | Modify | 异步化，并仅调用 Knowledge Search Service |
| `app/api/chat.py` | Modify | 异步路由及数据库/Embedding 安全错误映射 |
| `app/schemas/chat.py` | Modify | 保持响应兼容，更新 `index` 的兼容语义 |
| `app/main.py` | Modify | 描述从 FAISS 更新为 PostgreSQL + pgvector |
| `app/services/retriever.py` | Delete | 删除在线 FAISS Retriever 和全局索引单例 |
| `app/services/vector_store.py` | Delete | 删除生产 app 包中的 FAISS 实现 |
| `app/services/embedding_service.py` | Delete | 删除同步全局 Embedding 单例 |
| `scripts/build_index.py` | Delete | 禁止继续生成第二套在线事实来源 |
| `requirements.txt` / `requirements.lock.txt` | Modify | 生产依赖移除 `faiss-cpu` |
| `requirements-evaluation.txt` / `.lock.txt` | Create | 仅历史对账工具安装 FAISS |
| `README.md` | Modify | 更新开发、重建、检索与回滚命令 |
| `tests/unit/infrastructure/test_embedding_provider.py` | Create | Provider 输入、维度、有限值、顺序和错误测试 |
| `tests/unit/knowledge/test_search.py` | Create | 分数、去重、稳定排序和引用契约测试 |
| `tests/unit/knowledge/test_search_service.py` | Create | 外部调用先于 Session、批量查询和 Session 关闭测试 |
| `tests/unit/knowledge/test_reindex_service.py` | Create | 批次、可重入、失败状态和事务边界测试 |
| `tests/integration/knowledge/test_vector_search.py` | Create | 真实 pgvector 排序及过滤测试 |
| `tests/integration/knowledge/test_reindex.py` | Create | 真实 PostgreSQL 状态写回与回滚测试 |
| `tests/evaluation/test_faiss_pgvector_parity.py` | Create | 指标计算与安全 artifact 契约测试 |
| `tests/test_agentic_rag.py` | Modify | 异步多查询检索和最佳分数合并 |
| `tests/test_chat_api.py` | Modify | 异步管道 mock、无文件依赖、错误不泄密 |
| `tests/test_runtime_dependency_boundary.py` | Create | 在线 import graph、Docker lock 和路径引用防回归 |
| `docs/baselines/2026-07-30-m3-pgvector-retrieval-cutover.md` | Create | M3 现场证据、门槛和回滚说明 |

---

## Task 1: 建立异步 Embedding Provider 与严格向量契约

**Files:**

- Create: `app/infrastructure/embedding/__init__.py`
- Create: `app/infrastructure/embedding/provider.py`
- Modify: `app/modules/knowledge/errors.py`
- Create: `tests/unit/infrastructure/__init__.py`
- Create: `tests/unit/infrastructure/test_embedding_provider.py`

- [ ] **Step 1: 写失败测试**

测试使用假异步客户端，不访问网络，并固定空输入、输出顺序、维度、有限值和归一化：

```python
def test_provider_sorts_response_by_index_and_normalizes_vectors() -> None:
    client = FakeAsyncOpenAI(
        data=[FakeEmbedding(index=1, embedding=vector(4.0)), FakeEmbedding(index=0, embedding=vector(3.0))]
    )
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test", dimensions=1024)

    result = asyncio.run(provider.embed_texts(["第一个问题", "第二个问题"]))

    assert len(result) == 2
    assert result[0][0] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(1.0)
    assert client.inputs == [["第一个问题", "第二个问题"]]


@pytest.mark.parametrize(
    "bad_vector",
    [
        [0.0] * 1023,
        [0.0] * 1024,
        [float("nan")] + [0.0] * 1023,
        [float("inf")] + [0.0] * 1023,
    ],
)
def test_provider_rejects_invalid_vectors_without_echoing_input(bad_vector: list[float]) -> None:
    provider = OpenAIEmbeddingProvider(
        client=FakeAsyncOpenAI(data=[FakeEmbedding(index=0, embedding=bad_vector)]),
        model="embedding-test",
        dimensions=1024,
    )

    with pytest.raises(EmbeddingResponseError) as error:
        asyncio.run(provider.embed_texts(["不要出现在错误中的知识正文"]))

    assert "不要出现在错误中的知识正文" not in str(error.value)
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\infrastructure\test_embedding_provider.py -q
```

Expected: FAIL，提示 `app.infrastructure.embedding.provider` 不存在。

- [ ] **Step 3: 增加稳定异常层级**

在 `app/modules/knowledge/errors.py` 增加：

```python
class KnowledgeSearchError(Exception):
    """知识向量化或检索失败。"""


class EmbeddingInputError(KnowledgeSearchError, ValueError):
    """待向量化文本或配置不满足固定契约。"""


class EmbeddingResponseError(KnowledgeSearchError):
    """Embedding Provider 返回无效结果。"""


class EmbeddingUnavailableError(KnowledgeSearchError):
    """Embedding Provider 暂时不可用。"""
```

- [ ] **Step 4: 实现 Provider 协议、清洗、批处理和校验**

`provider.py` 的公共接口固定为：

```python
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.core.config import settings
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)


class EmbeddingProvider(Protocol):
    @property
    def model(self) -> str:
        raise NotImplementedError

    @property
    def dimensions(self) -> int:
        raise NotImplementedError

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        raise NotImplementedError


def validate_and_normalize_vector(values: Sequence[float], dimensions: int) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != dimensions:
        raise EmbeddingResponseError("Embedding 维度与配置不一致")
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingResponseError("Embedding 包含非有限数值")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        raise EmbeddingResponseError("Embedding 不能是零向量")
    return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = settings.EMBEDDING_MODEL,
        dimensions: int = settings.EMBEDDING_DIMENSIONS,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ) -> None:
        if not model.strip():
            raise EmbeddingInputError("EMBEDDING_MODEL 不能为空")
        if dimensions != 1024:
            raise EmbeddingInputError("EMBEDDING_DIMENSIONS 必须为 1024")
        if batch_size <= 0:
            raise EmbeddingInputError("EMBEDDING_BATCH_SIZE 必须大于 0")
        self._model = model.strip()
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = client or AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.EMBEDDING_TIMEOUT,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [" ".join(str(text).split()).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise EmbeddingInputError("Embedding 输入必须是非空文本数组")

        output: list[list[float]] = []
        for offset in range(0, len(cleaned), self._batch_size):
            batch = cleaned[offset : offset + self._batch_size]
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimensions,
                    encoding_format="float",
                )
            except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as exc:
                raise EmbeddingUnavailableError(type(exc).__name__) from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise EmbeddingResponseError("Embedding 返回索引不连续")
            output.extend(
                validate_and_normalize_vector(item.embedding, self._dimensions)
                for item in ordered
            )
        if len(output) != len(cleaned):
            raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
        return output

    async def aclose(self) -> None:
        await self._client.close()


_embedding_provider: OpenAIEmbeddingProvider | None = None


def get_embedding_provider() -> OpenAIEmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = OpenAIEmbeddingProvider()
    return _embedding_provider


async def dispose_embedding_provider() -> None:
    global _embedding_provider
    if _embedding_provider is not None:
        await _embedding_provider.aclose()
    _embedding_provider = None
```

在线请求复用一个仅持有 SDK 连接池的 Provider；它不保存会话、索引或查询结果。`app.main.lifespan()` 必须调用 `dispose_embedding_provider()`。错误不得包含 URL、密钥、原始响应或输入正文。

- [ ] **Step 5: 运行 Task 1 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\infrastructure\test_embedding_provider.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add app/infrastructure/embedding app/modules/knowledge/errors.py tests/unit/infrastructure
git commit -m "feat: add async embedding provider contract"
```

---

## Task 2: 增加 0003 向量就绪约束与普通过滤索引

**Files:**

- Modify: `app/modules/knowledge/models.py`
- Create: `alembic/versions/0003_enforce_vector_readiness.py`
- Modify: `tests/unit/knowledge/test_models.py`
- Modify: `tests/integration/knowledge/test_migration_schema.py`

- [ ] **Step 1: 写失败测试**

```python
def test_ready_chunk_requires_embedding_and_model() -> None:
    constraint_sql = {
        str(constraint.sqltext)
        for constraint in KnowledgeChunk.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)" in constraint_sql


async def exercise_ready_constraint(database_url: str) -> None:
    connection = await asyncpg.connect(database_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        item_id = await insert_item(connection, status="ready", visibility="public")
        with pytest.raises(asyncpg.CheckViolationError):
            await insert_chunk(
                connection,
                item_id=item_id,
                status="ready",
                embedding=None,
                embedding_model=None,
            )
    finally:
        await connection.close()


def test_ready_constraint_rejects_missing_vector() -> None:
    database_url = require_test_database_url(
        os.environ["TEST_DATABASE_URL"],
        os.getenv("DATABASE_URL"),
    )
    asyncio.run(exercise_ready_constraint(database_url))
```

同时断言 migration 只创建 B-tree/普通索引，不包含 `hnsw` 或 `ivfflat`。

- [ ] **Step 2: 确认测试先失败**

```powershell
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:***@localhost:5432/mathrag_test'
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_models.py tests\integration\knowledge\test_migration_schema.py -q
```

Expected: FAIL，缺少 ready 约束和 0003 migration。

- [ ] **Step 3: 修改 ORM 约束**

在 `KnowledgeChunk.__table_args__` 增加：

```python
CheckConstraint(
    "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)",
    name="ready_requires_embedding",
),
```

在 ORM 中为检索过滤注册两个普通复合索引：

```python
Index("ix_knowledge_items_visibility_status", "visibility", "status")
Index("ix_knowledge_chunks_status_embedding_model", "status", "embedding_model")
```

- [ ] **Step 4: 创建自包含 0003 migration**

```python
revision = "0003_enforce_vector_readiness"
down_revision = "0002_create_knowledge_tables"


def upgrade() -> None:
    op.create_check_constraint(
        "ck_knowledge_chunks_ready_requires_embedding",
        "knowledge_chunks",
        "status != 'ready' OR (embedding IS NOT NULL AND embedding_model IS NOT NULL)",
    )
    op.create_index(
        "ix_knowledge_items_visibility_status",
        "knowledge_items",
        ["visibility", "status"],
        unique=False,
    )
    op.create_index(
        "ix_knowledge_chunks_status_embedding_model",
        "knowledge_chunks",
        ["status", "embedding_model"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_knowledge_chunks_status_embedding_model", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_items_visibility_status", table_name="knowledge_items")
    op.drop_constraint(
        "ck_knowledge_chunks_ready_requires_embedding",
        "knowledge_chunks",
        type_="check",
    )
```

Migration 不得 import `app.*`，并在任何 downgrade 前继续调用 `require_test_database_url()`。

- [ ] **Step 5: 运行迁移往返与漂移检查**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_models.py tests\integration\knowledge\test_migration_schema.py tests\integration\test_migrations.py -q
$env:DATABASE_URL='postgresql+asyncpg://mathrag:***@localhost:5432/mathrag_test'
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic check
```

Expected: PASS；current 为 `0003_enforce_vector_readiness (head)`；`No new upgrade operations detected.`。

- [ ] **Step 6: 提交**

```powershell
git add app/modules/knowledge/models.py alembic/versions/0003_enforce_vector_readiness.py tests/unit/knowledge/test_models.py tests/integration/knowledge/test_migration_schema.py
git commit -m "feat: enforce knowledge vector readiness"
```

---

## Task 3: 实现精确 pgvector Repository 与稳定检索 DTO

**Files:**

- Create: `app/modules/knowledge/search.py`
- Modify: `app/modules/knowledge/repository.py`
- Create: `tests/unit/knowledge/test_search.py`
- Create: `tests/integration/knowledge/test_vector_search.py`

- [ ] **Step 1: 写纯函数失败测试**

```python
def test_merge_hits_deduplicates_by_database_chunk_and_keeps_best_score() -> None:
    low = make_hit(database_chunk_id=CHUNK_A, score=0.61)
    high = make_hit(database_chunk_id=CHUNK_A, score=0.88)
    other = make_hit(database_chunk_id=CHUNK_B, score=0.70)

    merged = merge_search_hits([[low, other], [high]], top_k=2)

    assert [(hit.database_chunk_id, hit.score) for hit in merged] == [
        (CHUNK_A, 0.88),
        (CHUNK_B, 0.70),
    ]


def test_hit_maps_to_backward_compatible_reference() -> None:
    reference = make_hit(score=0.75).to_reference(rank=1)
    assert reference["rank"] == 1
    assert reference["score"] == pytest.approx(0.75)
    assert reference["index"] is None
    assert reference["chunk_id"] == "k0001_chunk_0"
    assert reference["source_id"] == "k0001"
```

- [ ] **Step 2: 写真实数据库失败测试**

使用固定 1024 维单位向量插入：公开 ready A、公开 ready B、private ready、pending、不同模型 ready。查询 `[1.0] + [0.0] * 1023` 时只允许返回 A/B，并按余弦距离升序：

```python
def test_exact_vector_search_orders_and_filters_rows() -> None:
    rows = asyncio.run(exercise_vector_search(TEST_DATABASE_URL))
    assert [row.legacy_source_id for row in rows] == ["k0001", "k0002"]
    assert rows[0].distance == pytest.approx(0.0)
    assert rows[1].distance > rows[0].distance
```

- [ ] **Step 3: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_search.py tests\integration\knowledge\test_vector_search.py -q
```

Expected: FAIL，缺少 DTO、merge 和 repository 查询。

- [ ] **Step 4: 实现 DTO 与引用映射**

`search.py` 定义冻结 dataclass；`metadata` 在构造时深复制，内部审计键不重复暴露：

```python
@dataclass(frozen=True)
class KnowledgeSearchHit:
    database_chunk_id: UUID
    legacy_chunk_id: str
    legacy_source_id: str
    category: str
    title: str
    keywords: tuple[str, ...]
    content: str
    example: str
    steps: tuple[str, ...]
    difficulty: str
    answer_context: str
    retrieval_text: str
    source_line: int | None
    metadata: dict[str, object]
    distance: float

    @property
    def score(self) -> float:
        return 1.0 - self.distance

    def to_reference(self, *, rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "score": self.score,
            "index": None,
            "chunk_id": self.legacy_chunk_id,
            "source_id": self.legacy_source_id,
            "category": self.category,
            "title": self.title,
            "keywords": list(self.keywords),
            "content": self.content,
            "example": self.example,
            "steps": list(self.steps),
            "difficulty": self.difficulty,
            "answer_context": self.answer_context,
            "retrieval_text": self.retrieval_text,
            "source_line": self.source_line,
            "metadata": deepcopy(self.metadata),
        }


def merge_search_hits(groups: Sequence[Sequence[KnowledgeSearchHit]], top_k: int) -> list[KnowledgeSearchHit]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    best: dict[UUID, KnowledgeSearchHit] = {}
    for group in groups:
        for hit in group:
            current = best.get(hit.database_chunk_id)
            if current is None or hit.score > current.score:
                best[hit.database_chunk_id] = hit
    return sorted(
        best.values(),
        key=lambda hit: (-hit.score, str(hit.database_chunk_id)),
    )[:top_k]
```

- [ ] **Step 5: 实现 Repository 精确余弦查询**

```python
async def search_ready_chunks(
    self,
    *,
    query_vector: Sequence[float],
    embedding_model: str,
    limit: int,
) -> list[KnowledgeSearchHit]:
    if len(query_vector) != 1024:
        raise ValueError("query_vector 必须为 1024 维")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    distance = KnowledgeChunk.embedding.cosine_distance(list(query_vector)).label("distance")
    statement = (
        select(KnowledgeChunk, KnowledgeItem, distance)
        .join(KnowledgeItem, KnowledgeItem.id == KnowledgeChunk.knowledge_item_id)
        .where(
            KnowledgeItem.status == "ready",
            KnowledgeItem.visibility == "public",
            KnowledgeChunk.status == "ready",
            KnowledgeChunk.embedding.is_not(None),
            KnowledgeChunk.embedding_model == embedding_model,
        )
        .order_by(distance.asc(), KnowledgeChunk.id.asc())
        .limit(limit)
    )
    rows = (await self._session.execute(statement)).all()
    return [search_hit_from_row(chunk, item, float(row_distance)) for chunk, item, row_distance in rows]
```

`search_hit_from_row()` 必须从 metadata 提取并移除 `legacy_chunk_id`、`legacy_source_id`、`source_line`；类型不合法时抛 `KnowledgeSearchError`，错误只包含数据库 UUID，不包含知识正文。

- [ ] **Step 6: 验证 SQL 与过滤**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_search.py tests\integration\knowledge\test_vector_search.py -q
```

Expected: PASS；private、pending、错误模型均未出现在结果中。

- [ ] **Step 7: 静态确认 Repository 不持有事务**

```powershell
rg -n "commit\(|rollback\(|sessionmaker|create_async_engine" app/modules/knowledge/repository.py
```

Expected: 无匹配。

- [ ] **Step 8: 提交**

```powershell
git add app/modules/knowledge/search.py app/modules/knowledge/repository.py tests/unit/knowledge/test_search.py tests/integration/knowledge/test_vector_search.py
git commit -m "feat: add exact pgvector knowledge search"
```

---

## Task 4: 建立 Knowledge Search Service 与短会话边界

**Files:**

- Create: `app/modules/knowledge/search_service.py`
- Create: `tests/unit/knowledge/test_search_service.py`

- [ ] **Step 1: 写失败测试**

测试必须证明 Provider 调用时尚未打开 Session，所有 SQL 完成后 Session 已关闭，并且四条查询只调用一次批量 Embedding：

```python
def test_search_embeds_before_opening_session_and_batches_queries() -> None:
    events: list[str] = []
    provider = FakeProvider(events=events, vectors=[unit(0), unit(1)])
    session_factory = FakeSessionFactory(events=events)
    service = KnowledgeSearchService(session_factory=session_factory, provider=provider)

    hits = asyncio.run(service.search(["二次函数", "抛物线"], top_k=3))

    assert provider.calls == [["二次函数", "抛物线"]]
    assert events == ["embedding:start", "embedding:end", "session:open", "sql", "sql", "session:close"]
    assert len(hits) <= 3


def test_search_rejects_more_than_four_distinct_queries() -> None:
    service = make_service()
    with pytest.raises(ValueError, match="最多 4 条"):
        asyncio.run(service.search(["q1", "q2", "q3", "q4", "q5"], top_k=3))
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_search_service.py -q
```

Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 Service**

```python
class KnowledgeSearchService:
    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        provider: EmbeddingProvider,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider

    async def search(self, queries: Sequence[str], *, top_k: int) -> list[KnowledgeSearchHit]:
        normalized: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = " ".join(str(query).split()).strip()
            if text and text not in seen:
                normalized.append(text)
                seen.add(text)
        if not normalized:
            raise ValueError("检索查询不能为空")
        if len(normalized) > 4:
            raise ValueError("检索查询最多 4 条")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")

        vectors = await self._provider.embed_texts(normalized)
        groups: list[list[KnowledgeSearchHit]] = []
        async with self._session_factory() as session:
            repository = KnowledgeRepository(session)
            for vector in vectors:
                groups.append(
                    await repository.search_ready_chunks(
                        query_vector=vector,
                        embedding_model=self._provider.model,
                        limit=top_k,
                    )
                )
        return merge_search_hits(groups, top_k)
```

提供工厂函数 `build_knowledge_search_service()`，仅组合 `get_session_factory()` 与 `get_embedding_provider()`；测试和 RAGPipeline 都可以注入替身，不使用全局 Session 或索引状态。

- [ ] **Step 4: 验证 Service**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_search_service.py tests\unit\knowledge\test_search.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/modules/knowledge/search_service.py tests/unit/knowledge/test_search_service.py
git commit -m "feat: add knowledge vector search service"
```

---

## Task 5: 实现事务外批量向量化与可重入 reindex CLI

**Files:**

- Create: `app/modules/knowledge/reindex_service.py`
- Modify: `app/modules/knowledge/repository.py`
- Modify: `app/modules/knowledge/search.py`
- Create: `scripts/reindex_knowledge.py`
- Create: `tests/unit/knowledge/test_reindex_service.py`
- Create: `tests/integration/knowledge/test_reindex.py`
- Create: `tests/integration/knowledge/test_reindex_cli.py`

- [ ] **Step 1: 写状态机和事务边界失败测试**

```python
def test_reindex_closes_read_session_before_embedding_and_writes_short_batches() -> None:
    events: list[str] = []
    service = make_reindex_service(events=events, batch_size=2, candidate_count=3)

    summary = asyncio.run(service.reindex())

    assert events == [
        "read:open", "read:candidates", "read:close",
        "embedding:2", "write:open", "write:2", "write:close",
        "embedding:1", "write:open", "write:1", "write:close",
    ]
    assert summary.selected == 3
    assert summary.ready == 3
    assert summary.failed == 0


def test_reindex_marks_batch_failed_without_leaking_text() -> None:
    service = make_reindex_service(provider_error=RuntimeError("正文和密钥不应传播"))
    with pytest.raises(EmbeddingUnavailableError) as error:
        asyncio.run(service.reindex())
    assert "正文和密钥" not in str(error.value)
    assert service.failed_ids == service.selected_ids
```

- [ ] **Step 2: 写真实数据库失败测试**

覆盖以下场景：

1. pending 26 条全部写入 1024 维向量，chunk/item 均 ready；
2. 第二次同模型运行全部 skipped，不再次调用 Provider；
3. `embedding_model` 变化后 26 条重新选中；
4. Provider 失败只将当前 batch 标为 failed，已提交 batch 保持 ready；
5. ready 约束阻止空向量；
6. 每次测试前调用 `require_test_database_url()`。

- [ ] **Step 3: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_reindex_service.py tests\integration\knowledge\test_reindex.py -q
```

Expected: FAIL，缺少候选、状态写回和 reindex service。

- [ ] **Step 4: 增加 Repository 原语**

先在 `search.py` 增加不可变的写入快照，避免 Repository 反向依赖 Service：

```python
@dataclass(frozen=True)
class ReindexCandidate:
    chunk_id: UUID
    item_id: UUID
    retrieval_text: str


@dataclass(frozen=True)
class EmbeddingUpdate:
    chunk_id: UUID
    item_id: UUID
    expected_retrieval_text: str
    vector: tuple[float, ...]
```

Repository 增加以下完整原语，但不提交事务：

```python
async def list_reindex_candidates(self, *, embedding_model: str) -> list[ReindexCandidate]:
    statement = (
        select(
            KnowledgeChunk.id,
            KnowledgeChunk.knowledge_item_id,
            KnowledgeChunk.retrieval_text,
        )
        .where(
            or_(
                KnowledgeChunk.embedding.is_(None),
                KnowledgeChunk.status.in_(("pending", "failed")),
                KnowledgeChunk.embedding_model.is_distinct_from(embedding_model),
            )
        )
        .order_by(KnowledgeChunk.id.asc())
    )
    rows = (await self._session.execute(statement)).all()
    return [
        ReindexCandidate(
            chunk_id=chunk_id,
            item_id=item_id,
            retrieval_text=retrieval_text,
        )
        for chunk_id, item_id, retrieval_text in rows
    ]


async def mark_candidates_indexing(self, candidates: Sequence[ReindexCandidate]) -> int:
    chunk_ids = [candidate.chunk_id for candidate in candidates]
    item_ids = sorted({candidate.item_id for candidate in candidates}, key=str)
    chunk_result = await self._session.execute(
        update(KnowledgeChunk)
        .where(KnowledgeChunk.id.in_(chunk_ids))
        .values(status="pending")
    )
    await self._session.execute(
        update(KnowledgeItem)
        .where(KnowledgeItem.id.in_(item_ids))
        .values(status="indexing")
    )
    return int(chunk_result.rowcount or 0)


async def write_ready_embeddings(
    self,
    updates: Sequence[EmbeddingUpdate],
    *,
    embedding_model: str,
) -> int:
    written = 0
    for embedding_update in updates:
        result = await self._session.execute(
            update(KnowledgeChunk)
            .where(
                KnowledgeChunk.id == embedding_update.chunk_id,
                KnowledgeChunk.retrieval_text == embedding_update.expected_retrieval_text,
            )
            .values(
                embedding=list(embedding_update.vector),
                embedding_model=embedding_model,
                status="ready",
            )
        )
        written += int(result.rowcount or 0)
    if written != len(updates):
        raise KnowledgeSearchError("向量写回期间知识 chunk 已发生变化")
    return written


async def mark_chunks_failed(self, candidates: Sequence[ReindexCandidate]) -> int:
    chunk_ids = [candidate.chunk_id for candidate in candidates]
    item_ids = sorted({candidate.item_id for candidate in candidates}, key=str)
    result = await self._session.execute(
        update(KnowledgeChunk)
        .where(KnowledgeChunk.id.in_(chunk_ids))
        .values(embedding=None, embedding_model=None, status="failed")
    )
    await self._session.execute(
        update(KnowledgeItem)
        .where(KnowledgeItem.id.in_(item_ids))
        .values(status="failed")
    )
    return int(result.rowcount or 0)


async def refresh_item_statuses(self, item_ids: Sequence[UUID]) -> None:
    not_ready = exists(
        select(KnowledgeChunk.id).where(
            KnowledgeChunk.knowledge_item_id == KnowledgeItem.id,
            or_(
                KnowledgeChunk.status != "ready",
                KnowledgeChunk.embedding.is_(None),
                KnowledgeChunk.embedding_model.is_(None),
            ),
        )
    )
    await self._session.execute(
        update(KnowledgeItem)
        .where(KnowledgeItem.id.in_(list(item_ids)), ~not_ready)
        .values(status="ready")
    )
```

`list_reindex_candidates()` 的条件固定为：embedding 为空、status 为 pending/failed、或 `embedding_model != 当前模型`。`write_ready_embeddings()` 必须校验影响行数等于输入数；少写一行即抛 `KnowledgeSearchError` 并回滚当前 batch。

- [ ] **Step 5: 实现 Reindex Service**

```python
@dataclass(frozen=True)
class ReindexSummary:
    selected: int
    ready: int
    skipped: int
    failed: int
    embedding_model: str
    dimensions: int


class KnowledgeReindexService:
    async def reindex(self) -> ReindexSummary:
        candidates = await self._load_and_mark_candidates()
        if not candidates:
            return ReindexSummary(0, 0, await self._count_ready(), 0, self._provider.model, 1024)

        ready = 0
        for batch in chunked(candidates, self._batch_size):
            try:
                vectors = await self._provider.embed_texts(
                    [candidate.retrieval_text for candidate in batch]
                )
            except Exception as exc:
                await self._mark_failed(batch)
                raise EmbeddingUnavailableError(type(exc).__name__) from exc
            await self._write_ready(batch, vectors)
            ready += len(batch)

        return ReindexSummary(
            selected=len(candidates),
            ready=ready,
            skipped=await self._count_ready() - ready,
            failed=0,
            embedding_model=self._provider.model,
            dimensions=self._provider.dimensions,
        )


def chunked(
    items: Sequence[ReindexCandidate],
    batch_size: int,
) -> list[list[ReindexCandidate]]:
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]
```

`_load_and_mark_candidates()`、`_write_ready()`、`_mark_failed()` 各自创建独立 Session，并使用 `async with session.begin()`；Provider 调用发生在这些方法之间，不得持有 Session。

- [ ] **Step 6: 实现稳定 CLI**

`scripts/reindex_knowledge.py` 只输出单行 JSON：

```python
def main() -> int:
    try:
        summary = asyncio.run(run_reindex())
    except EmbeddingInputError as exc:
        write_error("invalid_embedding_config", type(exc).__name__)
        return 2
    except EmbeddingUnavailableError as exc:
        write_error("embedding_unavailable", type(exc).__name__)
        return 3
    except Exception as exc:
        write_error("database_error", type(exc).__name__)
        return 1
    sys.stdout.write(json.dumps(asdict(summary), ensure_ascii=False, sort_keys=True) + "\n")
    return 0
```

退出码固定：0 成功或幂等跳过、1 数据库/未知错误、2 配置/输入错误、3 Provider 不可用。stderr 不输出 URL、密钥、正文或异常栈；无论成功失败都 `await dispose_engine()` 和关闭 AsyncOpenAI client。

- [ ] **Step 7: 运行单元、集成和 CLI 双跑**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\knowledge\test_reindex_service.py tests\integration\knowledge\test_reindex.py tests\integration\knowledge\test_reindex_cli.py -q
```

Expected: PASS；测试库最终 0/0 或由 fixture 明确清理；主库不被测试触碰。

- [ ] **Step 8: 提交**

```powershell
git add app/modules/knowledge/reindex_service.py app/modules/knowledge/repository.py scripts/reindex_knowledge.py tests/unit/knowledge/test_reindex_service.py tests/integration/knowledge/test_reindex.py tests/integration/knowledge/test_reindex_cli.py
git commit -m "feat: reindex knowledge chunks into pgvector"
```

---

## Task 6: 建立 FAISS/pgvector 同模型对账与性能 artifact

**Files:**

- Create: `scripts/legacy_faiss_retriever.py`
- Create: `scripts/evaluate_pgvector_retrieval.py`
- Modify: `scripts/capture_retrieval_baseline.py`
- Create: `tests/evaluation/__init__.py`
- Create: `tests/evaluation/test_faiss_pgvector_parity.py`
- Modify: `tests/test_retrieval_baseline.py`

- [ ] **Step 1: 写指标失败测试**

```python
def test_calculate_parity_metrics_uses_set_overlap_and_expected_hits() -> None:
    rows = [
        make_question(expected={"k1"}, faiss=["k1", "k2", "k3"], pg=["k1", "k2", "k4"], ms=4.0),
        make_question(expected={"k5"}, faiss=["k5", "k6", "k7"], pg=["k8", "k6", "k7"], ms=6.0),
    ]
    metrics = calculate_metrics(rows)
    assert metrics.expected_hit_count == 1
    assert metrics.expected_hit_rate == pytest.approx(0.5)
    assert metrics.average_top_k_overlap == pytest.approx((2 / 3 + 2 / 3) / 2)
    assert metrics.pgvector_p95_ms == pytest.approx(6.0)


def test_artifact_does_not_contain_provider_url_key_or_knowledge_body() -> None:
    metrics = RetrievalMetrics(
        total_questions=26,
        expected_hit_count=26,
        expected_hit_rate=1.0,
        average_top_k_overlap=1.0,
        pgvector_p50_ms=3.0,
        pgvector_p95_ms=5.0,
    )
    artifact = build_artifact(
        metrics=metrics,
        questions=[],
        git_sha="a" * 40,
        fixture_sha256="b" * 64,
        seed_sha256="c" * 64,
        faiss_sha256="d" * 64,
        id_map_sha256="e" * 64,
        embedding_model="embedding-test",
        dimensions=1024,
        provider_origin_sha256="f" * 64,
    )
    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    for forbidden in ("api_key", "base_url", "authorization", "password", "secret", "正文"):
        assert forbidden not in serialized
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\evaluation\test_faiss_pgvector_parity.py -q
```

Expected: FAIL，缺少 evaluator。

- [ ] **Step 3: 提取只读 legacy FAISS adapter**

将当前 `Retriever` 中只读加载和搜索逻辑移到 `scripts/legacy_faiss_retriever.py`，但接口改为接受已经生成的 query vector：

```python
class LegacyFaissRetriever:
    def __init__(self, *, index_path: Path, id_map_path: Path) -> None:
        self._index = faiss.read_index(str(index_path))
        payload = json.loads(id_map_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("id_map 根节点必须是对象")
        self._id_map = payload
        if self._index.ntotal != len(self._id_map):
            raise ValueError("FAISS 索引数量与 id_map 不一致")

    def search_vector(self, vector: Sequence[float], *, top_k: int) -> list[str]:
        """只返回 legacy source_id，不读取 processed JSONL 或知识正文。"""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query = numpy.asarray([list(vector)], dtype="float32")
        _distances, indices = self._index.search(query, min(top_k, self._index.ntotal))
        output: list[str] = []
        for index in indices[0].tolist():
            if index < 0:
                continue
            row = self._id_map.get(str(index))
            if not isinstance(row, dict):
                raise ValueError("id_map 条目必须是对象")
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                raise ValueError("id_map 条目缺少 source_id")
            output.append(source_id)
        return output
```

这样 FAISS 与 pgvector 使用同一批 query vectors，避免调用两次 Provider 或模型漂移。Adapter 只读取冻结工件，不提供写入函数。

- [ ] **Step 4: 实现 evaluator**

执行顺序固定：

1. 加载并严格验证现有 26 题 fixture 和规范化 seed SHA；
2. 一次批量生成 26 个 query vectors；
3. 每题用同一 vector 分别查询 legacy FAISS 和 pgvector；
4. 只围绕 pgvector Repository 调用计时，排除网络与 FAISS 时间；
5. 计算 expected hit、Top-3 集合重合率和 P95；
6. 门槛失败时退出 4；成功时写 UTF-8 JSON artifact。

核心门槛函数必须是纯函数：

```python
@dataclass(frozen=True)
class RetrievalMetrics:
    total_questions: int
    expected_hit_count: int
    expected_hit_rate: float
    average_top_k_overlap: float
    pgvector_p50_ms: float
    pgvector_p95_ms: float


def build_artifact(
    *,
    metrics: RetrievalMetrics,
    questions: Sequence[Mapping[str, object]],
    git_sha: str,
    fixture_sha256: str,
    seed_sha256: str,
    faiss_sha256: str,
    id_map_sha256: str,
    embedding_model: str,
    dimensions: int,
    provider_origin_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "git_sha": git_sha,
        "inputs": {
            "fixture_sha256": fixture_sha256,
            "seed_sha256": seed_sha256,
            "faiss_sha256": faiss_sha256,
            "id_map_sha256": id_map_sha256,
        },
        "embedding": {
            "model": embedding_model,
            "dimensions": dimensions,
            "provider_origin_sha256": provider_origin_sha256,
        },
        "metrics": asdict(metrics),
        "questions": [dict(question) for question in questions],
    }


def assert_thresholds(metrics: RetrievalMetrics) -> None:
    if metrics.total_questions != 26:
        raise EvaluationThresholdError("固定题集必须包含 26 题")
    if metrics.expected_hit_rate < 0.90:
        raise EvaluationThresholdError("pgvector Top-3 期望命中率低于 90%")
    if metrics.average_top_k_overlap < 0.80:
        raise EvaluationThresholdError("FAISS/pgvector Top-3 平均重合率低于 80%")
    if metrics.pgvector_p95_ms > 100.0:
        raise EvaluationThresholdError("pgvector 精确检索 P95 超过 100 ms")
```

Artifact 只包含：提交 SHA、UTC 时间、fixture/seed/FAISS/id_map SHA、模型名、维度、provider origin 的 SHA-256、每题 legacy ID 结果、聚合指标；不包含完整向量、知识正文、URL 或密钥。

- [ ] **Step 5: 更新旧 baseline 工具依赖边界**

`scripts/capture_retrieval_baseline.py` 的默认路径改为调用 `LegacyFaissRetriever`，不再 import `app.services.retriever`。注入 `retrieve_fn` 的现有单元测试保持通过。

- [ ] **Step 6: 运行离线指标测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\evaluation\test_faiss_pgvector_parity.py tests\test_retrieval_baseline.py -q
```

Expected: PASS，不访问网络。

- [ ] **Step 7: 在真实 Provider 与主库上运行验收**

```powershell
$env:DATABASE_URL='postgresql+asyncpg://mathrag:***@localhost:5432/mathrag'
.\.venv\Scripts\python.exe -m scripts.reindex_knowledge
.\.venv\Scripts\python.exe -m scripts.evaluate_pgvector_retrieval --fixture tests/fixtures/retrieval_questions.json --output docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json
```

Expected: 26 个 chunk ready；至少 24/26 命中；平均重合率至少 0.80；P95 不超过 100 ms。若失败，停止在线切换，先核对模型、维度、规范化和距离度量，不降低门槛。

- [ ] **Step 8: 提交**

```powershell
git add scripts/legacy_faiss_retriever.py scripts/evaluate_pgvector_retrieval.py scripts/capture_retrieval_baseline.py tests/evaluation tests/test_retrieval_baseline.py docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json
git commit -m "test: verify pgvector retrieval parity"
```

---

## Task 7: 将在线 RAG 异步切换到 Knowledge Search Service

**Files:**

- Modify: `app/services/rag_pipeline.py`
- Modify: `app/api/chat.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/main.py`
- Modify: `tests/test_agentic_rag.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 将现有 RAG 测试改为异步失败测试**

```python
def test_rag_pipeline_uses_one_batched_knowledge_search() -> None:
    search = FakeKnowledgeSearchService(
        hits=[make_hit("c2", 0.88), make_hit("c1", 0.60)]
    )
    pipeline = RAGPipeline(knowledge_search=search, planner=MockPlanner(), llm=MockLLM())

    result = asyncio.run(pipeline.chat(question="什么是二次函数", history=[], top_k=2))

    assert search.calls == [(["什么是二次函数", "二次函数 定义", "抛物线 图像"], 2)]
    assert result["references"][0]["chunk_id"] == "c2"
    assert result["references"][0]["index"] is None
```

Chat API 中原有的 `mock_chat_with_rag` 全部改为与生产函数相同参数的 `async def`，并新增：

```python
def test_chat_returns_503_without_leaking_database_error(monkeypatch) -> None:
    async def fail(**kwargs):
        raise SQLAlchemyError("postgresql://user:password@host/database")
    monkeypatch.setattr("app.api.chat.chat_with_rag", fail)
    response = client.post("/api/chat", json={"question": "测试", "history": [], "top_k": 3})
    assert response.status_code == 503
    assert "password" not in response.text
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_rag.py tests\test_chat_api.py -q
```

Expected: FAIL，当前管道同步且仍 import FAISS retrieve。

- [ ] **Step 3: 异步化 RAGPipeline**

```python
class RAGPipeline:
    def __init__(
        self,
        *,
        knowledge_search: KnowledgeSearchService,
        planner: QueryPlanner | None = None,
    ) -> None:
        self._knowledge_search = knowledge_search
        self._planner = planner or get_query_planner()

    async def chat(self, question: str, history=None, top_k: int | None = None) -> dict[str, object]:
        question, k = self._validate(question, top_k)
        plan = await asyncio.to_thread(
            self._planner.create_plan,
            question=question,
            history=history,
        )
        queries = self._normalize_queries(question, plan.retrieval_queries)
        hits = await self._knowledge_search.search(queries, top_k=k)
        references = [hit.to_reference(rank=index) for index, hit in enumerate(hits, start=1)]
        messages = build_chat_messages(question=question, references=references, history=history)
        llm_result = await asyncio.to_thread(chat_json, messages=messages)
        return self._build_result(question, plan, references, llm_result.data)
```

删除 `_merge_references()` 和逐查询 `retrieve()` 循环；合并职责已经位于 Knowledge 模块。同步 LLM/Planner 通过 `asyncio.to_thread()` 过渡，避免阻塞 FastAPI event loop，M3 不扩大为 LLM Provider 重写。

- [ ] **Step 4: 异步化 Chat API 并安全映射错误**

```python
@router.post("/chat", response_model=ChatResponse, summary="数学 RAG 问答")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = await chat_with_rag(
            question=request.question,
            history=_history_to_dicts(request.history),
            top_k=request.top_k,
        )
        return ChatResponse(**result)
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="知识检索暂不可用。") from exc
    except EmbeddingUnavailableError as exc:
        raise HTTPException(status_code=502, detail="向量服务暂不可用。") from exc
```

保留现有 Pydantic 422、ValueError 400 和 LLM 错误映射；删除 FileNotFoundError/FAISS 文件缺失分支。catch-all 返回固定“系统内部错误”，不得拼接 `str(exc)`。

- [ ] **Step 5: 保持响应契约并更新应用描述**

- `ReferenceItem.index` 仍为 `Optional[int]`，description 改为“旧版兼容字段；pgvector 路径为空”。
- `app/main.py` 描述改为“基于 FastAPI + PostgreSQL/pgvector + 大模型 API”。
- `app/main.py` 的 lifespan 在 `finally` 中依次调用 `dispose_embedding_provider()` 与 `dispose_engine()`，即使应用关闭期间其中一步失败也尝试释放另一项资源。
- `ChatRequest.top_k` 保持 1 至 10。

- [ ] **Step 6: 运行 RAG 与 API 回归**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agentic_rag.py tests\test_chat_api.py tests\test_chat_schema.py -q
```

Expected: PASS；测试不读取 FAISS 文件，不连接真实外部 API。

- [ ] **Step 7: 提交**

```powershell
git add app/services/rag_pipeline.py app/api/chat.py app/schemas/chat.py app/main.py tests/test_agentic_rag.py tests/test_chat_api.py
git commit -m "feat: switch online rag retrieval to pgvector"
```

---

## Task 8: 删除在线 FAISS 依赖并完成 M3 验收

**Files:**

- Delete: `app/services/retriever.py`
- Delete: `app/services/vector_store.py`
- Delete: `app/services/embedding_service.py`
- Delete: `scripts/build_index.py`
- Modify: `scripts/demo_query.py`
- Modify: `app/core/config.py`
- Modify: `requirements.txt`
- Modify: `requirements.lock.txt`
- Create: `requirements-evaluation.txt`
- Create: `requirements-evaluation.lock.txt`
- Modify: `Dockerfile`
- Modify: `README.md`
- Create: `tests/test_runtime_dependency_boundary.py`
- Create: `docs/baselines/2026-07-30-m3-pgvector-retrieval-cutover.md`

- [ ] **Step 1: 先写运行时边界失败测试**

```python
def test_online_app_does_not_reference_faiss_artifacts() -> None:
    protected_files = [
        PROJECT_ROOT / "app" / "api" / "chat.py",
        PROJECT_ROOT / "app" / "services" / "rag_pipeline.py",
        PROJECT_ROOT / "app" / "modules" / "knowledge" / "search_service.py",
        PROJECT_ROOT / "app" / "main.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in protected_files).lower()
    for forbidden in ("faiss", "id_map.json", "kb_chunks.jsonl", "app.services.retriever"):
        assert forbidden not in combined


def test_runtime_lock_does_not_install_faiss() -> None:
    runtime_lock = (PROJECT_ROOT / "requirements.lock.txt").read_text(encoding="utf-8").lower()
    evaluation_lock = (PROJECT_ROOT / "requirements-evaluation.lock.txt").read_text(encoding="utf-8").lower()
    assert "faiss-cpu" not in runtime_lock
    assert "faiss-cpu" in evaluation_lock
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_dependency_boundary.py -q
```

Expected: FAIL，在线 app 和 runtime lock 仍包含 FAISS。

- [ ] **Step 3: 清理在线模块与脚本**

- 删除三个旧 app service；所有只读 FAISS 逻辑只保留在 `scripts/legacy_faiss_retriever.py`。
- 删除 `scripts/build_index.py`，避免生成并维护第二套在线索引。
- 将 `scripts/demo_query.py` 改为 `asyncio.run(build_knowledge_search_service().search([question], top_k=top_k))`。
- 从 `Settings` 删除只服务于在线 FAISS 的 `FAISS_INDEX_PATH`、`ID_MAP_PATH` 和 `USE_INNER_PRODUCT`；保留 `PROCESSED_KB_PATH`，因为 M2 离线导入仍需使用。
- README 删除“启动前必须存在 FAISS/index/id_map”的说明，新增 `alembic upgrade head`、M2 导入、M3 reindex、pgvector demo 和 evaluation 命令。

- [ ] **Step 4: 拆分生产与 evaluation 依赖**

`requirements.txt` 删除 `faiss-cpu`；创建：

```text
# requirements-evaluation.txt
-r requirements.txt
faiss-cpu>=1.8.0
```

生成锁文件：

```powershell
uv pip compile requirements.txt -o requirements.lock.txt
uv pip compile requirements-evaluation.txt -o requirements-evaluation.lock.txt
```

Dockerfile 继续只安装 `requirements.lock.txt`；开发/验收环境安装 `requirements-evaluation.lock.txt`。比较锁文件时不得手工猜测或遗漏传递依赖。

- [ ] **Step 5: 运行边界测试和静态搜索**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_runtime_dependency_boundary.py -q
rg -n "app\.services\.retriever|get_retriever|FAISS_INDEX_PATH|ID_MAP_PATH" app
rg -n "faiss|id_map\.json|kb_chunks\.jsonl" app/api app/modules app/services/rag_pipeline.py app/main.py
```

Expected: 测试 PASS；两次 `rg` 均无在线匹配。

- [ ] **Step 6: 在干净数据库执行最终现场流程**

```powershell
docker compose up -d postgres
$env:DATABASE_URL='postgresql+asyncpg://mathrag:***@localhost:5432/mathrag'
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:***@localhost:5432/mathrag_test'
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge
.\.venv\Scripts\python.exe -m scripts.reindex_knowledge
.\.venv\Scripts\python.exe -m scripts.reindex_knowledge
.\.venv\Scripts\python.exe -m scripts.evaluate_pgvector_retrieval --fixture tests/fixtures/retrieval_questions.json --output docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json
```

Expected:

- Alembic current=`0003_enforce_vector_readiness`；`alembic check` 无漂移。
- legacy 导入第一次按实际库状态创建或跳过，第二次固定跳过 26；无冲突。
- reindex 第一次把所有需处理 chunk 变为 ready，第二次 selected=0、failed=0。
- 主库 items/chunks=26/26；item ready=26；chunk ready=26；embedding 非空=26；当前模型=26。
- 对账达到本计划全部量化门槛。

- [ ] **Step 7: 运行完整测试与容器构建验证**

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp "$env:TEMP\mathrag-m3-pytest"
.\.venv\Scripts\python.exe -m alembic check
docker compose build mathrag
docker compose up -d mathrag
curl.exe -f http://127.0.0.1:8000/health/live
curl.exe -f http://127.0.0.1:8000/health/ready
```

Expected: 全量 0 failed；容器镜像不安装 faiss-cpu；live/ready 均 200。聊天实际请求仅在受控 Provider 配置下执行，不在日志输出密钥。

- [ ] **Step 8: 写 M3 验收基线**

`docs/baselines/2026-07-30-m3-pgvector-retrieval-cutover.md` 必须记录：

- base/head SHA、Python/PostgreSQL/pgvector/依赖版本；
- migration、导入双跑、reindex 双跑的脱敏命令和结果；
- 26 条状态、模型、维度和集合摘要；
- 26 题 expected hit、平均重合率、P50/P95；
- artifact SHA-256 和 provider origin SHA-256，不记录 URL/密钥；
- 在线 import graph 与 runtime lock 的 FAISS 清零证据；
- 回滚步骤：停止写入、切回 M2 提交/镜像、只读使用冻结 FAISS 工件；不得在同一在线版本长期启用双路径。

- [ ] **Step 9: 停止容器并最终检查**

```powershell
docker compose stop
docker compose ps --all
git diff --check main...HEAD
git status --short --branch
```

Expected: 项目容器 Exited；工作树除待提交验收文档/artifact 外无意外改动；`git diff --check` 通过。

- [ ] **Step 10: 提交**

```powershell
git add app scripts tests requirements.txt requirements.lock.txt requirements-evaluation.txt requirements-evaluation.lock.txt Dockerfile README.md docs/baselines
git commit -m "chore: complete pgvector online cutover"
```

---

## 最终验收清单

- [ ] `main` 的 M2 合并与换行哈希修复均为 M3 祖先。
- [ ] 0003 upgrade/downgrade/upgrade 往返成功，测试库守卫在所有破坏动作前执行。
- [ ] 26 个 chunk 使用同一模型、同一 1024 维规范和有限非零向量。
- [ ] public + ready + current model 过滤在 SQL 中完成，不在 Python 中事后删除。
- [ ] private、pending、failed、旧模型结果泄漏均为 0。
- [ ] Embedding 网络调用和 LLM 网络调用期间没有打开数据库事务。
- [ ] Repository 无 `commit()`、`rollback()`、engine 或 session factory。
- [ ] reindex 双跑可重入，失败 batch 可重试，错误不泄露正文/URL/密钥。
- [ ] 固定 26 题 Top-3 命中至少 24/26，FAISS/pgvector 平均重合率至少 0.80。
- [ ] pgvector 精确 SQL P95 不高于 100 ms，计时不含 Embedding API。
- [ ] `/api/chat` 保持现有响应字段和 top_k 1..10 行为，`index=None`。
- [ ] 在线 app import graph 不再引用 FAISS、id_map 或 processed JSONL。
- [ ] 生产依赖锁和 Docker 镜像不包含 `faiss-cpu`；evaluation 环境仍可只读复核历史工件。
- [ ] 全量测试、Alembic check、Docker build、live/ready 全部通过。
- [ ] M3 baseline/artifact 不含任何秘密或完整知识正文。

## 计划自检

- 需求覆盖：向量生成、状态迁移、精确 SQL、权限/状态/模型过滤、多查询去重、质量对账、P95、在线切换、FAISS 清理和回滚均有对应 Task。
- 范围控制：不创建用户/owner、会话持久化、文档导入、HNSW、Redis、Celery 或多 Worker 协调；这些属于 M4+ 或独立 ADR。
- 类型一致：Provider、Repository、Search Service、RAGPipeline 全程使用 1024 维 `list[float]`；公开结果统一为 `KnowledgeSearchHit`；API reference 由 `to_reference()` 单点映射。
- 事务一致：Repository 无事务；Search Service 在 Provider 之后打开短只读 Session；Reindex 每个状态写回使用独立短事务。
- 可执行性：每项实现先有明确 RED 命令，再有最小代码、GREEN 命令和独立提交。
