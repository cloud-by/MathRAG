# MathRAG M5 知识与导入管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付受认证保护的知识 CRUD、`revision` 乐观锁、安全 PDF 上传、可查询/取消/重试的 ingestion 状态机，并让 Web 与管理员 CLI 复用同一套数据库、抽取和向量化编排。

**Architecture:** PostgreSQL 继续是知识唯一事实源。知识写入和导入都先在短事务中保存可恢复状态，关闭 Session 后调用 PDF/网页读取、LLM 和 Embedding，再以新的短事务 CAS 写回终态；Web 只处理有严格上限的小文件并使用 FastAPI `BackgroundTasks`，大批量导入继续由管理员 CLI 驱动，但两者调用相同的 `IngestionService` 和 repository。文件只保存到受控根目录，数据库和 API 只持有相对路径或脱敏元数据。

**Tech Stack:** Python 3.11.9、FastAPI 0.140.13、Starlette `UploadFile`/`BackgroundTasks`、Pydantic 2.13.4、SQLAlchemy 2.0.51 asyncio、asyncpg 0.31.0、Alembic 1.18.5、PostgreSQL 18.4、pgvector 0.8.5、pypdf 6.14.2、OpenAI-compatible LLM/Embedding provider、pytest 9.1.1、Docker Compose。

---

## 起点与前置条件

- 开发分支：`codex/m5-knowledge-ingestion-management`。
- 分支基点：`main@366383b`，即 M4 验收提交；`main`、M4 分支和 M5 起点指向同一提交。
- 合并后的 `main` 已重新执行全量测试：`514 passed, 1 warning in 64.33s`；0 failed、0 skipped。
- Alembic 当前 head 是 `0004_create_identity_conversation_rag_tables`。项目完成计划中的 M5 `0004_create_documents_ingestion_jobs.py` 已被 M4 占用，本计划固定使用 `0005_create_documents_ingestion_jobs.py`。
- M4 已提供 Session、CSRF、`require_admin`/`require_admin_csrf`、`owner_id`、统一 `/api/v1` 错误包络和异步 Session factory；M5 必须复用这些边界。
- M3/M4 已保证在线检索只读 PostgreSQL/pgvector；M5 不恢复 JSONL、FAISS 或 `id_map.json` 在线写入/读取。
- 测试结束后 Compose 容器必须全部停止；未跟踪的 `tmp/` 属于用户数据，不读取、不修改、不暂存。
- 设计依据：`docs/superpowers/specs/2026-07-29-mathrag-project-architecture-design.md` 第 10.2、11.2、14.3、15、16、17、18、19、22、23 节，以及项目完成计划 M5 清单。

## 方案选择

### 采用：持久化状态机 + Web 小批量后台任务 + CLI 复用

- `POST /api/v1/documents` 只接收受限 PDF，先安全落盘并创建 `document`/`ingestion_job=pending`，返回 202 后通过 `BackgroundTasks` 启动同进程小任务。
- CLI 负责目录和网页批量导入，但只能调用 `IngestionService`，不能直接 `INSERT`、追加 seed JSONL 或自行实现向量写回。
- 任务状态、重试输入、错误码和关联知识全部持久化；进程中不保存影响正确性的任务队列。
- 应用重启时未完成任务仍可诊断；M5 由管理员显式重试，自动恢复和独立 Worker 留到出现规模证据后。

### 不采用：请求内同步完成全部导入

- 虽然实现较少，但上传请求会等待 PDF、LLM 和 Embedding，容易超过代理超时，也无法让 M6 稳定轮询任务状态。

### 不采用：Redis/Celery 或独立 Worker

- 当前是单实例、管理员小批量场景；提前增加 Broker、worker 生命周期和分布式幂等会扩大部署面，违反架构文档的扩展触发条件。

## 冻结范围

### M5 包含

- `/api/v1/knowledge-items` 列表、创建、读取、更新和归档；普通用户只能读取 `public + ready`，管理员可读写全部状态。
- `revision` CAS：更新必须携带期望 revision，冲突固定返回 `KNOWLEDGE_REVISION_CONFLICT`/409。
- 知识创建和影响检索文本的更新自动生成/刷新单个 chunk；Embedding 在事务外执行，成功后短事务切换为 `ready`。
- `documents`、`ingestion_jobs`、知识来源关联、PDF 上传根目录和 Compose 持久化卷。
- PDF 扩展名、声明 MIME、文件头、大小、页数、加密状态和可提取文本校验；服务端 UUID 文件名和路径越界防护。
- `pending -> running -> completed|failed`、`pending -> cancelled`、`failed -> running` 状态转换和 CAS。
- `GET /api/v1/documents`、`POST /api/v1/documents`、`GET /api/v1/ingestion-jobs/{id}`、取消和重试动作。
- 文本、网页、PDF 清洗/抽取适配器；CLI 和 Web 共用 ingestion service/repository。
- 任务错误摘要脱敏、重复执行不重复创建 chunk、不覆盖 completed 任务。

### M5 不包含

- Vue 3 页面、前端轮询和旧静态前端替换；属于 M6。
- Redis、Celery、跨进程任务租约、自动恢复 running 任务、多 Worker 协调；属于扩展触发项或 M7。
- OCR、Office 文件、图片、任意 URL 在线上传、匿名上传和普通用户知识写入。
- 硬删除文档/知识、知识审批工作流、全文搜索、批量编辑和 HNSW 调优。
- 修改 M4 `/api/v1/chat` 契约；RAG 继续只检索 `public + ready`。

## 冻结状态与权限

### 知识状态

```text
create/update content -> indexing -> ready
                         |          |
                         +-> failed +-> archived（管理员 DELETE）
```

- 创建时 revision=1；每次成功接受 PATCH 或归档都原子 `revision = revision + 1`。
- 修改 category/title/keywords/content/example/steps/difficulty 会重建检索文本和向量。
- 只修改 visibility 也必须使用 revision CAS，但不重算向量。
- archived 条目不参与检索；DELETE 是软归档，返回 204。

### 文档状态

```text
pending -> processing -> ready
                    +--> failed
pending/ready/failed -> archived（M5 仅保留 service 能力，不发布硬删除）
```

### 导入任务状态

```text
pending -> running -> completed
                   +-> failed -> running（管理员重试）
pending -> cancelled
```

- `claim_pending()`、`claim_retry()`、`complete()`、`fail()`、`cancel_pending()` 都使用状态条件 UPDATE；rowcount 不是 1 即冲突。
- completed 任务不可重试；running 任务不可取消；重复后台回调不能再次执行外部调用。
- progress 只允许 0..100，pending=0、running 分阶段推进、completed=100；失败保留最后进度。

### 权限

| 操作 | 未登录 | 普通用户 | 管理员 |
|---|---:|---:|---:|
| 读取 public+ready 知识 | 401 | 允许 | 允许 |
| 读取 private/draft/failed/archived | 401 | 404 | 允许 |
| 创建/更新/归档知识 | 401 | 403 | 允许，修改请求需 CSRF |
| 上传/列出文档 | 401 | 403 | 允许，上传需 CSRF |
| 读取/取消/重试 ingestion job | 401 | 403 | 允许，动作需 CSRF |

## 冻结 API 契约

### Knowledge Items

```text
GET    /api/v1/knowledge-items?page=1&page_size=20&status=ready&visibility=public&category=algebra
POST   /api/v1/knowledge-items
GET    /api/v1/knowledge-items/11111111-1111-4111-8111-111111111111
PATCH  /api/v1/knowledge-items/11111111-1111-4111-8111-111111111111
DELETE /api/v1/knowledge-items/11111111-1111-4111-8111-111111111111?revision=3
```

创建请求：

```json
{
  "category": "代数",
  "title": "一元二次方程求根公式",
  "keywords": ["一元二次方程", "判别式", "求根公式"],
  "content": "当二次项系数不为零时，可以使用求根公式求解。",
  "example": "求解 x^2-3x+2=0。",
  "steps": ["计算判别式", "代入求根公式", "化简结果"],
  "difficulty": "easy",
  "visibility": "public"
}
```

更新请求只接受可编辑字段和 revision：

```json
{
  "revision": 3,
  "content": "更新后的完整解释。",
  "visibility": "private"
}
```

资源响应不暴露 ORM、embedding 或内部来源路径：

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "legacy_id": null,
  "owner_id": "22222222-2222-4222-8222-222222222222",
  "category": "代数",
  "title": "一元二次方程求根公式",
  "keywords": ["一元二次方程", "判别式", "求根公式"],
  "content": "更新后的完整解释。",
  "example": "求解 x^2-3x+2=0。",
  "steps": ["计算判别式", "代入求根公式", "化简结果"],
  "difficulty": "easy",
  "visibility": "private",
  "status": "ready",
  "revision": 4,
  "created_at": "2026-07-31T06:00:00Z",
  "updated_at": "2026-07-31T06:05:00Z"
}
```

### Documents 与 ingestion jobs

```text
POST /api/v1/documents              multipart/form-data: file=<PDF>, category=代数
GET  /api/v1/documents?page=1&page_size=20&status=pending
GET  /api/v1/ingestion-jobs/33333333-3333-4333-8333-333333333333
POST /api/v1/ingestion-jobs/33333333-3333-4333-8333-333333333333/cancel
POST /api/v1/ingestion-jobs/33333333-3333-4333-8333-333333333333/retry
```

- 上传成功返回 202，响应包含安全的 `document` 和 `job` 资源；不返回 `storage_path` 或绝对路径。
- cancel 成功返回更新后的 cancelled job；只有 pending 可取消。
- retry 成功先以 CAS 执行 failed->running、attempt_count+1，再返回 202 并安排后台续跑。
- `GET job` 返回 `error_code` 和最多 500 字的管理员摘要，但不包含密钥、连接串、SQL、堆栈、Cookie、绝对路径或供应商原始响应。

## 数据库契约

### documents

| 字段 | 类型/约束 |
|---|---|
| id | UUID PK |
| owner_id | UUID nullable FK users ON DELETE SET NULL |
| original_name | varchar(255) not null，仅元数据 |
| storage_path | varchar(512) not null unique，仅受控根目录相对路径 |
| mime_type | varchar(128) not null，M5 固定 application/pdf |
| size_bytes | bigint not null，1..MAX_UPLOAD_BYTES |
| sha256 | char(64) not null，小写十六进制 |
| status | pending/processing/ready/failed/archived |
| created_at/updated_at | timestamptz not null |

- `(owner_id, sha256)` 唯一，避免同一管理员重复上传同一内容。

### ingestion_jobs

| 字段 | 类型/约束 |
|---|---|
| id | UUID PK |
| requested_by | UUID nullable FK users ON DELETE SET NULL |
| document_id | UUID nullable FK documents ON DELETE SET NULL |
| job_type | text/pdf/web/reindex |
| status | pending/running/completed/failed/cancelled |
| progress | integer 0..100 |
| request_payload | JSONB not null，重试所需的非密钥输入 |
| attempt_count | integer >=0 |
| error_code | varchar(64) nullable |
| error_message | varchar(500) nullable |
| started_at/finished_at | timestamptz nullable |
| created_at/updated_at | timestamptz not null |

- `(document_id, job_type)` 在 document_id 非空时唯一；PDF 重试复用原 job。
- request_payload 的 pdf 类型只保存 category 等参数，不保存本机路径；text 保存受大小限制的原文；web 保存 sources/keywords/limit/category/delay。

### 知识来源关联

- `knowledge_items.ingestion_job_id UUID NULL REFERENCES ingestion_jobs(id) ON DELETE SET NULL`，增加普通索引。
- `knowledge_chunks.document_id UUID NULL REFERENCES documents(id) ON DELETE SET NULL`，增加普通索引和 `(document_id, chunk_index)` 部分唯一约束。
- `knowledge_item_id` 继续 not null；一次导入产生的 chunk 同时关联 item 和可选 document。
- 文档内 chunk_index 使用全局递增序号，避免多个知识条目各自从 0 开始导致冲突。

## 文件结构

```text
alembic/versions/
└── 0005_create_documents_ingestion_jobs.py

app/core/config.py
app/main.py
app/modules/knowledge/
├── errors.py
├── management_repository.py
├── management_schemas.py
├── management_service.py
├── models.py
├── rendering.py
├── router.py
└── schemas.py
app/modules/ingestion/
├── __init__.py
├── errors.py
├── extractors.py
├── models.py
├── repository.py
├── router.py
├── schemas.py
├── service.py
└── storage.py
app/services/
├── knowledge_extractor.py
├── math_knowledge_importer.py
└── pdf_knowledge_importer.py
scripts/
├── import_math_knowledge.py
└── import_pdf_knowledge.py
tests/api/
├── test_documents.py
├── test_ingestion_jobs.py
└── test_knowledge_items.py
tests/integration/ingestion/
├── __init__.py
├── test_cli.py
├── test_ingestion_pipeline.py
├── test_repository.py
└── test_retry_idempotency.py
tests/integration/
├── test_m5_migration_schema.py
└── test_m5_workflow.py
tests/unit/ingestion/
├── __init__.py
├── test_config.py
├── test_extractors.py
├── test_pipeline.py
├── test_service.py
└── test_storage.py
tests/unit/knowledge/
├── test_management_schemas.py
├── test_management_service.py
└── test_rendering.py
```

---

## Task 1: 锁定 M5 配置、上传根目录和公开 schema

**Files:**
- Modify: `app/core/config.py`
- Modify: `.env.example`
- Modify: `.gitignore`
- Modify: `docker-compose.yml`
- Create: `app/modules/knowledge/management_schemas.py`
- Create: `app/modules/ingestion/__init__.py`
- Create: `app/modules/ingestion/schemas.py`
- Test: `tests/unit/ingestion/test_config.py`
- Test: `tests/unit/knowledge/test_management_schemas.py`

- [ ] **Step 1: 写失败的配置和 schema 测试**

```python
def test_ingestion_limits_must_be_positive(tmp_path):
    configured = Settings(
        UPLOAD_DIR=tmp_path / "uploads",
        MAX_UPLOAD_BYTES=10,
        MAX_PDF_PAGES=2,
        MAX_INGESTION_TEXT_CHARS=100,
        INGESTION_CHUNK_CHARS=50,
    )
    assert configured.UPLOAD_DIR == tmp_path / "uploads"
    assert configured.MAX_UPLOAD_BYTES == 10
    with pytest.raises(ValueError, match="MAX_UPLOAD_BYTES"):
        Settings(MAX_UPLOAD_BYTES=0)


def test_knowledge_update_requires_revision_and_rejects_unknown_fields():
    request = KnowledgeItemUpdate(revision=4, visibility="private")
    assert request.revision == 4
    with pytest.raises(ValidationError):
        KnowledgeItemUpdate.model_validate({"revision": 4, "status": "ready"})
```

- [ ] **Step 2: 运行测试并确认因 Settings 字段和 schema 不存在而失败**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_config.py tests/unit/knowledge/test_management_schemas.py -q
```

Expected: FAIL，提示缺少 ingestion 配置或公开 schema。

- [ ] **Step 3: 增加冻结配置**

在 `Settings` 增加：

```python
UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "data" / "uploads")))
MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "200"))
MAX_INGESTION_TEXT_CHARS: int = int(os.getenv("MAX_INGESTION_TEXT_CHARS", "200000"))
INGESTION_CHUNK_CHARS: int = int(os.getenv("INGESTION_CHUNK_CHARS", "4000"))
```

把五个数值设置加入正数校验。`UPLOAD_DIR` 只定义受控根目录，不把客户端输入拼入该路径。

- [ ] **Step 4: 定义 Pydantic 契约**

`KnowledgeItemCreate`/`KnowledgeItemUpdate`/`KnowledgeItemRead`/`KnowledgeItemPage` 使用 `extra="forbid"`；字符串去空白、keywords/steps 去空值和重复项；category/title/content 必填；difficulty 和 visibility 使用 Literal；更新 revision `ge=1`。

`DocumentRead`、`DocumentPage`、`IngestionJobRead` 和 `DocumentAccepted` 不包含 `storage_path`、`request_payload`。状态字段全部使用冻结 Literal。

- [ ] **Step 5: 配置持久化卷和忽略项**

`.env.example` 增加五个配置；`.gitignore` 增加 `data/uploads/`。Compose 的 mathrag 服务增加：

```yaml
environment:
  UPLOAD_DIR: /app/data/uploads
volumes:
  - upload_data:/app/data/uploads
```

顶层 `volumes` 同时声明 `upload_data:`，不能挂载宿主任意路径。

- [ ] **Step 6: 运行定向测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_config.py tests/unit/knowledge/test_management_schemas.py tests/test_compose_contract.py -q
git add .env.example .gitignore docker-compose.yml app/core/config.py app/modules/knowledge/management_schemas.py app/modules/ingestion/__init__.py app/modules/ingestion/schemas.py tests/unit/ingestion/test_config.py tests/unit/knowledge/test_management_schemas.py tests/test_compose_contract.py
git commit -m "feat: define m5 ingestion contracts"
```

Expected: 定向测试 PASS；提交不包含 `.env`、uploads 或 `tmp/`。

---

## Task 2: 创建 0005 schema、ORM 和迁移回环

**Files:**
- Create: `app/modules/ingestion/models.py`
- Modify: `app/modules/knowledge/models.py`
- Modify: `alembic/env.py`
- Create: `alembic/versions/0005_create_documents_ingestion_jobs.py`
- Create: `tests/integration/test_m5_migration_schema.py`
- Modify: `tests/test_runtime_dependency_boundary.py`

- [ ] **Step 1: 写 migration 自包含、约束和回环失败测试**

测试必须从 `0004_create_identity_conversation_rag_tables` 升到 0005，检查 `documents`/`ingestion_jobs`、新增列、FK、check、索引和部分唯一约束；再 downgrade 0004，确认两表和两列消失；最后恢复 head。

```python
def test_m5_migration_is_self_contained():
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert all(not module.startswith("app.") for module in imported)
    assert 'revision: str = "0005_create_documents_ingestion_jobs"' in source
    assert 'down_revision: str | None = "0004_create_identity_conversation_rag_tables"' in source
```

- [ ] **Step 2: 运行测试并确认缺少 0005**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_m5_migration_schema.py -q
```

Expected: FAIL，提示迁移文件或表不存在。

- [ ] **Step 3: 实现 Document 和 IngestionJob ORM**

模型必须使用 `UTCDateTime`、数据库 check 和命名约束；不在 import 时执行 I/O。核心状态约束：

```python
CheckConstraint(
    "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
    name="status",
)
CheckConstraint("progress BETWEEN 0 AND 100", name="progress")
CheckConstraint("attempt_count >= 0", name="attempt_count")
```

Document 使用 `BigInteger` 记录大小、`String(64)` 记录 sha256；KnowledgeItem 增加 nullable `ingestion_job_id`，KnowledgeChunk 增加 nullable `document_id`。跨模块只保存 UUID，不要求 ingestion ORM 反向访问 knowledge ORM。

- [ ] **Step 4: 实现严格升级/降级顺序**

升级顺序：documents -> ingestion_jobs -> knowledge_items.ingestion_job_id -> knowledge_chunks.document_id -> 索引/唯一约束。降级严格逆序。部分唯一索引固定为：

```python
op.create_index(
    "uq_ingestion_jobs_document_id_job_type",
    "ingestion_jobs",
    ["document_id", "job_type"],
    unique=True,
    postgresql_where=sa.text("document_id IS NOT NULL"),
)
```

- [ ] **Step 5: 更新 metadata 注册并运行回环**

`alembic/env.py` 显式导入 Document/IngressJob，`tests/test_runtime_dependency_boundary.py` 证明模型导入不触发 engine/provider 初始化。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_m5_migration_schema.py tests/test_runtime_dependency_boundary.py -q
.\.venv\Scripts\alembic.exe check
```

Expected: 回环 PASS；`alembic check` 输出 `No new upgrade operations detected.`。

- [ ] **Step 6: 提交**

```powershell
git add alembic/env.py alembic/versions/0005_create_documents_ingestion_jobs.py app/modules/ingestion/models.py app/modules/knowledge/models.py tests/integration/test_m5_migration_schema.py tests/test_runtime_dependency_boundary.py
git commit -m "feat: add document and ingestion schema"
```

---

## Task 3: 提取知识渲染规则并实现权限感知读取

**Files:**
- Create: `app/modules/knowledge/rendering.py`
- Create: `app/modules/knowledge/management_repository.py`
- Create: `app/modules/knowledge/management_service.py`
- Modify: `scripts/build_kb.py`
- Modify: `app/modules/knowledge/errors.py`
- Create: `tests/unit/knowledge/test_rendering.py`
- Create: `tests/unit/knowledge/test_management_service.py`
- Create: `tests/integration/knowledge/test_management_repository.py`

- [ ] **Step 1: 写渲染一致性和读权限失败测试**

```python
def test_rendering_matches_legacy_builder():
    payload = {
        "category": "代数",
        "title": "配方法",
        "keywords": ["二次式", "完全平方"],
        "content": "把二次式整理为完全平方。",
        "example": "x^2+2x+1=(x+1)^2",
        "steps": ["补项", "整理"],
        "difficulty": "easy",
    }
    assert build_retrieval_text(payload) == legacy_build_retrieval_text(payload)
    assert build_answer_context(payload) == legacy_build_answer_context(payload)


async def test_user_cannot_observe_private_or_unready_item():
    service = make_service(items=[private_item(), failed_public_item()])
    with pytest.raises(KnowledgeNotFoundError):
        await service.get(private_item().id, user_principal())
```

- [ ] **Step 2: 运行并确认新模块不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/knowledge/test_rendering.py tests/unit/knowledge/test_management_service.py -q
```

Expected: FAIL，提示缺少 rendering/management 模块。

- [ ] **Step 3: 将 build_kb 文本拼接移动到应用模块**

`rendering.py` 提供 `build_retrieval_text(values: Mapping[str, object]) -> str` 和 `build_answer_context(values: Mapping[str, object]) -> str`。`scripts/build_kb.py` 改为导入这两个函数，不保留第二套拼接规则。

- [ ] **Step 4: 实现权限写入 SQL 条件的读取 repository**

```python
def _visibility_predicate(principal: AuthenticatedPrincipal):
    if principal.role == "admin":
        return true()
    return and_(
        KnowledgeItem.visibility == "public",
        KnowledgeItem.status == "ready",
    )
```

`get_visible()` 和 `list_visible()` 都使用该 predicate；普通用户访问 private/unready/archived 与不存在统一返回 `KNOWLEDGE_NOT_FOUND`/404。列表按 `updated_at DESC, id DESC`，分页上限 100。

- [ ] **Step 5: 运行 unit/integration 并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/knowledge/test_rendering.py tests/unit/knowledge/test_management_service.py tests/integration/knowledge/test_management_repository.py -q
git add app/modules/knowledge/rendering.py app/modules/knowledge/management_repository.py app/modules/knowledge/management_service.py app/modules/knowledge/errors.py scripts/build_kb.py tests/unit/knowledge/test_rendering.py tests/unit/knowledge/test_management_service.py tests/integration/knowledge/test_management_repository.py
git commit -m "feat: add permission aware knowledge reads"
```

---

## Task 4: 实现知识写入、revision CAS 和事务外向量化

**Files:**
- Modify: `app/modules/knowledge/management_repository.py`
- Modify: `app/modules/knowledge/management_service.py`
- Modify: `app/modules/knowledge/errors.py`
- Test: `tests/unit/knowledge/test_management_service.py`
- Test: `tests/integration/knowledge/test_management_repository.py`

- [ ] **Step 1: 写 create/update/archive 和事务事件失败测试**

事件探针必须证明 `provider.embed_texts` 发生在两个 Session 生命周期之间：

```python
assert events == [
    "tx1.begin",
    "item.indexing",
    "tx1.commit",
    "tx1.close",
    "embedding.start",
    "embedding.end",
    "tx2.begin",
    "item.ready",
    "tx2.commit",
    "tx2.close",
]
```

并覆盖：revision=3 更新后变 4；旧 revision 返回 409；只改 visibility 不调用 provider；Embedding 失败后 item/chunk 为 failed；archive revision 原子递增。

- [ ] **Step 2: 运行并确认写操作失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/knowledge/test_management_service.py tests/integration/knowledge/test_management_repository.py -q
```

- [ ] **Step 3: 实现 tx1 快照和 CAS**

Repository 方法固定为：

```python
async def create_indexing(self, *, owner_id: UUID, values: Mapping[str, object]) -> IndexingSnapshot
async def update_with_revision(
    self,
    item_id: UUID,
    *,
    expected_revision: int,
    values: Mapping[str, object],
    reindex: bool,
) -> IndexingSnapshot | KnowledgeItem | None
async def archive_with_revision(self, item_id: UUID, expected_revision: int) -> bool
async def complete_indexing(self, snapshot: IndexingSnapshot, vector: Sequence[float], model: str) -> KnowledgeItem | None
async def fail_indexing(self, snapshot: IndexingSnapshot) -> None
```

`update_with_revision` 的 SQL where 同时包含 id、revision 和非 archived 状态；rowcount=0 后额外查询 id，存在则 revision conflict，不存在则 404。

- [ ] **Step 4: 实现 service 两短事务**

```python
async def _embed_and_finalize(self, snapshot: IndexingSnapshot) -> KnowledgeItemRead:
    try:
        vectors = await self._provider.embed_texts([snapshot.retrieval_text])
        vector = validate_and_normalize_vector(vectors[0], self._provider.dimensions)
    except Exception as exc:
        await self._mark_failed(snapshot)
        raise map_knowledge_embedding_error(exc) from None
    async with self._session_factory() as session:
        async with session.begin():
            item = await self._repository_factory(session).complete_indexing(
                snapshot,
                vector,
                self._provider.model,
            )
    if item is None:
        raise KnowledgeRevisionConflictError()
    return KnowledgeItemRead.model_validate(item)
```

跨事务 snapshot 使用 frozen dataclass，只携带 UUID、revision、chunk_id 和文本；不携带 ORM 或 Session。

- [ ] **Step 5: 运行知识全套并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/knowledge tests/integration/knowledge -q
git add app/modules/knowledge/management_repository.py app/modules/knowledge/management_service.py app/modules/knowledge/errors.py tests/unit/knowledge/test_management_service.py tests/integration/knowledge/test_management_repository.py
git commit -m "feat: add revision safe knowledge writes"
```

---

## Task 5: 发布 `/api/v1/knowledge-items` 并完成角色边界

**Files:**
- Create: `app/modules/knowledge/router.py`
- Modify: `app/main.py`
- Create: `tests/api/test_knowledge_items.py`
- Modify: `tests/api/test_errors_v1.py`

- [ ] **Step 1: 写匿名、普通用户、管理员和 409 API 失败测试**

使用 dependency overrides 提供 fake principal/service，不让 API contract 测试访问真实数据库或 provider。

```python
response = client.patch(
    f"/api/v1/knowledge-items/{item_id}",
    json={"revision": 7, "content": "并发后的旧请求"},
    headers=admin_headers,
)
assert response.status_code == 409
assert response.json()["error"]["code"] == "KNOWLEDGE_REVISION_CONFLICT"
assert response.json()["error"]["request_id"]
```

- [ ] **Step 2: 运行并确认路由不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api/test_knowledge_items.py tests/api/test_errors_v1.py -q
```

- [ ] **Step 3: 实现路由依赖**

- GET 使用 `get_current_principal`。
- POST/PATCH/DELETE 使用 `require_admin_csrf`。
- POST 返回 201；DELETE 接受 `revision: Query(ge=1)` 并返回 204。
- Query 只接受冻结 status/visibility/category/page/page_size，普通用户传 private/unready 仍不能扩大权限。
- `get_knowledge_management_service()` 复用 `get_session_factory()` 和 `get_embedding_provider()`。

- [ ] **Step 4: 注册路由并运行 API 全套**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api -q
git add app/main.py app/modules/knowledge/router.py tests/api/test_knowledge_items.py tests/api/test_errors_v1.py
git commit -m "feat: publish knowledge management api"
```

---

## Task 6: 实现受控 PDF 存储和内容校验

**Files:**
- Create: `app/modules/ingestion/errors.py`
- Create: `app/modules/ingestion/storage.py`
- Create: `app/modules/ingestion/extractors.py`
- Modify: `app/services/pdf_knowledge_importer.py`
- Create: `tests/unit/ingestion/test_storage.py`
- Create: `tests/unit/ingestion/test_extractors.py`

- [ ] **Step 1: 写路径、MIME、大小、页数、加密和空 PDF 失败测试**

覆盖 `../escape.pdf`、`folder\\escape.pdf`、`.txt`、`text/plain`、伪 `%PDF-`、超过字节上限、0 页、超过页数、加密 PDF、所有页无文本；断言失败后 `.part` 和最终文件均不存在。

```python
with pytest.raises(DocumentPathError):
    validate_original_name("../escape.pdf")
with pytest.raises(DocumentMimeError):
    await storage.save_upload(fake_upload("notes.pdf", "text/plain", valid_pdf))
assert list(upload_root.rglob("*.part")) == []
```

- [ ] **Step 2: 运行并确认 storage/extractors 不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_storage.py tests/unit/ingestion/test_extractors.py -q
```

- [ ] **Step 3: 实现 UUID 落盘和越界防护**

```python
def resolve_stored_path(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root_resolved / relative_path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        raise DocumentPathError() from None
    return candidate
```

上传按 1 MiB 块读取并同步计算 SHA-256；超过上限立即终止。临时名和最终名都由服务端 UUID 生成，最终路径格式 `YYYY/MM/<uuid>.pdf`。只有全部校验成功才 `Path.replace()` 原子改名。

- [ ] **Step 4: 实现 PDF 验证与抽取**

- 同时要求 `.pdf`、`application/pdf` 和前 5 字节 `%PDF-`。
- `PdfReader.is_encrypted` 为真即拒绝。
- 页数必须 1..MAX_PDF_PAGES。
- 每页提取异常转换为稳定 `DOCUMENT_PDF_INVALID`，不把异常正文返回 API。
- 清洗后总文本为空返回 `DOCUMENT_PDF_EMPTY`。
- `app/services/pdf_knowledge_importer.py` 改为复用该 extractor 的纯函数，不再把绝对路径写进 SourceDocument 或错误响应。

- [ ] **Step 5: 运行安全测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_storage.py tests/unit/ingestion/test_extractors.py tests/test_pdf_knowledge_importer.py -q
git add app/modules/ingestion/errors.py app/modules/ingestion/storage.py app/modules/ingestion/extractors.py app/services/pdf_knowledge_importer.py tests/unit/ingestion/test_storage.py tests/unit/ingestion/test_extractors.py tests/test_pdf_knowledge_importer.py
git commit -m "feat: secure pdf ingestion storage"
```

---

## Task 7: 实现 ingestion repository 和 CAS 状态机

**Files:**
- Create: `app/modules/ingestion/repository.py`
- Create: `app/modules/ingestion/service.py`
- Create: `tests/unit/ingestion/test_service.py`
- Create: `tests/integration/ingestion/__init__.py`
- Create: `tests/integration/ingestion/test_repository.py`

- [ ] **Step 1: 写完整状态转换失败测试**

```python
@pytest.mark.parametrize(
    ("source", "operation", "target"),
    [
        ("pending", "claim", "running"),
        ("running", "complete", "completed"),
        ("running", "fail", "failed"),
        ("pending", "cancel", "cancelled"),
        ("failed", "retry", "running"),
    ],
)
async def test_allowed_job_transitions(source, operation, target):
    job = await seeded_job(status=source)
    updated = await invoke(repository, operation, job.id)
    assert updated.status == target
```

再逐一验证 completed/running/cancelled 的非法动作返回 `INGESTION_JOB_STATE_CONFLICT`，两个并发 claim 只有一个 rowcount=1。

- [ ] **Step 2: 运行并确认 repository/service 不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_service.py tests/integration/ingestion/test_repository.py -q
```

- [ ] **Step 3: 实现原子状态方法**

Repository 固定提供：

```python
def add_document(self, document: Document) -> None
def add_job(self, job: IngestionJob) -> None
async def list_documents(self, *, offset: int, limit: int, status: str | None) -> tuple[list[Document], int]
async def get_job(self, job_id: UUID) -> IngestionJob | None
async def claim_pending(self, job_id: UUID, now: datetime) -> JobSnapshot | None
async def claim_retry(self, job_id: UUID, now: datetime) -> JobSnapshot | None
async def set_progress(self, job_id: UUID, expected_attempt: int, progress: int) -> bool
async def complete(self, job_id: UUID, expected_attempt: int, now: datetime) -> bool
async def fail(self, job_id: UUID, expected_attempt: int, code: str, message: str, now: datetime) -> bool
async def cancel_pending(self, job_id: UUID, now: datetime) -> IngestionJob | None
```

每个终态写入都包含 `status='running' AND attempt_count=:expected_attempt`，防止旧 worker 覆盖新重试。

- [ ] **Step 4: 实现文档接收短事务**

`accept_pdf()` 先在无 Session 状态调用 UploadStorage；成功后开启短事务创建 document pending 与 job pending。唯一冲突映射 `DOCUMENT_DUPLICATE`/409，并删除本次新保存文件。返回脱离 Session 的 `DocumentAccepted`。

- [ ] **Step 5: 运行 repository 测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_service.py tests/integration/ingestion/test_repository.py -q
git add app/modules/ingestion/repository.py app/modules/ingestion/service.py tests/unit/ingestion/test_service.py tests/integration/ingestion/__init__.py tests/integration/ingestion/test_repository.py
git commit -m "feat: add ingestion job state machine"
```

---

## Task 8: 编排抽取、知识持久化和事务外 Embedding

**Files:**
- Modify: `app/modules/ingestion/service.py`
- Modify: `app/modules/ingestion/repository.py`
- Modify: `app/services/knowledge_extractor.py`
- Modify: `app/services/math_knowledge_importer.py`
- Create: `tests/unit/ingestion/test_pipeline.py`
- Create: `tests/integration/ingestion/test_ingestion_pipeline.py`
- Create: `tests/integration/ingestion/test_retry_idempotency.py`
- Modify: `tests/test_math_knowledge_importer.py`

- [ ] **Step 1: 写端到端事件和失败收口测试**

事件序列固定为：

```python
assert events == [
    "claim.tx.commit",
    "source.read",
    "llm.extract",
    "knowledge.tx.commit",
    "embedding.call",
    "finalize.tx.commit",
]
```

LLM/Embedding 探针执行时 `active_business_sessions == 0`。分别注入 PDF、LLM、Embedding 和数据库写入失败，断言 job/document/item/chunk 状态与错误码一致且错误摘要已脱敏。

- [ ] **Step 2: 运行并确认 pipeline 能力失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion/test_pipeline.py tests/integration/ingestion/test_ingestion_pipeline.py tests/integration/ingestion/test_retry_idempotency.py -q
```

- [ ] **Step 3: 将 LLM 抽取拆成无 JSONL 依赖的纯接口**

`knowledge_extractor.py` 增加：

```python
def extract_knowledge_drafts(text: str, category: str | None = None) -> list[KnowledgeDraft]:
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("text cannot be empty")
    result = chat_json(messages=_build_messages(normalized, category), temperature=0.1)
    return normalize_drafts(result.data, category)
```

旧 `/api/knowledge/extract` 的 `extract_knowledge_records()` 作为预览兼容包装，但 `save=true` 继续 410；ingestion 和新 API 不读取或追加 seed JSONL。

- [ ] **Step 4: 实现可重试 pipeline**

`run_pending(job_id)` 先 claim；`resume_retry(snapshot)` 直接使用 claim_retry 返回的 attempt。核心规则：

1. 读取 job snapshot 和受控 source。
2. 若该 job 已有知识条目，直接加载其 failed/pending chunks，不再次 LLM 抽取。
3. 若无条目，在事务外清洗并抽取 drafts；短事务批量创建 indexing items 和 pending chunks，关联 job/document。
4. 在事务外分批调用 provider；严格校验数量、索引、1024 维有限非零向量。
5. 短事务使用 job id + attempt + chunk id + retrieval_text 快照 CAS 写向量，刷新 item ready、document ready、job completed。
6. 任一步失败都用新 Session 尝试 fail；错误摘要只使用稳定映射文本。

- [ ] **Step 5: 实现重试幂等断言**

第一次 Embedding 失败后记录 item/chunk 数；retry 成功后数量必须相同，LLM 调用次数仍为 1，Embedding 调用次数为 2。并发 retry 只有一个执行者。

```python
assert counts_after_retry == counts_after_failure
assert fake_extractor.calls == 1
assert fake_embedding.calls == 2
assert completed_job.attempt_count == 2
```

- [ ] **Step 6: 运行 ingestion 与既有导入测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/ingestion tests/integration/ingestion tests/test_math_knowledge_importer.py tests/test_pdf_knowledge_importer.py -q
git add app/modules/ingestion/service.py app/modules/ingestion/repository.py app/services/knowledge_extractor.py app/services/math_knowledge_importer.py tests/unit/ingestion/test_pipeline.py tests/integration/ingestion/test_ingestion_pipeline.py tests/integration/ingestion/test_retry_idempotency.py tests/test_math_knowledge_importer.py
git commit -m "feat: persist retryable ingestion pipeline"
```

---

## Task 9: 发布 documents/ingestion API 和后台执行边界

**Files:**
- Create: `app/modules/ingestion/router.py`
- Modify: `app/main.py`
- Create: `tests/api/test_documents.py`
- Create: `tests/api/test_ingestion_jobs.py`
- Modify: `tests/api/test_auth.py`

- [ ] **Step 1: 写 API contract 失败测试**

覆盖匿名 401、普通用户 403、管理员上传 202、列表分页、job 查询、pending cancel、failed retry、completed retry 409、无 CSRF 403。Fake service 记录 BackgroundTasks 执行次数。

```python
accepted = client.post(
    "/api/v1/documents",
    files={"file": ("lesson.pdf", valid_pdf, "application/pdf")},
    data={"category": "代数"},
    headers=admin_csrf_headers,
)
assert accepted.status_code == 202
assert "storage_path" not in accepted.text
assert accepted.json()["job"]["status"] == "pending"
```

- [ ] **Step 2: 运行并确认路由不存在**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api/test_documents.py tests/api/test_ingestion_jobs.py -q
```

- [ ] **Step 3: 实现路由**

- router prefix 固定 `/api/v1`，tags 分别为 documents/ingestion。
- POST documents 使用 `require_admin_csrf`，创建成功后 `background_tasks.add_task(service.run_pending, accepted.job.id)`。
- GET documents/job 使用 `require_admin`；cancel/retry 使用 `require_admin_csrf`。
- retry 先 `await service.claim_retry()`，成功后安排 `resume_retry(snapshot)`；不能先安排任务再改状态。
- BackgroundTasks 捕获的领域失败已由 service 持久化，不把供应商异常重新暴露给上传响应。

- [ ] **Step 4: 注册路由并运行 API 全套**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/api -q
git add app/main.py app/modules/ingestion/router.py tests/api/test_documents.py tests/api/test_ingestion_jobs.py tests/api/test_auth.py
git commit -m "feat: publish document ingestion api"
```

---

## Task 10: 将网页与 PDF CLI 切换到统一 ingestion service

**Files:**
- Modify: `scripts/import_math_knowledge.py`
- Modify: `scripts/import_pdf_knowledge.py`
- Create: `app/modules/ingestion/factory.py`
- Create: `tests/integration/ingestion/test_cli.py`
- Modify: `tests/test_math_knowledge_importer.py`
- Modify: `tests/test_pdf_knowledge_importer.py`
- Modify: `tests/test_runtime_dependency_boundary.py`

- [ ] **Step 1: 写 CLI 不再写 JSONL 的失败测试**

测试 monkeypatch `build_ingestion_service()`，断言脚本只传递 sources/keywords/path/category/requested_by 并调用 service；工作目录中不存在新增 seed/error/text chunk JSONL。

```python
result = subprocess.run(
    [
        sys.executable,
        "-m",
        "scripts.import_pdf_knowledge",
        "--data-dir",
        str(pdf_dir),
        "--requested-by",
        "admin",
    ],
    cwd=PROJECT_ROOT,
    env=environment,
    check=True,
    capture_output=True,
    text=True,
)
assert "Completed jobs:" in result.stdout
assert list(output_dir.rglob("*.jsonl")) == []
```

- [ ] **Step 2: 运行并确认旧脚本仍追加文件**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/ingestion/test_cli.py -q
```

- [ ] **Step 3: 实现共享 factory 和 CLI 参数**

`factory.py` 只组装 session factory、storage、extractor、provider。两个脚本都要求 `--requested-by <admin username>`；通过 repository 验证 active admin，不接受普通用户。

网页脚本保留 sources/keywords/limit/category/delay/max-chunk 参数，创建 web job 并同步等待 service 完成。PDF 脚本保留 data-dir/recursive/max-chunks/category，逐文件调用 `accept_local_pdf()` 和 `run_pending()`。删除 CLI 的 `--output`、`--text-output`、`--append-text-output` 和 JSONL next-step 提示。

- [ ] **Step 4: 证明没有第二套数据库逻辑**

AST 测试断言两个 CLI 不导入 SQLAlchemy model/repository，不包含 `session.execute`、`jsonl` 写入或 `append_records`；只允许通过 factory/service 进入数据库。

- [ ] **Step 5: 运行 CLI/legacy 回归并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/ingestion/test_cli.py tests/test_math_knowledge_importer.py tests/test_pdf_knowledge_importer.py tests/test_runtime_dependency_boundary.py -q
git add app/modules/ingestion/factory.py scripts/import_math_knowledge.py scripts/import_pdf_knowledge.py tests/integration/ingestion/test_cli.py tests/test_math_knowledge_importer.py tests/test_pdf_knowledge_importer.py tests/test_runtime_dependency_boundary.py
git commit -m "refactor: route import cli through ingestion service"
```

---

## Task 11: 完成 M5 隔离、恢复、迁移和运行时验收

**Files:**
- Create: `tests/integration/test_m5_workflow.py`
- Modify: `README.md`
- Create: `docs/baselines/2026-07-31-m5-knowledge-ingestion-management.md`

- [ ] **Step 1: 写真实 PostgreSQL + fake provider 的完整工作流**

流程固定为：admin login -> upload PDF -> background run -> poll completed -> list document -> read created ready knowledge -> stale revision PATCH 409 -> user login -> public read 200/private read 404 -> user write 403 -> admin archive 204 -> RAG search 不再命中 archived。

- [ ] **Step 2: 写失败与重试工作流**

第一次 provider 超时后 job/document/item/chunk 均为 failed 且错误摘要无供应商正文；管理员 retry 后同一 job id、document id、item ids、chunk ids 进入 completed/ready，数量不增加。

- [ ] **Step 3: 运行分层测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/knowledge tests/unit/ingestion -q
.\.venv\Scripts\python.exe -m pytest tests/api -q
.\.venv\Scripts\python.exe -m pytest tests/integration -q
```

Expected: 0 failed、0 unexpected skipped；数据库测试使用与主库物理隔离的 `TEST_DATABASE_URL`。

- [ ] **Step 4: 执行空库迁移回环和 autogenerate 检查**

```powershell
.\.venv\Scripts\alembic.exe downgrade base
.\.venv\Scripts\alembic.exe upgrade head
.\.venv\Scripts\alembic.exe current
.\.venv\Scripts\alembic.exe check
```

Expected: current 为 `0005_create_documents_ingestion_jobs (head)`；check 为 `No new upgrade operations detected.`。测试库回环完成后不能对用户主库执行 downgrade。

- [ ] **Step 5: 运行完整测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Expected: 0 failed、0 unexpected skipped；只允许已有 Starlette TestClient/httpx 弃用 warning，M5 不新增 warning。

- [ ] **Step 6: Compose smoke 和上传卷验证**

使用进程级临时 `SESSION_SECRET`/`ALLOWED_ORIGINS`，不改 `.env`。构建并启动 postgres/mathrag，执行 migration，验证 `/health/live`、`/health/ready`、管理员上传小 PDF、任务 completed，以及容器重建后 document 元数据和上传文件仍存在。结束后：

```powershell
docker compose down
docker compose ps --all
```

Expected: `docker compose ps --all` 无项目容器；不执行 `down -v`，避免删除持久化测试卷。

- [ ] **Step 7: 更新 README 和验收基线**

记录新环境变量、管理员登录/CSRF 上传示例、知识 CRUD、轮询/取消/重试、CLI 新参数、备份需同时包含 PostgreSQL 与 upload_data、M6 前端消费契约和回滚步骤。

- [ ] **Step 8: 提交最终验收**

```powershell
git add README.md docs/baselines/2026-07-31-m5-knowledge-ingestion-management.md tests/integration/test_m5_workflow.py
git commit -m "docs: record m5 acceptance"
git status --short --branch
```

Expected: 分支只保留用户已有的 `tmp/` 未跟踪项，无 M5 未提交文件。

---

## 稳定错误码矩阵

| 场景 | HTTP/API | job error_code | 状态 |
|---|---:|---|---|
| 无/无效 Session | 401 | 不创建 | 不创建 |
| 普通用户写知识/上传/任务管理 | 403 | 不创建 | 不创建 |
| CSRF/Origin 失败 | 403 | 不创建 | 不创建 |
| 知识不存在或用户不可见 | 404 `KNOWLEDGE_NOT_FOUND` | 不适用 | 不变 |
| revision 不一致 | 409 `KNOWLEDGE_REVISION_CONFLICT` | 不适用 | 不变 |
| 重复文档 | 409 `DOCUMENT_DUPLICATE` | 不创建 | 已有资源不变 |
| 文件过大 | 413 `DOCUMENT_TOO_LARGE` | 不创建 | 不创建 |
| MIME/扩展名错误 | 415 `DOCUMENT_MIME_UNSUPPORTED` | 不创建 | 不创建 |
| 路径/文件名非法 | 422 `DOCUMENT_PATH_INVALID` | 不创建 | 不创建 |
| PDF 损坏/加密/页数/空文本 | 422 对应稳定 code | `INGESTION_PDF_INVALID` | job/document failed（已创建时） |
| LLM 鉴权/连接/API 错误 | 上传已 202；job 查询 | `INGESTION_LLM_UNAVAILABLE` | failed |
| LLM 限流 | 上传已 202；job 查询 | `INGESTION_LLM_RATE_LIMITED` | failed |
| LLM/Embedding 超时 | 上传已 202；job 查询 | `INGESTION_UPSTREAM_TIMEOUT` | failed |
| Embedding 无效/不可用 | 上传已 202；job 查询 | `INGESTION_EMBEDDING_UNAVAILABLE` | failed |
| 非法 cancel/retry/重复 worker | 409 `INGESTION_JOB_STATE_CONFLICT` | 不覆盖 | 原状态 |
| PostgreSQL 暂不可用 | 503 `DATABASE_UNAVAILABLE` | 能收口则 `INGESTION_DATABASE_UNAVAILABLE` | failed 或明确未确认 |
| 未知后台异常 | 上传已 202；job 查询 | `INGESTION_INTERNAL_ERROR` | failed |

## 事务与幂等不变量

```text
HTTP upload（无 DB 事务）
  -> 受控 .part + 校验 + 原子改名
  -> tx1: document pending + job pending
  -> 202 + BackgroundTasks
  -> tx2: CAS pending/failed -> running，提交并关闭
  -> PDF/网页读取 + LLM（无业务 Session）
  -> tx3: items indexing + chunks pending，提交并关闭
  -> Embedding（无业务 Session）
  -> tx4: chunks/items/document/job CAS -> ready/completed
```

- 任意 LLM/Embedding/文件上传等待期间无 AsyncSession 和事务存活。
- 同一个 document 只有一个 pdf job；retry 复用同一 job，并通过 attempt_count 隔离旧执行者。
- 已存在 job items 时 retry 不重复抽取，不新增 item/chunk；只重试未完成向量化。
- completed job、ready item 和历史 RAG reference snapshot 不被 retry 覆盖。
- 文件绝对路径只在 storage 内部短暂存在；数据库只存相对路径，公开 schema 连相对路径也不返回。
- 上传数据库写失败时删除本次新文件；数据库记录已提交后不因后台失败删除原文件，以便诊断和重试。
- service/repository 不跨事务返回 ORM；只返回 Pydantic DTO 或 frozen dataclass snapshot。

## 最终验收清单

- [ ] `main@366383b` 的 514 项测试无非预期回归。
- [ ] 空测试库可从 base 升到 `0005_create_documents_ingestion_jobs`，downgrade/upgrade 回环和 `alembic check` 通过。
- [ ] 普通用户只能读取 public+ready，所有管理员写接口均覆盖 Session、角色、Origin 和 CSRF。
- [ ] knowledge create/update/archive 与 revision 冲突有 unit、integration、API 测试。
- [ ] 路径穿越、伪 MIME/扩展名、超限、损坏、加密、超页数、空 PDF 全部拒绝且不遗留临时文件。
- [ ] pending/running/completed/failed/cancelled 和 failed->running 转换全部通过 CAS 测试。
- [ ] LLM/Embedding 等待期间无业务数据库事务；失败状态和错误摘要可解释且不泄密。
- [ ] retry 不重复 item/chunk，不重复 LLM 抽取，不覆盖 completed 结果。
- [ ] Web 与两个 CLI 都调用同一 IngestionService/repository；在线和 CLI 都不追加 seed JSONL。
- [ ] RAG 仍只检索 PostgreSQL 中 public+ready，archived/private/failed 泄漏为 0。
- [ ] Compose 上传卷可持久化，README 包含备份/恢复边界；测试后所有容器停止。
- [ ] `tmp/`、`.env`、用户上传样本和其他用户文件未读取、未修改、未暂存。

## 计划自检

- 架构覆盖：knowledge CRUD、revision 409、documents、ingestion_jobs、安全上传、状态机、错误恢复、重试、短事务、CLI 复用均有对应任务和测试。
- 迁移一致：M4 已占用 0004，本计划唯一 head 是 0005；ORM、migration 和 Alembic metadata 同步。
- 类型一致：Document/Job/Knowledge 公开 DTO 不暴露 ORM；跨事务 snapshot 只携带冻结值；UUID/FK 命名一致。
- 幂等一致：document+job 唯一约束、状态 CAS、attempt_count 和已有 job item 复用共同防止重复。
- 安全一致：管理员依赖、CSRF、受控根目录、UUID 文件名、大小/MIME/PDF 校验和错误脱敏没有绕过路径。
- 数据源一致：PostgreSQL/pgvector 仍是唯一在线知识事实源；JSONL 只保留 M2 历史迁移工件，不接收 M5 新写入。
- 范围一致：没有 Vue、Redis、Celery、OCR、多 Worker、自动任务租约或生产压测。
- 可执行性：每个任务包含失败测试、实现边界、精确命令、预期结果和独立提交点；无待定字段或未分配需求。
