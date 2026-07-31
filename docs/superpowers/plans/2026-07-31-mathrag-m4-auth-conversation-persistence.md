# MathRAG M4 认证、会话与 RAG 持久化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不破坏 M3 pgvector 在线检索的前提下，交付服务端 Session 认证、`admin`/`user` 角色、严格按用户隔离的会话与消息，以及可追溯、可恢复、可幂等重放的 RAG 运行和引用快照。

**Architecture:** 浏览器只持有随机原始 Session Cookie，PostgreSQL 只保存 SHA-256；认证依赖返回脱离 ORM 会话的 principal，任何等待 LLM/Embedding 的阶段都不占用数据库连接。`POST /api/v1/chat` 先在短事务中保存用户消息、pending 助手消息和 running `rag_run`，关闭会话后执行 RAG，再用新短事务保存回答、引用快照和终态。所有会话查询把 `user_id` 写入 SQL 条件，跨用户资源统一按不存在处理。旧 `/api/chat` 在 M4 仅作只读兼容基线，不接入新的持久化链路。

**Tech Stack:** Python 3.11.9、FastAPI 0.140.13、Starlette、Pydantic 2.13.4、SQLAlchemy 2.0.51 asyncio、asyncpg 0.31.0、Alembic 1.18.5、PostgreSQL 18.4、pgvector 0.8.5、`pwdlib[argon2]==0.3.0`、pytest 9.1.1、Docker Compose。

---

## 起点与前置条件

- 开发分支：`codex/m4-auth-conversation-persistence`。
- 分支基点：`main@bcf8e9d`，即 M3 验收提交。
- 合并后的 `main` 已重新执行全量测试：`417 passed, 1 warning in 28.04s`；0 failed、0 skipped。
- Alembic 当前 head：`0003_enforce_vector_readiness`；`alembic check` 输出 `No new upgrade operations detected.`。
- M3 在线检索唯一数据源是 PostgreSQL + pgvector；M4 不恢复 FAISS、JSONL 或 `id_map.json` 在线依赖。
- 测试完成后 Compose 容器必须全部停止；未跟踪的 `tmp/` 属于用户数据，不读取、不修改、不暂存。
- 总体设计依据：`docs/superpowers/specs/2026-07-29-mathrag-project-architecture-design.md` 第 10、11.2、14.1、14.2、15、16、17、18、19、23 节。
- 已接受安全决策：`docs/adr/0001-m0-engineering-baseline.md` 的“认证与会话基线”和“CSRF 与跨域基线”。M4 不另行改用 JWT，也不降低 Cookie 或 CSRF 要求。

## 冻结范围

### M4 包含

- `users`、`user_sessions`、`conversations`、`messages`、`rag_runs`、`rag_references` 六张表及 ORM 模型。
- 为现有 `knowledge_items` 补充 nullable `owner_id` 外键和普通索引；26 条公共遗留知识保持 `owner_id=NULL`。
- `pwdlib` Argon2 密码哈希、服务端随机 Session、Cookie、签名 double-submit CSRF、显式 CORS 来源。
- `POST /api/v1/auth/login`、`POST /api/v1/auth/logout`、`GET /api/v1/auth/me`。
- 会话 CRUD、消息分页、`POST /api/v1/chat`、统一 `/api/v1` 错误包络。
- 管理员角色 dependency 和安全的用户创建 CLI；不增加公开注册接口。
- RAG 执行 DTO、两个短事务、引用快照、失败/取消终态和 `client_request_id` 幂等约束。
- 关闭旧 `/api/knowledge/extract` 的 JSONL 追加能力；仅管理员带 CSRF 可使用 `save=false` 预览，`save=true` 明确拒绝，避免恢复第二事实源。
- 双用户隔离、迁移、API、事务边界、快照与安全回归测试。

### M4 不包含

- Vue 3 登录页、会话页或前端重写；这些属于 M6。
- 知识 CRUD、revision 冲突、文档上传和 ingestion job；这些属于 M5。
- Redis、JWT、OAuth 第三方登录、Celery、HNSW、多 Worker 或微服务。
- 真正可中断的同步 LLM SDK 调用、全链路限流和压测；M4 记录取消终态，M7 完成 I/O 与容量加固。

## 冻结 API 契约

### 认证

```text
POST /api/v1/auth/login
request:  {"username":"alice","password":"用户输入的密码"}
response: {"id":"UUID","username":"alice","email":null,"role":"user","status":"active"}
cookies:  session HttpOnly + csrf non-HttpOnly

POST /api/v1/auth/logout
request:  empty body + session Cookie + csrf Cookie + X-CSRF-Token
response: 204，并删除两个 Cookie

GET /api/v1/auth/me
response: 当前用户资源；无有效 Session 返回 401
```

- 登录不存在、密码错误或哈希无法识别统一返回 `AUTH_INVALID_CREDENTIALS`，不暴露用户名是否存在。
- 禁用用户不能创建或继续使用 Session；当前请求返回 `AUTH_SESSION_INVALID`。
- 生产 Session Cookie 名固定为 `__Host-mathrag_session`，CSRF Cookie 名固定为 `__Host-mathrag_csrf`；开发环境对应 `mathrag_session`、`mathrag_csrf`。
- 生产 Cookie 均为 `Secure; SameSite=Lax; Path=/` 且无 `Domain`；Session Cookie 额外为 `HttpOnly`，CSRF Cookie 必须可由前端读取。

### 会话与消息

```text
GET    /api/v1/conversations?page=1&page_size=20&status=active
POST   /api/v1/conversations
GET    /api/v1/conversations/{conversation_id}
PATCH  /api/v1/conversations/{conversation_id}
DELETE /api/v1/conversations/{conversation_id}
GET    /api/v1/conversations/{conversation_id}/messages?page=1&page_size=50
```

- `POST` 请求体为 `{"title":"新对话"}`，`title` 可省略；服务端默认值为“新对话”。
- `PATCH` 只接受 `title` 和 `status`；`status` 只能是 `active` 或 `archived`。
- `DELETE` 是幂等软删除，把状态改为 `archived` 并返回 204，不物理删除消息和 RAG 审计数据。
- 会话列表按 `updated_at DESC, id DESC`；消息列表按 `created_at ASC, id ASC`。
- 列表响应使用有业务意义的 `items/page/page_size/total`，资源响应不额外套 `data`。
- 不存在和跨用户访问统一返回 `CONVERSATION_NOT_FOUND`/404，避免泄露其他用户资源是否存在。

### 持久化聊天

```json
POST /api/v1/chat
{
  "conversation_id": "UUID",
  "client_request_id": "UUID",
  "question": "导数的几何意义是什么？",
  "top_k": 3
}
```

成功响应保留 M3 的 `question/answer/steps/used_knowledge/related_questions/references/agentic_plan/reasoning_content`，并增加：

```json
{
  "conversation_id": "UUID",
  "question_message_id": "UUID",
  "answer_message_id": "UUID",
  "rag_run_id": "UUID",
  "client_request_id": "UUID"
}
```

- history 不再由客户端提交；服务端从同一会话读取最近 8 条 `completed` 的 user/assistant 消息，并在关闭数据库会话后传给 RAG。
- `(conversation_id, client_request_id)` 唯一。重复请求已 completed 时返回同一持久化结果；running 时返回 `RAG_REQUEST_IN_PROGRESS`/409；failed 或 cancelled 时返回相应已落盘错误，不重复调用外部服务。
- archived 会话拒绝新问题并返回 `CONVERSATION_ARCHIVED`/409。
- `/api/v1/chat` 的所有状态变更都要求 CSRF；旧 `/api/chat` 保持 M3 契约，仅在 development 作为兼容基线，添加 `deprecated=True` 和退役响应头。staging/production 固定返回 410，不保留匿名调用旁路；删除门固定为 M6 Vue 切换提交，不允许新前端继续调用旧接口。

### 统一错误包络

```json
{
  "error": {
    "code": "CONVERSATION_NOT_FOUND",
    "message": "会话不存在。",
    "request_id": "请求关联 ID",
    "details": {}
  }
}
```

- `/api/v1` 的领域错误、HTTP 错误、Pydantic 请求校验和未处理异常均使用该结构。
- `details` 不包含密码、Cookie、Session 原始令牌、哈希、数据库 URL、外部 API 原始正文或完整聊天正文。
- 旧 `/api/*` 在兼容期保持现有 `{"detail":"..."}` 错误结构，避免破坏静态原型。

## 数据库约束

| 表 | 关键约束与索引 |
|---|---|
| `users` | username 规范化为小写 ASCII，唯一；email nullable unique；role/status CHECK |
| `user_sessions` | `token_hash BYTEA UNIQUE`；`user_id` 外键；expiry/revocation 时间约束；用户活跃 Session 索引 |
| `conversations` | `user_id` 外键；status CHECK；`(user_id, updated_at, id)` 索引 |
| `messages` | role/status CHECK；`UNIQUE(conversation_id,id)`；`(conversation_id, created_at, id)` 索引；conversation 级联删除 |
| `rag_runs` | `(conversation_id, client_request_id)` UNIQUE；top_k、status、latency CHECK；问题/答案使用 `(conversation_id,message_id)` 复合外键，禁止跨会话挂接 |
| `rag_references` | `(rag_run_id, rank)` 主键；`(rag_run_id, chunk_id)` UNIQUE；chunk 删除时 `SET NULL`，快照仍保留 |
| `knowledge_items` | 新增 nullable `owner_id -> users.id ON DELETE SET NULL` 和普通索引 |

- UUID 由应用生成，时间使用 `UTCDateTime()`/`TIMESTAMPTZ`。
- Repository 不调用 `begin/commit/rollback/close`；Service 或顶层用例决定事务边界。
- 数据库 CHECK 负责拒绝非法枚举和负 latency；Pydantic 校验不能替代数据库约束。
- `rag_references.snapshot` 至少保存 `source_id/category/title/keywords/content/example/steps/difficulty/answer_context/retrieval_text/metadata`，不得依赖后来变化的知识表重建历史响应。
- `rag_references` 是关联审计表，明确作为“通用 UUID 主键”规则的例外，以 `(rag_run_id, rank)` 作为复合主键；`rank >= 1`，score 由 service 校验为有限浮点数。
- 所有使用 Core/ORM `update()` 的路径显式写 `updated_at=now`，不依赖 `onupdate=func.now()` 自动生效。

## 官方实现依据

- FastAPI 当前密码示例和 pwdlib 官方 API 都使用 `PasswordHash.recommended()`、`hash()`、`verify()`；M0 ADR 已锁定 `pwdlib[argon2]==0.3.0`：https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/ ，https://frankie567.github.io/pwdlib/reference/pwdlib/
- Starlette `Response.set_cookie()` 明确提供 `secure`、`httponly`、`samesite`、`path` 等参数：https://www.starlette.io/responses/
- FastAPI 支持通过 `Response` 参数设置 Cookie并继续应用 response model：https://fastapi.tiangolo.com/advanced/response-cookies/
- OWASP 推荐把签名 double-submit token 显式绑定到已认证 Session，并以 Origin/Referer 校验作纵深防御：https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html
- FastAPI 允许覆盖 `RequestValidationError` handler，用于稳定 `/api/v1` 错误包络：https://fastapi.tiangolo.com/tutorial/handling-errors/
- SQLAlchemy 2.0 明确要求一个并发 task 使用一个 AsyncSession，并建议保持事务短小、由外层划定范围：https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- PostgreSQL UNIQUE 约束会建立唯一 B-tree 索引；M4 的幂等键使用数据库唯一约束而不是进程内锁：https://www.postgresql.org/docs/current/ddl-constraints.html

## 文件结构

| 文件 | 操作 | 单一职责 |
|---|---|---|
| `requirements.txt` / `requirements.lock.txt` | Modify | 锁定 `pwdlib[argon2]==0.3.0` 及 Argon2 传递依赖 |
| `.env.example` | Modify | Session、TTL、显式 origins 示例 |
| `app/core/config.py` | Modify | 安全配置解析、Cookie 派生属性和生产校验 |
| `app/core/errors.py` | Modify | 稳定 `AppError` 领域错误 |
| `app/core/exception_handlers.py` | Create | `/api/v1` 错误包络和旧 API 兼容处理 |
| `app/core/middleware.py` | Modify | 保持 request ID，并提供来源校验所需请求状态 |
| `app/modules/users/models.py` | Create | User ORM |
| `app/modules/users/schemas.py` | Create | 用户公开 DTO，不含 password/session hash |
| `app/modules/users/repository.py` | Create | 用户读取、写入和状态更新 |
| `app/modules/users/service.py` | Create | 用户创建、禁用、改密和会话撤销用例 |
| `app/modules/auth/models.py` | Create | UserSession ORM |
| `app/modules/auth/security.py` | Create | Argon2、令牌哈希、HMAC CSRF、常量时间比较 |
| `app/modules/auth/repository.py` | Create | Session 查询、撤销和 last_seen 更新 |
| `app/modules/auth/service.py` | Create | 登录、principal 解析、退出 |
| `app/modules/auth/dependencies.py` | Create | current user、admin、CSRF、Origin/Referer dependencies |
| `app/modules/auth/schemas.py` | Create | Login request 和公开响应 |
| `app/modules/auth/router.py` | Create | `/api/v1/auth` 路由与 Cookie 生命周期 |
| `app/modules/conversations/models.py` | Create | Conversation、Message ORM |
| `app/modules/conversations/schemas.py` | Create | CRUD、分页和消息 DTO |
| `app/modules/conversations/repository.py` | Create | owner-scoped SQL 和历史读取 |
| `app/modules/conversations/service.py` | Create | 会话用例、标题和归档规则 |
| `app/modules/conversations/router.py` | Create | `/api/v1/conversations` 路由 |
| `app/modules/rag/models.py` | Create | RAGRun、RAGReference ORM |
| `app/modules/rag/schemas.py` | Create | v1 Chat request/response |
| `app/modules/rag/repository.py` | Create | RAG 状态 CAS、幂等查询和引用写入 |
| `app/modules/rag/service.py` | Create | 两短事务持久化编排与失败/取消收口 |
| `app/modules/rag/router.py` | Create | `POST /api/v1/chat` |
| `app/modules/rag/execution.py` | Create | 脱离 ORM 的 RAGExecution 和 Pipeline protocol |
| `app/modules/knowledge/models.py` | Modify | KnowledgeItem 增加 nullable owner_id |
| `app/modules/knowledge/search_service.py` | Modify | 公开只读 `embedding_model` 属性 |
| `app/services/rag_pipeline.py` | Modify | 增加保留数据库 chunk UUID 的 `execute()`，兼容 `chat()` |
| `app/api/chat.py` | Modify | 标记旧端点 deprecated 并写退役头，不持久化 |
| `app/api/knowledge.py` / `app/schemas/knowledge.py` | Modify | 旧抽取端点只允许 preview，拒绝 JSONL 追加并清理上游错误泄漏 |
| `app/main.py` | Modify | 注册 v1 routers、异常 handler、显式 CORS |
| `alembic/env.py` | Modify | 注册 M4 模型 metadata |
| `alembic/versions/0004_create_identity_conversation_rag_tables.py` | Create | M4 全部 schema 变更和可逆 downgrade |
| `scripts/create_user.py` | Create | getpass 输入密码的 admin/user 创建 CLI |
| `tests/unit/auth/` | Create | 哈希、Session、CSRF、依赖和服务测试 |
| `tests/unit/conversations/` | Create | owner scope、归档和分页测试 |
| `tests/unit/rag/` | Create | 执行 DTO、短事务、幂等、失败和取消测试 |
| `tests/api/test_errors_v1.py` | Create | 错误包络、request ID 和敏感信息防泄漏 |
| `tests/api/test_auth.py` | Create | login/logout/me、Cookie、CSRF、角色测试 |
| `tests/api/test_conversations.py` | Create | CRUD、分页、跨用户 404 |
| `tests/api/test_chat_v1.py` | Create | v1 成功、错误映射和兼容契约 |
| `tests/test_knowledge_api.py` | Modify | 锁定 preview-only 和禁止 JSONL 写入 |
| `tests/integration/test_m4_migration_schema.py` | Create | 表、约束、索引和 downgrade/upgrade |
| `tests/integration/test_user_isolation.py` | Create | 真实 PostgreSQL 双用户隔离 |
| `tests/integration/test_rag_persistence.py` | Create | 两事务、引用快照和幂等终态 |
| `tests/test_runtime_configuration.py` | Modify | 生产 Session/CORS 配置失败测试 |
| `tests/test_runtime_dependency_boundary.py` | Modify | 禁止原始 token 持久化及外部等待持有 Session |
| `README.md` | Modify | 用户创建、登录、CSRF、v1 API 与兼容退役说明 |
| `docs/baselines/2026-07-31-m4-auth-conversation-persistence.md` | Create | M4 验收证据与回滚边界 |

---

## Task 1: 锁定认证依赖、配置和 `/api/v1` 错误包络

**Files:**

- Modify: `requirements.txt`
- Modify: `requirements.lock.txt`
- Modify: `.env.example`
- Modify: `app/core/config.py`
- Modify: `app/core/errors.py`
- Create: `app/core/exception_handlers.py`
- Modify: `app/main.py`
- Create: `tests/unit/auth/__init__.py`
- Create: `tests/unit/auth/test_config.py`
- Create: `tests/api/test_errors_v1.py`
- Modify: `tests/test_runtime_configuration.py`

- [ ] **Step 1: 写失败测试**

覆盖以下行为：

```python
def test_production_requires_session_secret_and_explicit_origins() -> None:
    with pytest.raises(ConfigurationError):
        Settings(APP_ENV="production", SESSION_SECRET="", ALLOWED_ORIGINS=())


def test_v1_validation_error_uses_stable_envelope_and_request_id() -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": ""},
        headers={"X-Request-ID": "m4-validation-001"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"
    assert response.json()["error"]["request_id"] == "m4-validation-001"
    assert "input" not in response.text
```

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth\test_config.py tests\api\test_errors_v1.py tests\test_runtime_configuration.py -q
```

Expected: FAIL，缺少 Session 配置、v1 handler 或 auth route；既有测试仍可收集。

- [ ] **Step 3: 增加固定依赖和配置**

`requirements.txt` 增加精确直接依赖：

```text
pwdlib[argon2]==0.3.0
```

`Settings` 增加并校验：

```python
SESSION_SECRET: str
SESSION_TTL_SECONDS: int = 604800
ALLOWED_ORIGINS: tuple[str, ...]

@property
def session_cookie_name(self) -> str:
    return "mathrag_session" if self.APP_ENV == "development" else "__Host-mathrag_session"

@property
def csrf_cookie_name(self) -> str:
    return "mathrag_csrf" if self.APP_ENV == "development" else "__Host-mathrag_csrf"
```

要求 `SESSION_TTL_SECONDS > 0`；staging/production 的 `SESSION_SECRET` 至少 32 个 UTF-8 字节，`ALLOWED_ORIGINS` 非空且不得包含 `*`。development 默认只允许 `http://127.0.0.1:8000` 和 `http://localhost:8000`。

- [ ] **Step 4: 建立稳定异常和 handler**

公共异常签名固定为：

```python
class AppError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(code)
```

`install_exception_handlers(app)` 必须：

- 处理 `AppError`、`HTTPException`、`RequestValidationError` 和未处理 `Exception`。
- 只对 `/api/v1` 输出新包络；旧 `/api` 保持 `detail` 兼容。
- 从 `request.state.request_id` 读取关联 ID。
- 校验错误 details 只保留 `loc/type/msg`，使用 `errors(include_input=False)`。
- 未处理异常固定映射为 `INTERNAL_ERROR`/500，不回显异常文本。

- [ ] **Step 5: 收紧 CORS 并重建锁文件**

`app/main.py` 的 `allow_origins` 改为 `list(settings.ALLOWED_ORIGINS)`，仍允许 credentials；不再出现 `allow_origins=["*"]`。

```powershell
uv pip compile requirements.txt --output-file requirements.lock.txt
.\.venv\Scripts\python.exe -m pytest tests\unit\auth\test_config.py tests\api\test_errors_v1.py tests\test_runtime_configuration.py -q
rg -n 'allow_origins=\["\*"\]|allow_origins=\[' app\main.py
```

Expected: PASS；`rg` 只匹配从 settings 读取的显式列表，不匹配通配符。

- [ ] **Step 6: 提交**

```powershell
git add requirements.txt requirements.lock.txt .env.example app/core/config.py app/core/errors.py app/core/exception_handlers.py app/main.py tests/unit/auth tests/api/test_errors_v1.py tests/test_runtime_configuration.py
git commit -m "feat: establish m4 security configuration"
```

## Task 2: 创建 M4 schema、ORM 模型和迁移回环

**Files:**

- Create: `app/modules/users/__init__.py`
- Create: `app/modules/users/models.py`
- Create: `app/modules/auth/__init__.py`
- Create: `app/modules/auth/models.py`
- Create: `app/modules/conversations/__init__.py`
- Create: `app/modules/conversations/models.py`
- Create: `app/modules/rag/__init__.py`
- Create: `app/modules/rag/models.py`
- Modify: `app/modules/knowledge/models.py`
- Modify: `alembic/env.py`
- Create: `alembic/versions/0004_create_identity_conversation_rag_tables.py`
- Create: `tests/integration/test_m4_migration_schema.py`
- Modify: `tests/integration/test_migrations.py`
- Modify: `tests/unit/knowledge/test_models.py`

- [ ] **Step 1: 写 schema 失败测试**

测试从专用 `TEST_DATABASE_URL` 执行 `upgrade head` 后检查：

- 六张新表存在，`knowledge_items.owner_id` 存在且 nullable。
- 所有 CHECK、FK、UNIQUE 和普通索引名称与本计划“数据库约束”一致。
- `rag_references.chunk_id` 为 nullable，删除知识 chunk 后 snapshot 行仍存在且 `chunk_id IS NULL`。
- 把其他 conversation 的 message ID 写入 rag_run 时，复合 FK 必须拒绝。
- duplicate `(conversation_id, client_request_id)` 被数据库拒绝。
- downgrade 到 `0003_enforce_vector_readiness` 后只移除 M4 schema，再 upgrade head 可恢复。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_m4_migration_schema.py tests\integration\test_migrations.py tests\unit\knowledge\test_models.py -q
```

Expected: FAIL，head 仍为 0003，M4 表和 owner_id 不存在。

- [ ] **Step 3: 实现 ORM 模型**

模型枚举固定使用字符串 CHECK，不引入 PostgreSQL enum type。核心字段：

```python
class User(Base):
    id: Mapped[UUID]
    username: Mapped[str]
    email: Mapped[str | None]
    password_hash: Mapped[str]
    role: Mapped[str]
    status: Mapped[str]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class UserSession(Base):
    id: Mapped[UUID]
    user_id: Mapped[UUID]
    token_hash: Mapped[bytes]
    expires_at: Mapped[datetime]
    revoked_at: Mapped[datetime | None]
    created_at: Mapped[datetime]
    last_seen_at: Mapped[datetime]


class Conversation(Base):
    id: Mapped[UUID]
    user_id: Mapped[UUID]
    title: Mapped[str]
    status: Mapped[str]
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]


class Message(Base):
    id: Mapped[UUID]
    conversation_id: Mapped[UUID]
    role: Mapped[str]
    content: Mapped[str]
    status: Mapped[str]
    model_metadata: Mapped[dict[str, object]]
    created_at: Mapped[datetime]
```

`RAGRun` 额外包含 `client_request_id`，`RAGReference` 使用 `(rag_run_id, rank)` 复合主键。`Message` 增加 `(conversation_id, id)` UNIQUE，RAGRun 的问题和答案分别通过复合 FK 引用同一 conversation 下的 Message。不要在跨模块模型上建立不必要的 eager relationship；若声明 question/answer relationship，必须分别指定 `foreign_keys` 以消除歧义。Repository 通过显式 join/select 加载所需数据。

- [ ] **Step 4: 编写 0004 migration**

升级顺序固定为 users → user_sessions → conversations → messages → rag_runs → rag_references → knowledge owner FK/index。降级严格逆序，并在删除 users 前先删除 knowledge owner FK。

`rag_runs.answer_message_id` nullable；`latency_ms` nullable 且非负；`error_code` nullable；`status` 为 running/completed/failed/cancelled。`rag_references.snapshot` 使用 JSONB，默认不得为空对象，由 service 写完整快照。

- [ ] **Step 5: 注册 metadata 并验证迁移**

```powershell
$env:DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
docker compose up -d postgres
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic check
.\.venv\Scripts\python.exe -m pytest tests\integration\test_m4_migration_schema.py tests\integration\test_migrations.py tests\unit\knowledge\test_models.py -q
```

Expected: current 为 `0004_create_identity_conversation_rag_tables (head)`；`alembic check` 无新增操作；测试 PASS。

- [ ] **Step 6: 提交**

```powershell
git add app/modules/users app/modules/auth/models.py app/modules/auth/__init__.py app/modules/conversations app/modules/rag/models.py app/modules/rag/__init__.py app/modules/knowledge/models.py alembic/env.py alembic/versions/0004_create_identity_conversation_rag_tables.py tests/integration/test_m4_migration_schema.py tests/integration/test_migrations.py tests/unit/knowledge/test_models.py
git commit -m "feat: add identity conversation and rag schema"
```

## Task 3: 实现用户领域、Argon2 和安全用户创建 CLI

**Files:**

- Create: `app/modules/users/schemas.py`
- Create: `app/modules/users/repository.py`
- Create: `app/modules/users/service.py`
- Create: `app/modules/auth/security.py`
- Create: `scripts/create_user.py`
- Create: `tests/unit/auth/test_security.py`
- Create: `tests/unit/auth/test_user_service.py`
- Create: `tests/integration/test_user_repository.py`
- Create: `tests/integration/test_create_user_cli.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- username 只允许 3 至 64 位小写 ASCII 字母、数字、点、下划线和连字符；service 统一 `strip().lower()`。
- password 为 12 至 128 个字符，hash 后不含明文，每次独立盐产生不同 hash。
- `verify_password` 正确识别成功、错误密码和未知 hash，错误不回显 password/hash。
- 重复 username/email 返回稳定领域冲突；disabled 和 password reset 会在同一事务撤销该用户全部 Session。
- CLI 只通过 `getpass.getpass()` 读取密码，不接受 `--password`，stdout 只输出用户 UUID、username、role 和稳定状态。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth\test_security.py tests\unit\auth\test_user_service.py tests\integration\test_user_repository.py tests\integration\test_create_user_cli.py -q
```

Expected: FAIL，用户 service、安全函数和 CLI 尚不存在。

- [ ] **Step 3: 实现安全原语**

`app/modules/auth/security.py` 公共接口固定为：

```python
SESSION_TOKEN_BYTES = 32

async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_password_hash.hash, password)


async def verify_password(password: str, encoded_hash: str) -> bool:
    return await asyncio.to_thread(_safe_verify, password, encoded_hash)


def generate_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def issue_csrf_token(session_hash: bytes, secret: str) -> str:
    nonce = secrets.token_urlsafe(32)
    signature = hmac.new(secret.encode("utf-8"), session_hash + b"." + nonce.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{nonce}.{signature}"


def verify_csrf_token(token: str, session_hash: bytes, secret: str) -> bool:
    try:
        nonce, supplied_signature = token.split(".", 1)
    except ValueError:
        return False
    expected_signature = hmac.new(
        secret.encode("utf-8"),
        session_hash + b"." + nonce.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)
```

`verify_csrf_token` 不生成新 nonce；它拆分已有 token、重算签名并常量时间比较。任何解析异常只返回 False，不回显 token。

- [ ] **Step 4: 实现 UserRepository 和 UserService**

Repository 只提供 `get_by_username`、`get_by_id`、`add`、`email_exists`、`set_status`、`set_password_hash`；Session 撤销由 AuthRepository 在同一 service 用例中执行。UserService 接受现有 session，并由调用方 `async with session.begin()` 划定事务。

公开 `UserRead` 仅包含 `id/username/email/role/status/created_at/updated_at`，永不包含 `password_hash`。

- [ ] **Step 5: 实现 CLI**

```powershell
.\.venv\Scripts\python.exe -m scripts.create_user --username admin --role admin --email admin@example.local
```

CLI 两次读取密码并比较；数据库写入使用 `get_session_factory()` 和单个短事务。重复用户返回退出码 2，输入错误返回 2，数据库错误返回 1；输出和日志不得包含密码、hash 或数据库 URL。

- [ ] **Step 6: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth\test_security.py tests\unit\auth\test_user_service.py tests\integration\test_user_repository.py tests\integration\test_create_user_cli.py -q
git add app/modules/users app/modules/auth/security.py scripts/create_user.py tests/unit/auth tests/integration/test_user_repository.py tests/integration/test_create_user_cli.py
git commit -m "feat: add secure user management foundation"
```

Expected: PASS；CLI 测试使用 mock getpass，不在命令行传真实密码。

## Task 4: 实现服务端 Session、Cookie、CSRF 和认证 API

**Files:**

- Create: `app/modules/auth/schemas.py`
- Create: `app/modules/auth/repository.py`
- Create: `app/modules/auth/service.py`
- Create: `app/modules/auth/dependencies.py`
- Create: `app/modules/auth/router.py`
- Modify: `app/main.py`
- Create: `tests/unit/auth/test_session_service.py`
- Create: `tests/unit/auth/test_dependencies.py`
- Create: `tests/api/test_auth.py`
- Create: `tests/integration/test_auth_sessions.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 登录成功只把 raw token 放入响应 Cookie，数据库只出现 32-byte SHA-256。
- 未知用户和错误密码使用同一个 401 code/message，并执行 dummy Argon2 verify 防止明显时序枚举。
- Session 过期、撤销、用户 disabled 均返回 401；有效 Session 解析为不可变 `AuthenticatedPrincipal`。
- production Cookie 的 `__Host-`、Secure、HttpOnly、SameSite、Path 和无 Domain 属性精确匹配 ADR。
- 登录验证 Origin/Referer；logout 同时验证 header/cookie 的 CSRF token、HMAC Session 绑定和来源。
- `GET /me` 不需要 CSRF；`POST /logout` 缺少或串用另一 Session 的 CSRF token 返回 403。
- `require_admin` 接受 admin，普通 user 返回 `AUTH_FORBIDDEN`/403。
- 现存 `POST /api/knowledge/extract` 作为 M4 的管理员 dependency 落地验证：未登录 401，普通用户 403，管理员仍须通过 CSRF；`save=true` 始终在外部调用前拒绝。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth\test_session_service.py tests\unit\auth\test_dependencies.py tests\api\test_auth.py tests\integration\test_auth_sessions.py -q
```

Expected: FAIL，auth repository/service/router 尚未实现。

- [ ] **Step 3: 实现短生命周期 principal 解析**

```python
@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: UUID
    session_id: UUID
    username: str
    role: Literal["admin", "user"]
    session_token_hash: bytes
```

`get_current_principal` 自行从 `get_session_factory()` 创建短会话，查询结束即关闭，不把 ORM User/UserSession 返回给路由。`last_seen_at` 只在距上次写入至少 5 分钟时更新，更新在同一短事务完成。该 dependency 返回后不得存在活动数据库 Session。

- [ ] **Step 4: 实现认证 service**

核心公开签名：

```python
class AuthService:
    async def login(self, username: str, password: str, now: datetime) -> IssuedSession:
        raise NotImplementedError

    async def resolve(self, raw_token: str, now: datetime) -> AuthenticatedPrincipal:
        raise NotImplementedError

    async def logout(self, session_id: UUID, now: datetime) -> None:
        raise NotImplementedError
```

`IssuedSession` 可以暂存 raw token 和 CSRF token，但不可被 ORM、日志或 exception details 引用。`AuthRepository.find_active_by_hash()` 必须在 SQL 中同时过滤 token hash、`revoked_at IS NULL`、`expires_at > now` 并 join active User。

- [ ] **Step 5: 实现来源和 CSRF dependency**

- unsafe methods 为 POST/PUT/PATCH/DELETE。
- login 只验证 `Origin`，缺失时验证 `Referer`；URL 比较精确到 scheme/host/port。
- 已认证 unsafe v1 route 先解析 principal，再验证 CSRF Cookie 与 `X-CSRF-Token` 完全相同，随后验证 HMAC 和来源。
- 不信任 `Host` 头推导允许来源；只使用 `settings.ALLOWED_ORIGINS`。

- [ ] **Step 6: 实现路由和 Cookie 生命周期**

登录事务成功后设置 Session/CSRF 两个 Cookie；若设置响应失败，不记录 raw token。logout 即使 Session 已撤销也幂等返回 204并删除 Cookie。删除 Cookie 使用与设置相同的 key/path/domain。

- [ ] **Step 7: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth tests\api\test_auth.py tests\integration\test_auth_sessions.py -q
git add app/modules/auth app/main.py tests/unit/auth tests/api/test_auth.py tests/integration/test_auth_sessions.py
git commit -m "feat: add server session authentication"
```

Expected: PASS；响应、异常和测试输出中都没有 raw token/hash/password marker。

## Task 5: 实现 owner-scoped 会话、消息和 REST API

**Files:**

- Create: `app/modules/conversations/errors.py`
- Create: `app/modules/conversations/schemas.py`
- Create: `app/modules/conversations/repository.py`
- Create: `app/modules/conversations/service.py`
- Create: `app/modules/conversations/router.py`
- Modify: `app/main.py`
- Create: `tests/unit/conversations/__init__.py`
- Create: `tests/unit/conversations/test_schemas.py`
- Create: `tests/unit/conversations/test_repository.py`
- Create: `tests/unit/conversations/test_service.py`
- Create: `tests/api/test_conversations.py`
- Create: `tests/integration/test_conversation_repository.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- 创建、读取、改名、archive/unarchive、幂等 DELETE 和 page/page_size 边界。
- Repository 的单资源查询、更新和删除 SQL 必须同时包含 `Conversation.id` 与 `Conversation.user_id`。
- 用户 A 对用户 B 会话的 GET/PATCH/DELETE/messages 全部为相同 404 包络。
- messages 只从 owner-scoped Conversation join 进入，不能先按 message ID 查出后再用 Python 判断 owner。
- 同一时间戳下按 UUID 稳定排序；page_size 只允许 1 至 100，消息默认 50，会话默认 20。
- archived 会话仍可读取历史，但不能由 ChatService 写新消息。
- Repository AST 检查禁止调用 `begin/commit/rollback/close` 或直接管理注入 Session。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\conversations tests\api\test_conversations.py tests\integration\test_conversation_repository.py -q
```

Expected: FAIL，schema、repository、service 和 router 尚不存在。

- [ ] **Step 3: 实现 schema 和领域规则**

公开 DTO：

```python
class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=255)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    status: Literal["active", "archived"] | None = None


class ConversationRead(BaseModel):
    id: UUID
    title: str
    status: Literal["active", "archived"]
    created_at: datetime
    updated_at: datetime


class MessageRead(BaseModel):
    id: UUID
    conversation_id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    status: Literal["pending", "completed", "failed"]
    model_metadata: dict[str, object]
    created_at: datetime
```

所有标题执行 `" ".join(value.split())`，结果不能为空。API 不返回 `user_id`；归属由身份上下文决定。

- [ ] **Step 4: 实现 Repository**

核心查询签名固定为：

```python
async def get_owned(self, conversation_id: UUID, user_id: UUID) -> Conversation | None:
    raise NotImplementedError

async def list_owned(
    self,
    user_id: UUID,
    *,
    status: str,
    offset: int,
    limit: int,
) -> tuple[list[Conversation], int]:
    raise NotImplementedError

async def list_owned_messages(
    self,
    conversation_id: UUID,
    user_id: UUID,
    *,
    offset: int,
    limit: int,
) -> tuple[list[Message], int]:
    raise NotImplementedError
```

写操作先使用 owner 条件锁定或 UPDATE with WHERE；受影响行为 0 时统一抛 `ConversationNotFoundError`。不要把用户 B 的记录加载进 Python 后再判定。

- [ ] **Step 5: 实现 Service 和 router**

Router 只做 HTTP schema 转换，通过 `AuthenticatedPrincipal` 注入 user_id；POST/PATCH/DELETE 使用 `require_csrf`。Service 使用请求级短 session/transaction，不调用外部服务。

`DELETE` 重复调用同一已 archived 会话仍返回 204；真正不存在或跨用户为 404。PATCH 空对象返回 `REQUEST_VALIDATION_FAILED`/422，不产生无意义 UPDATE。

- [ ] **Step 6: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\conversations tests\api\test_conversations.py tests\integration\test_conversation_repository.py -q
git add app/modules/conversations app/main.py tests/unit/conversations tests/api/test_conversations.py tests/integration/test_conversation_repository.py
git commit -m "feat: add isolated conversation api"
```

Expected: PASS；双用户 API 测试中不存在任何 B 的 title、message ID 或正文泄漏。

## Task 6: 将 M3 RAG 改为可持久化的纯执行结果

**Files:**

- Create: `app/modules/rag/execution.py`
- Modify: `app/modules/knowledge/search_service.py`
- Modify: `app/services/rag_pipeline.py`
- Modify: `app/schemas/chat.py`
- Modify: `tests/test_agentic_rag.py`
- Modify: `tests/test_chat_api.py`
- Create: `tests/unit/rag/__init__.py`
- Create: `tests/unit/rag/test_execution.py`

- [ ] **Step 1: 写失败测试**

覆盖：

- `RAGPipeline.execute()` 返回不可变 `RAGExecution`，内部 references 保留真实 `KnowledgeSearchHit.database_chunk_id`。
- 公开响应仍把 `ReferenceItem.chunk_id` 映射为既有 legacy chunk id，旧 `/api/chat` JSON 精确兼容。
- execution 保存最终使用的 strategy、去重后的 retrieval queries、top_k、LLM model、Embedding model 和安全 model metadata。
- 客户端 history 仍只由 legacy `chat()` 接受；新的执行层只消费 service 传入的受信历史。
- 当前问题不同时出现在 history 和独立 question 中。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\rag\test_execution.py tests\test_agentic_rag.py tests\test_chat_api.py -q
```

Expected: FAIL，当前管道只有公开 dict，不保留数据库 chunk UUID。

- [ ] **Step 3: 定义执行 DTO 和协议**

```python
@dataclass(frozen=True)
class RAGExecution:
    question: str
    answer: str
    steps: tuple[str, ...]
    used_knowledge: tuple[str, ...]
    related_questions: tuple[str, ...]
    hits: tuple[KnowledgeSearchHit, ...]
    strategy: str
    retrieval_queries: tuple[str, ...]
    top_k: int
    llm_model: str
    embedding_model: str
    reasoning_content: str | None
    model_metadata: dict[str, object]

    def to_public_response(self) -> dict[str, object]:
        raise NotImplementedError


class RAGExecutor(Protocol):
    async def execute(
        self,
        *,
        question: str,
        history: Sequence[dict[str, str]],
        top_k: int | None,
    ) -> RAGExecution:
        raise NotImplementedError
```

`model_metadata` 只允许稳定的模型名、finish reason 和数值 token usage；不保存 raw provider response、prompt、API key 或完整 reasoning。

- [ ] **Step 4: 重构管道但保留兼容 facade**

- `RAGPipeline.execute()` 执行原规划、检索和回答逻辑并返回 `RAGExecution`。
- `RAGPipeline.chat()` 仅调用 `execute().to_public_response()`。
- `chat_with_rag()` 的签名和返回 dict 不变。
- `KnowledgeSearchService.embedding_model` 只读返回 Provider model。
- `RAGExecution.to_reference_snapshots()` 使用 hit 中的数据库 UUID 建立内部持久化载荷，不从公开 `ReferenceItem.chunk_id` 反查数据库。

- [ ] **Step 5: 验证 legacy 零回归并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\rag\test_execution.py tests\test_agentic_rag.py tests\test_chat_api.py -q
git add app/modules/rag/execution.py app/modules/knowledge/search_service.py app/services/rag_pipeline.py app/schemas/chat.py tests/unit/rag tests/test_agentic_rag.py tests/test_chat_api.py
git commit -m "refactor: expose persistable rag execution"
```

Expected: PASS；旧接口的请求、响应和固定错误状态码不变。

## Task 7: 实现两短事务 RAG 持久化和幂等状态机

**Files:**

- Create: `app/modules/rag/errors.py`
- Create: `app/modules/rag/repository.py`
- Create: `app/modules/rag/service.py`
- Create: `tests/unit/rag/test_repository.py`
- Create: `tests/unit/rag/test_service.py`
- Create: `tests/integration/test_rag_persistence.py`
- Modify: `tests/test_runtime_dependency_boundary.py`

- [ ] **Step 1: 写失败测试**

使用 fake session factory 和 fake RAGExecutor 记录事件顺序，精确断言：

```text
session-1-open
transaction-1-begin
owned-conversation-and-history-read
question-assistant-run-written
transaction-1-commit
session-1-close
rag-execute
session-2-open
transaction-2-begin
answer-references-run-finalized
transaction-2-commit
session-2-close
```

另外覆盖：

- tx1 写入 user/completed、assistant/pending、rag_run/running；assistant content 初始为空字符串。
- 成功 tx2 把 assistant 改为 completed，写完整快照，把 run 改为 completed 并记录非负 latency。
- Embedding、LLM timeout、rate limit、provider error 映射稳定 `error_code`，assistant 为 failed 且正文只含通用说明。
- `asyncio.CancelledError` 重新抛出前使用独立短事务写 assistant/failed 和 rag_run/cancelled。
- 状态更新使用 `WHERE status='running'` 或 `WHERE status='pending'` CAS；rowcount 不为 1 视为并发冲突。
- tx2 失败后收口逻辑使用第三个新 Session 尝试标记 failed，主体异常优先；数据库持续不可用时不伪称已持久化终态。
- 外部执行开始时所有先前 Session 均已关闭。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\rag\test_repository.py tests\unit\rag\test_service.py tests\integration\test_rag_persistence.py tests\test_runtime_dependency_boundary.py -q
```

Expected: FAIL，持久化 Repository/Service 尚不存在。

- [ ] **Step 3: 实现 Repository**

核心方法：

```python
async def get_by_client_request(
    self,
    conversation_id: UUID,
    user_id: UUID,
    client_request_id: UUID,
) -> PersistedRun | None:
    raise NotImplementedError

async def create_running(
    self,
    *,
    conversation: Conversation,
    client_request_id: UUID,
    question: str,
    top_k: int,
    history_limit: int,
) -> PendingRun:
    raise NotImplementedError

async def complete(
    self,
    pending: PendingRun,
    execution: RAGExecution,
    latency_ms: int,
) -> CompletedRun:
    raise NotImplementedError

async def fail(
    self,
    pending: PendingRun,
    *,
    run_status: Literal["failed", "cancelled"],
    error_code: str,
    public_message: str,
    latency_ms: int,
) -> None:
    raise NotImplementedError
```

Repository 返回 dataclass 快照，离开 Session 后不暴露 ORM。历史查询只选 completed user/assistant，按倒序 limit 8 后恢复正序；当前 question 在读取历史之后才 add，因此不会重复。

- [ ] **Step 4: 实现 ChatPersistenceService**

```python
class ChatPersistenceService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        executor: RAGExecutor,
        clock: Callable[[], datetime],
    ) -> None:
        raise NotImplementedError

    async def chat(
        self,
        *,
        principal: AuthenticatedPrincipal,
        conversation_id: UUID,
        client_request_id: UUID,
        question: str,
        top_k: int | None,
    ) -> PersistedChatResult:
        raise NotImplementedError
```

标题仍为“新对话”时，tx1 用规范化问题前 40 个字符生成标题并更新 `updated_at`。不调用 LLM 生成标题。任何异常映射只保存稳定 code 和安全 message；异常链仅供受控日志记录类型，不能进入响应或数据库 error 字段。

- [ ] **Step 5: 实现幂等重放**

- tx1 首先查询 `(conversation_id, user_id, client_request_id)`。
- completed：从 answer message metadata 与 rag reference snapshots 重建并返回同一 IDs 和公开响应。
- running：返回 409，不进入 executor。
- failed/cancelled：返回已保存的稳定错误 code/status，不进入 executor。
- 并发首次 INSERT 由数据库 UNIQUE 裁决；捕获唯一约束后 rollback 当前短事务，再开新 Session 走同一重放查询。

- [ ] **Step 6: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\rag tests\integration\test_rag_persistence.py tests\test_runtime_dependency_boundary.py -q
git add app/modules/rag app/modules/conversations/repository.py tests/unit/rag tests/integration/test_rag_persistence.py tests/test_runtime_dependency_boundary.py
git commit -m "feat: persist rag runs with short transactions"
```

Expected: PASS；事件序列证明外部调用两侧没有活动 Session。

## Task 8: 发布认证的 `/api/v1/chat` 并冻结旧接口退役门

**Files:**

- Create: `app/modules/rag/schemas.py`
- Create: `app/modules/rag/router.py`
- Modify: `app/api/chat.py`
- Modify: `app/api/knowledge.py`
- Modify: `app/schemas/knowledge.py`
- Modify: `app/main.py`
- Create: `tests/api/test_chat_v1.py`
- Modify: `tests/test_chat_api.py`
- Modify: `tests/test_knowledge_api.py`

- [ ] **Step 1: 写失败 API 测试**

覆盖：

- 缺 Session 为 401；缺 CSRF/Origin 为 403；非法 UUID/空问题/top_k 越界为 v1 422 包络。
- 正常请求返回新 IDs 和完整 M3 响应字段，HTTP response model 不暴露内部 database chunk UUID。
- cross-owner 会话为 404；archived 为 409。
- executor 的 Embedding、LLM、database、timeout、rate-limit 错误映射为稳定 code/status，marker 不进入响应。
- 同一 `client_request_id` completed 重放响应完全一致且 executor 只调用一次。
- `/api/chat` 仍接受旧 `{question,history,top_k}`，响应不增加 M4 IDs，错误仍是 `detail`。
- development 的旧 endpoint 返回 `Deprecation: true` 和 `Link: </api/v1/chat>; rel="successor-version"`；staging/production 返回 410；README 明确 M6 Vue 切换时删除。
- `/api/knowledge/extract` 的 `save` 缺省改为 false；route 要求 admin principal 和 CSRF。`save=false` 保持抽取预览，`save=true` 在调用 LLM 或打开文件前返回 410。响应不再返回本机绝对 `knowledge_path`，上游异常不回显供应商 message 或 `str(exc)`。
- `/openapi.json` 包含 auth、conversation 和 v1 chat 路径，Session 使用 cookie security scheme；所有 401/403/404/409/422 response 被声明，schema 中不存在 `password_hash` 或 `token_hash`。

- [ ] **Step 2: 确认测试先失败**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_chat_v1.py tests\test_chat_api.py tests\test_knowledge_api.py -q
```

Expected: FAIL，v1 router 不存在；legacy 测试继续 PASS。

- [ ] **Step 3: 实现 v1 schema**

`ChatV1Request` 要求显式 `client_request_id`，不使用服务端自动 UUID 替代客户端幂等键；question 规范化后长度 1 至 8000；top_k 为 1 至 10，缺省使用 settings。

`ChatV1Response` 组合 M4 IDs 与现有 `ChatResponse` 字段；所有 ORM → Pydantic 转换在 service 返回 detached dataclass 后完成。

- [ ] **Step 4: 注册 router 和兼容响应头**

v1 route 注入 principal、CSRF 和 `ChatPersistenceService`，路由不直接创建 Session。legacy route 不注入 auth、不持久化、不接受 conversation_id，也不复用 v1 service，并只在 development 放行。旧知识抽取 route 在 M4 仅保留 preview；任何环境都不能再写 raw JSONL。

- [ ] **Step 5: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\api\test_chat_v1.py tests\test_chat_api.py tests\test_knowledge_api.py -q
git add app/modules/rag/schemas.py app/modules/rag/router.py app/api/chat.py app/api/knowledge.py app/schemas/knowledge.py app/main.py tests/api/test_chat_v1.py tests/test_chat_api.py tests/test_knowledge_api.py
git commit -m "feat: add authenticated persistent chat api"
```

Expected: PASS；v1 与 legacy 契约并存且互不污染。

## Task 9: 完成双用户隔离、引用快照和并发验收

**Files:**

- Create: `tests/integration/test_user_isolation.py`
- Modify: `tests/integration/test_rag_persistence.py`
- Create: `tests/api/test_m4_workflow.py`
- Modify: `tests/test_runtime_dependency_boundary.py`

- [ ] **Step 1: 建立真实 PostgreSQL 双用户 fixture**

fixture 在专用测试库创建 user A/B、各自 Session、Conversation 和知识引用，结束时按外键逆序清理。必须继续调用 `require_test_database_url()`，禁止把主库误当测试库。

- [ ] **Step 2: 编写隔离矩阵**

| 操作 | A 访问 A | A 访问 B | 未登录 |
|---|---:|---:|---:|
| conversation GET/list | 200 | 404/列表无 B | 401 |
| PATCH/DELETE | 200/204 | 404 | 401 |
| messages list | 200 | 404 | 401 |
| v1 chat | 200 | 404 | 401 |
| admin dependency | admin 通过 | user 403 | 401 |

所有失败响应都断言不包含 B 的 UUID、title、正文、email 或 username。

- [ ] **Step 3: 编写端到端 fake-provider 工作流**

使用依赖覆盖的 fake RAGExecutor，不访问外网：登录 → 取 Cookie/CSRF → 创建会话 → chat → 获取消息 → logout → me 401。验证数据库存在一条 user message、一条 assistant message、一条 completed run 和按 rank 排序的引用。

- [ ] **Step 4: 验证快照不可变和并发幂等**

- 完成回答后修改或删除知识 chunk；历史引用 snapshot 内容保持不变，chunk 删除只把 FK 置 NULL。
- 同一用户并发发送相同 client_request_id，两次请求最终只产生一个 run 和一对 messages，executor 最多一次进入有效执行；另一请求返回 completed 重放或 running 409。
- 两个用户并发对各自会话提问，消息、run、reference 不交叉。

- [ ] **Step 5: 执行测试并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\integration\test_user_isolation.py tests\integration\test_rag_persistence.py tests\api\test_m4_workflow.py tests\test_runtime_dependency_boundary.py -q
git add tests/integration/test_user_isolation.py tests/integration/test_rag_persistence.py tests/api/test_m4_workflow.py tests/test_runtime_dependency_boundary.py
git commit -m "test: prove m4 isolation and persistence"
```

Expected: PASS；泄漏计数为 0，重复 run/message 计数为 0。

## Task 10: 完成文档、空库迁移和 M4 全量验收

**Files:**

- Modify: `README.md`
- Create: `docs/baselines/2026-07-31-m4-auth-conversation-persistence.md`
- Verify: `docs/adr/0001-m0-engineering-baseline.md`
- Verify: all production and test files changed in Tasks 1-9

- [ ] **Step 1: 更新运行文档**

README 必须记录：

- `SESSION_SECRET`、`SESSION_TTL_SECONDS`、`ALLOWED_ORIGINS` 的开发/生产差异。
- `scripts.create_user` 的交互式 admin/user 创建命令。
- Cookie 登录、读取 CSRF Cookie、发送 `X-CSRF-Token` 的 curl/PowerShell 示例，但示例不得出现真实 token。
- Conversation 和 v1 chat 请求顺序、client_request_id 重试规则。
- legacy `/api/chat` 只在 M4 保持兼容，并在 M6 Vue 切换提交删除。
- 失败状态查询和 request ID 排查方式。

- [ ] **Step 2: 从空测试库执行完整 migration 回环**

```powershell
$m4ApplicationDatabaseUrl='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag'
$m4TestDatabaseUrl='postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test'
$env:DATABASE_URL=$m4TestDatabaseUrl
docker compose up -d postgres
.\.venv\Scripts\python.exe -c "from tests.integration.database_safety import require_test_database_url; import os; require_test_database_url(os.environ['DATABASE_URL'], '$m4ApplicationDatabaseUrl')"
.\.venv\Scripts\python.exe -m alembic downgrade base
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic check
$env:DATABASE_URL=$m4ApplicationDatabaseUrl
$env:TEST_DATABASE_URL=$m4TestDatabaseUrl
```

Expected: 空测试库升级到 0004；无 metadata drift。

- [ ] **Step 3: 运行分层和全量测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\unit\auth tests\unit\conversations tests\unit\rag -q
.\.venv\Scripts\python.exe -m pytest tests\api -q
.\.venv\Scripts\python.exe -m pytest tests\integration -q -rs
.\.venv\Scripts\python.exe -m pytest -q -rs
```

Expected: 0 failed、0 unexpected skipped；只允许已经记录的 Starlette/httpx TestClient 弃用 warning。新增 warning 必须先解决或写入验收报告，不能静默忽略。

- [ ] **Step 4: 执行静态边界和敏感信息扫描**

```powershell
rg -n "faiss|id_map\.json|kb_chunks\.jsonl" app
rg -n "token_hash|password_hash" app\modules app\core
rg -n "allow_origins=.*\*|SESSION_SECRET=.*(changeme|secret)" app .env.example
git diff --check
git status --short
```

Expected:

- 第一个命令无在线依赖匹配。
- hash 只出现在模型、安全和仓储逻辑，不进入公开 schema/日志。
- 无通配 CORS 或硬编码生产 secret。
- `git diff --check` 通过；`tmp/` 仍未跟踪且未暂存。

- [ ] **Step 5: 运行 Compose smoke 并停止容器**

```powershell
docker compose up -d --build
docker compose ps
curl.exe -fsS http://127.0.0.1:8000/health/live
curl.exe -fsS http://127.0.0.1:8000/health/ready
docker compose down
docker compose ps --all
```

Expected: live/ready 均为 200；最终所有项目容器为 Exited 或列表为空。

- [ ] **Step 6: 写 M4 验收报告并提交**

报告记录 commit、迁移 head、测试数量/耗时、双用户隔离矩阵、幂等结果、失败状态、快照验证、Cookie/CSRF 属性、容器停止状态和已知限制。不得记录完整聊天正文、Cookie 或凭据。

```powershell
git add README.md docs/baselines/2026-07-31-m4-auth-conversation-persistence.md
git commit -m "docs: record m4 acceptance"
git status --short --branch
```

Expected: 分支只保留用户已有的 `tmp/` 未跟踪项，无 M4 未提交文件。

---

## 稳定错误码矩阵

| 场景 | HTTP | code | 持久化 run 状态 |
|---|---:|---|---|
| 无/无效/过期 Session | 401 | `AUTH_SESSION_INVALID` | 不创建 |
| 登录凭据错误 | 401 | `AUTH_INVALID_CREDENTIALS` | 不创建 |
| 角色不足 | 403 | `AUTH_FORBIDDEN` | 不创建 |
| CSRF/来源失败 | 403 | `CSRF_VALIDATION_FAILED` | 不创建 |
| 会话不存在或跨用户 | 404 | `CONVERSATION_NOT_FOUND` | 不创建 |
| 会话已归档 | 409 | `CONVERSATION_ARCHIVED` | 不创建 |
| 幂等请求仍运行 | 409 | `RAG_REQUEST_IN_PROGRESS` | 保持 running |
| 请求输入无效 | 422 | `REQUEST_VALIDATION_FAILED` | 不创建 |
| Embedding 不可用 | 502 | `EMBEDDING_UNAVAILABLE` | failed |
| LLM 鉴权/连接/API 错误 | 502 | `LLM_UNAVAILABLE` | failed |
| LLM 限流 | 429 | `LLM_RATE_LIMITED` | failed |
| 外部请求超时 | 504 | `RAG_UPSTREAM_TIMEOUT` | failed |
| 客户端取消 | 请求终止 | `RAG_CANCELLED` | cancelled |
| PostgreSQL 暂不可用 | 503 | `DATABASE_UNAVAILABLE` | 能收口时 failed，否则明确记录未确认 |
| 未处理异常 | 500 | `INTERNAL_ERROR` | failed |

## 事务与状态不变量

```text
HTTP auth + CSRF
        |
        v
短 Session/事务 1
  - owner 条件读取 active conversation
  - 读取最近 8 条 completed 历史
  - user message = completed
  - assistant message = pending
  - rag_run = running
  - commit + close
        |
        v
无业务 Session 存活
  - planner
  - Embedding
  - pgvector 自有短 Session
  - LLM
        |
        v
短 Session/事务 2
  - assistant pending -> completed/failed
  - rag_run running -> completed/failed/cancelled
  - references + snapshot
  - commit + close
```

- 每个 accepted v1 chat 最多一条 user message、一条 assistant message和一条 rag_run。
- run completed 必须有 completed answer message；run running 的 answer message 必须 pending。
- failed/cancelled 不写 rag_references；completed reference rank 从 1 连续递增。
- 用户问题一经成功写入即为 completed；外部失败只把助手占位消息设为 failed，保留用户原问题用于解释和人工重试。
- service 不在 Session 外访问未加载 ORM 属性；跨事务只传 UUID、字符串和冻结 dataclass。
- 不在全局 `_rag_pipeline` 中缓存 principal、conversation、message、rag_run 或 AsyncSession。

## 最终验收清单

- [ ] `main@bcf8e9d` 合并基线的 417 项测试无回归。
- [ ] 空测试库可从 base 升到 `0004_create_identity_conversation_rag_tables`，downgrade/upgrade 回环通过。
- [ ] 数据库从未保存 raw Session token，API 从未返回 password/session hash。
- [ ] 生产 Cookie、CSRF、Origin/Referer 和 CORS 符合 ADR-0001。
- [ ] admin/user dependency、禁用用户和 Session 撤销有测试。
- [ ] 两个用户的会话、消息、run、reference 泄漏为 0。
- [ ] `/api/v1/chat` 等待外部服务时无业务 Session/transaction 存活。
- [ ] completed/failed/cancelled 三类 RAG 终态均可从数据库解释。
- [ ] 重复 client_request_id 不重复调用外部服务或新增消息/run。
- [ ] 知识修改或删除不改变历史引用 snapshot。
- [ ] legacy `/api/chat` 在 M4 契约不变并明确标记退役，在线仍只有 pgvector 检索。
- [ ] staging/production 不暴露匿名 legacy chat；旧知识抽取仅管理员可预览，不能再追加 JSONL 或泄露本机路径。
- [ ] 全量测试 0 failed、0 unexpected skipped，Compose smoke 通过后所有容器停止。
- [ ] `tmp/` 及其他用户未跟踪内容未修改、未暂存。

## 计划自检

- 架构覆盖：users、sessions、conversations、messages、rag_runs、rag_references、auth、role、CSRF、owner isolation、两短事务、失败状态和引用快照均有实现任务与验收测试。
- 依赖一致：沿用 M3 的 AsyncSession factory、pgvector Knowledge Search 和 `RAGPipeline`，不新建第二套检索或数据库层。
- 兼容一致：v1 使用新错误与持久化契约；legacy 请求/响应/错误保持 M3 行为。
- 类型一致：数据库内部 chunk UUID 与公开 legacy chunk id 分离；ORM 不作为 API response。
- 安全一致：M0 ADR 的 pwdlib、Cookie、token hash、TTL、CSRF 和 CORS 决策均未漂移。
- 范围一致：没有 Vue、ingestion、Redis、JWT、HNSW、多 Worker 或生产压测任务。
- 可执行性：每个任务包含失败测试、最小实现、精确命令、预期结果和独立提交点。
