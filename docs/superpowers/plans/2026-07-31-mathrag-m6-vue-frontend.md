# MathRAG M6 Vue 3 前端代码级开发计划

> **执行要求：** 实施本计划时，使用 `superpowers:subagent-driven-development`（当前会话）或 `superpowers:executing-plans`（独立会话），逐任务执行测试、实现、复核和提交。

**目标：** 用 Vue 3 + TypeScript 替换现有静态聊天页，交付登录、持久化对话、知识条目、文档和摄取任务管理界面，并由 FastAPI 同源托管生产构建产物。

**架构：** 前端采用 Vite、Vue Router、Composition API 和按功能拆分的服务端状态；API 类型由 FastAPI OpenAPI 生成，所有请求统一经过同源客户端。生产镜像通过 Node 构建阶段生成 `frontend/dist`，Python 运行阶段只携带静态产物。M6 继续使用现有单进程 `BackgroundTasks` 摄取模型；Redis、Celery、多 worker、OCR 和生产限流属于 M7。

**技术栈：** Vue 3.5、TypeScript 5.9、Vite 8、Vue Router 5、KaTeX、Lucide Vue、Vitest、Vue Test Utils、Testing Library、MSW、Playwright、FastAPI、pytest、Docker Compose。

---

## 1. 基线与固定决策

### 1.1 M6 起点

- 分支：`codex/m6-vue-frontend`
- 基线：合并 M5 后的 `main@fce45f6`
- M5 合并前完整 runtime：`572 passed, 1 warning`
- 合并后 `main` 完整 runtime：`572 passed, 1 warning`
- 完整 runtime 命令沿用仓库边界，排除依赖可选 FAISS 的评测测试：

```powershell
docker compose run --rm app pytest -q -rs --ignore=tests/evaluation --ignore=tests/test_retrieval_baseline.py
```

### 1.2 产品与协议决策

1. 路由固定为 `/login`、`/chat`、`/conversations`、`/conversations/:id`、`/knowledge`、`/knowledge/new`、`/knowledge/:id`、`/documents`、`/jobs`。
2. `/knowledge`、`/documents`、`/jobs` 仅管理员可访问；普通用户可登录、聊天和管理自己的对话。
3. 不引入 Pinia。当前用户与当前对话使用应用级 composable，页面查询状态归属 feature composable，局部交互状态留在组件内。
4. 不在浏览器存储 session token、CSRF token、对话或任务真相。session 使用 HttpOnly cookie；CSRF 从 `mathrag_csrf` 或 `__Host-mathrag_csrf` cookie 读取。
5. 功能组件不得直接调用 `fetch`。所有请求通过统一 API 客户端，使用相对 URL、`credentials: 'same-origin'`、`X-Request-ID`、结构化错误和 unsafe method CSRF 头。
6. 数学内容先以纯文本进入 DOM，再由本地 KaTeX 渲染；禁止用 `v-html` 注入模型响应。
7. `/jobs` 需要服务端任务列表。M6 增加管理员只读契约 `GET /api/v1/ingestion-jobs`，不使用 `localStorage` 拼装伪列表。
8. 当前回答显示响应引用；恢复历史消息时显示 `model_metadata.response` 中已有的步骤、知识使用、关联问题和 agentic 计划。M6 不新增 RAG run 读取接口，历史引用缺失时省略引用区。
9. 旧 `/api/chat` 在 Vue 切换完成后删除；`/api/v1/rag/answer` 是唯一聊天写入入口。
10. SPA fallback 只处理前端路由。`/api`、`/health`、`/docs`、`/redoc`、`/openapi.json` 保留后端语义，未知 API 必须返回 JSON 404。

### 1.3 布局与响应式约束

- 桌面端：216px 固定导航栏，主区域使用紧凑工具栏；移动端使用顶栏和抽屉导航。
- 管理页面优先表格和无外框分区，不使用卡片套卡片。
- 无渐变，圆角不超过 8px；图标按钮使用 Lucide、固定尺寸并提供 tooltip。
- 字号不随 viewport 宽度缩放；动态内容不得造成工具栏跳动。
- 验收视口：`1440x900`、`1024x768`、`390x844`。

### 1.4 目标目录

```text
frontend/
  index.html
  package.json
  package-lock.json
  tsconfig*.json
  vite.config.ts
  playwright.config.ts
  openapi.json
  src/
    app/
    api/
    assets/
    components/
    composables/
    features/{auth,chat,conversations,knowledge,documents,jobs}/
    router/
    styles/
    test/
  tests/e2e/
scripts/export_openapi.py
app/core/frontend.py
tests/api/test_ingestion_jobs.py
tests/test_frontend_spa.py
tests/test_app_lifespan.py
```

---

## 2. 代码任务

## Task 1：建立 Vue、TypeScript 与测试骨架

**文件：**

- 新建：`frontend/package.json`、`frontend/package-lock.json`、`frontend/index.html`
- 新建：`frontend/tsconfig.json`、`frontend/tsconfig.app.json`、`frontend/tsconfig.node.json`
- 新建：`frontend/vite.config.ts`
- 新建：`frontend/src/main.ts`、`frontend/src/App.vue`、`frontend/src/vite-env.d.ts`
- 新建：`frontend/src/styles/tokens.css`、`frontend/src/styles/base.css`
- 新建：`frontend/src/test/setup.ts`、`frontend/src/app/App.spec.ts`、`frontend/src/styles/tokens.spec.ts`
- 修改：`.gitignore`、`.dockerignore`

### Step 1：先写失败测试

验证根应用提供 router view；样式测试验证核心颜色/间距 token、最大 8px 圆角且不存在 gradient。

```powershell
Set-Location frontend
npm.cmd test -- --run src/app/App.spec.ts src/styles/tokens.spec.ts
```

预期：FAIL，项目和入口尚不存在。

### Step 2：声明固定依赖并生成 lockfile

`package.json` 固定以下版本：

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vue-tsc -b && vite build",
    "typecheck": "vue-tsc -b",
    "lint": "eslint . --max-warnings=0",
    "format:check": "prettier --check .",
    "test": "vitest",
    "e2e": "playwright test",
    "api:generate": "openapi-typescript openapi.json -o src/api/schema.d.ts",
    "api:check": "npm run api:generate && git diff --exit-code -- openapi.json src/api/schema.d.ts"
  },
  "dependencies": {
    "katex": "0.18.1",
    "lucide-vue-next": "1.0.0",
    "vue": "3.5.40",
    "vue-router": "5.2.0"
  },
  "devDependencies": {
    "@eslint/js": "10.0.1",
    "@playwright/test": "1.62.1",
    "@testing-library/vue": "8.1.0",
    "@vitejs/plugin-vue": "6.0.8",
    "@vue/test-utils": "2.4.11",
    "eslint": "10.8.0",
    "jsdom": "30.0.1",
    "msw": "2.15.0",
    "openapi-typescript": "7.13.0",
    "prettier": "3.9.6",
    "typescript": "5.9.3",
    "typescript-eslint": "8.65.0",
    "vite": "8.2.0",
    "vitest": "4.1.10",
    "vue-tsc": "3.3.9"
  }
}
```

运行 `npm.cmd install` 生成 lockfile，不手写 `package-lock.json`。

### Step 3：实现最小入口

- `App.vue` 只提供应用挂载点和 `<RouterView />`，不承担 feature 状态。
- `tokens.css` 定义中性色、蓝色操作色、绿色成功、琥珀警告、红色错误、边框和间距。
- `base.css` 设置字体栈、`box-sizing`、focus-visible、表单和滚动基础样式。
- Vite dev proxy 只代理 `/api`、`/health`、`/openapi.json` 到 `127.0.0.1:8000`。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run
npm.cmd run typecheck
npm.cmd run lint
npm.cmd run format:check
git add frontend .gitignore .dockerignore
git commit -m "build: scaffold vue frontend"
```

---

## Task 2：生成 OpenAPI 类型并建立统一 API 客户端

**文件：**

- 新建：`scripts/export_openapi.py`
- 新建：`frontend/openapi.json`、`frontend/src/api/schema.d.ts`
- 新建：`frontend/src/api/client.ts`、`frontend/src/api/errors.ts`、`frontend/src/api/csrf.ts`
- 新建：`frontend/src/api/client.spec.ts`
- 修改：`frontend/package.json`

### Step 1：写失败的客户端契约测试

使用 MSW 验证相对 URL、same-origin credentials、每次请求的 `X-Request-ID`、unsafe method CSRF、login 例外、204、AbortError，以及统一错误 envelope：

```ts
await expect(apiRequest('/api/v1/example')).rejects.toMatchObject({
  code: 'EXAMPLE_ERROR',
  requestId: 'request-123',
})
```

### Step 2：实现确定性契约导出

`scripts/export_openapi.py` 导入应用工厂，对 schema key 稳定排序，以 UTF-8、LF、末尾换行写入 `frontend/openapi.json`；导出过程不得连接数据库。

```powershell
python scripts/export_openapi.py
Set-Location frontend
npm.cmd run api:generate
```

### Step 3：实现客户端

```ts
export interface ApiRequestOptions<TBody = unknown> {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'
  body?: TBody | FormData
  signal?: AbortSignal
  requestId?: string
}

export function apiRequest<TResponse, TBody = unknown>(
  path: `/api/${string}`,
  options?: ApiRequestOptions<TBody>,
): Promise<TResponse>
```

JSON、FormData、204、结构化错误和取消均在此层处理；写请求不自动重试，业务 feature 决定幂等策略。CSRF 同时识别开发与生产 cookie 名，不手工设置 Origin。

### Step 4：增加漂移门禁并提交

```powershell
npm.cmd test -- --run src/api/client.spec.ts
npm.cmd run api:check
npm.cmd run typecheck
git add scripts/export_openapi.py frontend/openapi.json frontend/src/api frontend/package.json frontend/package-lock.json
git commit -m "feat: add generated api client contract"
```

---

## Task 3：补齐摄取任务列表 API

**文件：**

- 修改：`app/schemas/ingestion.py`
- 修改：`app/repositories/ingestion_job_repository.py`
- 修改：`app/services/ingestion_service.py`
- 修改：`app/api/v1/ingestion.py`
- 修改：`tests/api/test_ingestion_jobs.py`
- 重新生成：`frontend/openapi.json`、`frontend/src/api/schema.d.ts`

### Step 1：写失败的 API 测试

覆盖管理员分页读取、普通用户 403、`status`/`job_type`/`document_id` 筛选、`created_at DESC, id DESC` 稳定排序、空页元数据、非法分页与枚举 422。

```json
{ "items": [], "total": 0, "offset": 0, "limit": 25 }
```

### Step 2：实现 repository、service、router

- schema 新增 `IngestionJobPage`。
- repository 新增 `list_jobs(...) -> tuple[list[IngestionJob], int]`，count 和 list 复用筛选条件。
- service 负责分页边界和 schema 转换。
- collection 路径新增 `GET /api/v1/ingestion-jobs`，复用管理员依赖且不影响 `/{job_id}`。

### Step 3：验证、生成契约并提交

```powershell
docker compose run --rm app pytest -q tests/api/test_ingestion_jobs.py
python scripts/export_openapi.py
Set-Location frontend
npm.cmd run api:generate
npm.cmd run api:check
Set-Location ..
git add app tests/api/test_ingestion_jobs.py frontend/openapi.json frontend/src/api/schema.d.ts
git commit -m "feat: expose ingestion job listing"
```

---

## Task 4：实现认证状态、登录页与路由守卫

**文件：**

- 新建：`frontend/src/router/index.ts`、`frontend/src/router/meta.d.ts`
- 新建：`frontend/src/features/auth/api.ts`、`frontend/src/features/auth/useAuth.ts`
- 新建：`frontend/src/features/auth/LoginPage.vue`
- 新建：`frontend/src/features/auth/auth.spec.ts`、`frontend/src/router/router.spec.ts`
- 修改：`frontend/src/main.ts`

### Step 1：写失败测试

覆盖单次 `/auth/me` bootstrap、并发消费者共享 Promise、登录后重新读取用户、401 清空状态、logout、管理员守卫和安全 `next`。`next` 只允许站内绝对路径，拒绝协议、双斜线和反斜线绕过；bootstrap 期间不闪现保护页面。

### Step 2：实现认证状态

```ts
type AuthState =
  | { status: 'unknown'; user: null }
  | { status: 'anonymous'; user: null }
  | { status: 'authenticated'; user: AuthUser }
```

暴露 `bootstrap()`、`login()`、`logout()` 和 readonly state。登录成功后请求 `/auth/me`，不信任表单构造身份；不写 local/session storage。

### Step 3：实现路由和登录页

- meta 使用 `requiresAuth`、`requiresAdmin`。
- `/` 按认证态跳 `/chat` 或 `/login`。
- 管理员路由拒绝普通用户并返回 `/chat`。
- 登录页包含邮箱、密码、提交中、字段错误、服务端错误和禁用态。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/features/auth/auth.spec.ts src/router/router.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src
git commit -m "feat: add frontend authentication flow"
```

---

## Task 5：实现响应式工作台壳层与通用控件

**文件：**

- 新建：`frontend/src/app/AppShell.vue`、`AppNavigation.vue`、`AppHeader.vue`
- 新建：`frontend/src/components/IconButton.vue`、`InlineAlert.vue`、`LoadingState.vue`
- 新建：`frontend/src/components/EmptyState.vue`、`PaginationControls.vue`、`ConfirmDialog.vue`
- 新建：`frontend/src/app/AppShell.spec.ts`
- 修改：`frontend/src/App.vue`、`frontend/src/styles/base.css`

### Step 1：写失败测试

验证桌面导航、移动菜单、当前路由、管理员菜单可见性、Escape 关闭抽屉/dialog、focus 回到触发按钮，以及图标按钮有 accessible name 和 tooltip。

### Step 2：实现壳层

- 导航项：聊天、对话；管理员额外显示知识、文档、任务。
- header 显示页面标题、用户菜单和退出命令。
- `IconButton` 固定 `36x36`，动态文案不改变布局。
- `ConfirmDialog` 提供 focus trap；危险操作明确显示对象名。
- loading/empty/error 是页面状态，不套嵌装饰卡片。

### Step 3：验证并提交

```powershell
npm.cmd test -- --run src/app/AppShell.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/app frontend/src/components frontend/src/App.vue frontend/src/styles
git commit -m "feat: add responsive application shell"
```

---

## Task 6：安全渲染数学内容、回答详情与引用

**文件：**

- 新建：`frontend/src/components/MathContent.vue`、`MathContent.spec.ts`
- 新建：`frontend/src/features/chat/AnswerView.vue`、`ReferenceList.vue`
- 新建：`frontend/src/features/chat/ReasoningSteps.vue`、`RelatedQuestions.vue`
- 新建：`frontend/src/features/chat/AnswerView.spec.ts`
- 修改：`frontend/src/main.ts`

### Step 1：写失败测试

覆盖 inline/display math、转义美元、错误 LaTeX、`<script>`、`<img onerror>`、超长公式和空文本。断言 HTML 不被解释、KaTeX 错误回落原始文本、长公式仅在自身容器滚动。

```ts
expect(wrapper.find('script').exists()).toBe(false)
expect(wrapper.text()).toContain('<script>alert(1)</script>')
```

### Step 2：实现纯文本优先渲染

- 先设置 `textContent`，再调用本地 KaTeX auto-render；禁止 `v-html`。
- delimiters 固定为 `$$...$$`、`\[...\]`、`\(...\)`、`$...$`，单美元最后匹配。
- 使用 `throwOnError: false`、`strict: 'warn'`，不允许远程资源命令。
- 从 `main.ts` 导入本地 `katex.min.css`，不使用 CDN。

### Step 3：实现回答详情

- 主答案始终可见；引用展示序号、来源、片段和相关度。
- 推理步骤和 agentic 计划可折叠，详细内容默认收起。
- `used_knowledge`、`related_questions` 独立展示；相关问题只发出填充事件。
- 历史消息没有引用时不显示空引用标题。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/components/MathContent.spec.ts src/features/chat/AnswerView.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/components frontend/src/features/chat frontend/src/main.ts
git commit -m "feat: render math answers safely"
```

---

## Task 7：实现对话列表、历史恢复、重命名与归档

**文件：**

- 新建：`frontend/src/features/conversations/api.ts`、`types.ts`、`useConversations.ts`
- 新建：`frontend/src/features/conversations/ConversationListPage.vue`
- 新建：`frontend/src/features/conversations/ConversationHistoryPage.vue`、`ConversationRow.vue`
- 新建：`frontend/src/features/conversations/conversations.spec.ts`
- 修改：`frontend/src/router/index.ts`

### Step 1：写失败测试

覆盖初始加载、分页、刷新、空态、错误重试、URL 参数同步、重命名、归档确认、403/404/409，以及历史消息按服务端顺序并从 `model_metadata.response` 恢复回答详情。

### Step 2：实现查询状态

```ts
type QueryState<T> =
  | { status: 'idle'; data: null }
  | { status: 'loading'; data: T | null }
  | { status: 'success'; data: T }
  | { status: 'error'; data: T | null; error: ApiError }
```

新查询用序号或 AbortController 防止旧响应覆盖新筛选；服务端是唯一真相，mutation 成功后更新缓存或重新拉取。

### Step 3：实现页面

- `/conversations`：后端支持的筛选、分页、最后活动时间、重命名和归档。
- `/conversations/:id`：只读历史、继续对话、刷新和错误态。
- 日期使用 `Intl.DateTimeFormat`，完整 ISO 时间放在可访问 title。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/features/conversations/conversations.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/features/conversations frontend/src/router
git commit -m "feat: add conversation management views"
```

---

## Task 8：实现持久化聊天、取消与幂等重试

**文件：**

- 新建：`frontend/src/features/chat/api.ts`、`types.ts`、`useChat.ts`
- 新建：`frontend/src/features/chat/ChatPage.vue`、`ChatComposer.vue`、`MessageList.vue`
- 新建：`frontend/src/features/chat/chat.spec.ts`
- 修改：`frontend/src/features/conversations/ConversationHistoryPage.vue`
- 修改：`frontend/src/router/index.ts`

### Step 1：写失败的状态机测试

状态固定为 `idle | submitting | success | error | cancelled`，覆盖：

- `/chat` 首次发送先创建 conversation，再调用 `/api/v1/rag/answer`；
- `/conversations/:id` 继续聊天复用 conversation id；
- submitting 禁止重复提交，但允许取消；
- AbortController 取消后进入 cancelled，不显示一般网络错误；
- transport error 或 `RAG_REQUEST_IN_PROGRESS` 重试复用原 `client_request_id`；
- `RAG_CANCELLED` 后再次发送生成新 id；
- 相同 id 的成功响应不重复插入消息；
- top-k 控件限制最小、最大和默认值；
- 相关问题点击只填输入框，不自动发送。

### Step 2：实现请求生命周期

```ts
interface PendingTurn {
  conversationId: string
  clientRequestId: string
  question: string
  topK: number
  controller: AbortController
}
```

- id 使用 `crypto.randomUUID()`。
- 创建对话后立即 replace URL 为 `/conversations/:id`，避免刷新丢上下文。
- 只有可安全重试失败保留 PendingTurn；校验错误要求用户修改输入。
- 卸载时 abort，并用 request identity 阻止迟到响应写入。
- 成功后以服务端消息和 metadata 为准，不构造假 id/时间戳。

### Step 3：实现聊天工作区

- 用户消息和回答有清晰角色区分，composer 包含多行输入、top-k stepper、发送/停止图标按钮。
- 输入不为空且非 submitting 才能发送；IME composition 中不得误提交。
- 错误显示结构化 message 和 request id，不显示技术堆栈。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/features/chat/chat.spec.ts src/features/conversations/conversations.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/features/chat frontend/src/features/conversations frontend/src/router
git commit -m "feat: add persistent rag chat workflow"
```

---

## Task 9：实现知识条目管理与 revision 冲突处理

**文件：**

- 新建：`frontend/src/features/knowledge/api.ts`、`types.ts`、`useKnowledge.ts`
- 新建：`frontend/src/features/knowledge/KnowledgeListPage.vue`
- 新建：`frontend/src/features/knowledge/KnowledgeEditorPage.vue`
- 新建：`frontend/src/features/knowledge/KeywordInput.vue`、`StepEditor.vue`
- 新建：`frontend/src/features/knowledge/knowledge.spec.ts`
- 修改：`frontend/src/router/index.ts`

### Step 1：写失败的 CRUD 测试

覆盖列表筛选/分页、创建、读取、编辑、归档/删除（按真实契约）、关键字去空去重、步骤增删排序、字段级 422、权限错误和 revision 冲突。

冲突时必须保留本地草稿，显示服务器已更新，并提供“重新载入服务器版本”和“保留草稿后重新应用”两个动作；不得静默覆盖。

### Step 2：实现表单映射

- 使用后端 schema 的真实字段，不发明同义字段。
- keywords 用 token 输入，提交为 API 数组。
- steps 每行有顺序、内容、删除；排序后统一重算 position。
- 数学题干和解答预览复用 `MathContent`。
- mutation 成功后用服务端实体替换表单基线并更新 revision。

### Step 3：实现管理页面

- `/knowledge`：搜索、类型/状态筛选、分页、编辑和归档确认。
- `/knowledge/new`：创建表单。
- `/knowledge/:id`：详情、脏表单离开确认、保存状态和冲突区。
- 页面使用表格和分栏编辑器，不把字段逐个包装成卡片。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/features/knowledge/knowledge.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/features/knowledge frontend/src/router
git commit -m "feat: add knowledge management interface"
```

---

## Task 10：实现文档上传与摄取任务监控

**文件：**

- 新建：`frontend/src/features/documents/api.ts`、`DocumentsPage.vue`、`DocumentUpload.vue`
- 新建：`frontend/src/features/documents/documents.spec.ts`
- 新建：`frontend/src/features/jobs/api.ts`、`useJobPolling.ts`
- 新建：`frontend/src/features/jobs/JobsPage.vue`、`JobStatusBadge.vue`、`jobs.spec.ts`
- 修改：`frontend/src/router/index.ts`

### Step 1：写失败的上传和轮询测试

覆盖 FormData、上传成功和错误、任务筛选/分页/取消/重试；仅 pending/running 轮询；页面 hidden 或离开路由暂停；恢复可见立即刷新；同一 job 只有一个 timer；终态停止轮询。

### Step 2：实现文档页

- 使用真实 file input；FormData 不手工设置 Content-Type。
- 上传队列逐项显示文件名、大小、提交中、成功和结构化错误。
- 只展示后端可查询的数据范围，不以浏览器缓存伪造文档全集。
- 成功后提供“查看任务”，跳到 `/jobs` 并携带筛选参数。

### Step 3：实现任务页和调度器

```ts
const ACTIVE_JOB_STATUSES = new Set(['pending', 'running'])
const POLL_INTERVAL_MS = 2_000
```

- 列表使用 Task 3 的 collection API。
- 取消/重试按钮仅在后端允许的状态显示，请求期间禁用。
- mutation 后刷新该 job 和当前页。
- 轮询失败保留最后成功数据，不创建额外 timer；提供手动刷新。

### Step 4：验证并提交

```powershell
npm.cmd test -- --run src/features/documents/documents.spec.ts src/features/jobs/jobs.spec.ts
npm.cmd run typecheck
npm.cmd run lint
git add frontend/src/features/documents frontend/src/features/jobs frontend/src/router
git commit -m "feat: add document ingestion management views"
```

---

## Task 11：切换 FastAPI SPA 托管并删除旧聊天入口

**文件：**

- 新建：`app/core/frontend.py`、`tests/test_frontend_spa.py`、`tests/test_app_lifespan.py`
- 修改：`app/core/config.py`、`app/main.py`、`app/schemas/chat.py`
- 修改：`tests/test_runtime_dependency_boundary.py`
- 删除：`app/api/chat.py`
- 删除：`app/frontend/index.html`、`app/frontend/app.js`、`app/frontend/style.css`
- 删除：`tests/test_chat_api.py`

### Step 1：先迁移有效 lifespan 测试

从 `tests/test_chat_api.py` 原样迁移以下测试到 `tests/test_app_lifespan.py`，先运行确认断言未变：

- `test_lifespan_preserves_app_error_while_attempting_both_cleanups`
- `test_lifespan_rebuilds_rag_dependencies_and_preserves_app_error`

### Step 2：写失败的 SPA 测试

先执行前端 build，再覆盖：

- `/`、`/chat`、`/conversations/<uuid>` 返回 Vue index；
- JS/CSS 返回正确 MIME 和缓存头；
- 无扩展前端深路径回落 index；
- `/api/v1/not-found` 和 `POST /api/chat` 返回 JSON 404；
- `/health`、`/docs`、`/redoc`、`/openapi.json` 保持原路由；
- dist 缺失时应用可启动，根路径返回明确 503 JSON。

### Step 3：实现受限 fallback

```python
RESERVED_PREFIXES = ("api/", "health", "docs", "redoc", "openapi.json")
```

只对 `GET`/`HEAD`、非保留前缀、Accept 允许 HTML 的无扩展路径回落 index。缺失 `.js` 等真实静态文件返回 404。`FRONTEND_DIST_DIR` 默认指向 `frontend/dist` 且可在测试覆盖；SPA 在所有 API/health 路由之后挂载到 `/`。

### Step 4：删除旧入口并收紧边界

- 从 `app.main` 移除旧 router，删除旧静态资源和旧 API 测试。
- `app/schemas/chat.py` 只删 legacy request/turn；v1 RAG 仍用的响应、引用、步骤类型保留。
- runtime dependency boundary 改为扫描 Vue 源码/配置，不再引用旧文件。
- 全仓搜索 `/api/chat`、`app.api.chat`、`app/frontend`，除迁移文档外为零。

### Step 5：验证并提交

```powershell
Set-Location frontend
npm.cmd run build
Set-Location ..
docker compose run --rm app pytest -q tests/test_app_lifespan.py tests/test_frontend_spa.py tests/api/test_chat_v1.py tests/test_runtime_dependency_boundary.py
rg -n "/api/chat|app\.api\.chat|app/frontend" app tests frontend
git add app frontend tests
git commit -m "feat: cut over to vue single page application"
```

---

## Task 12：生产镜像、CI 与浏览器端到端测试

**文件：**

- 修改：`Dockerfile`、`docker-compose.yml`、`tests/test_compose_contract.py`
- 新建：`.github/workflows/quality.yml`
- 新建：`frontend/playwright.config.ts`、`frontend/tests/e2e/fixtures.ts`
- 新建：`frontend/tests/e2e/auth-chat.spec.ts`
- 新建：`frontend/tests/e2e/admin-management.spec.ts`
- 新建：`frontend/tests/e2e/responsive.spec.ts`

### Step 1：写失败的容器契约测试

要求 Dockerfile 使用：

```dockerfile
FROM node:24.11.1-bookworm-slim AS frontend-build
FROM python:3.11.9-slim AS runtime
```

并验证 Node 阶段执行 `npm ci`/`npm run build`，runtime 只复制 dist、不携带 `node_modules`，最终仍是单 Uvicorn worker。

### Step 2：实现 multi-stage 构建

- 先复制 manifest/lockfile 缓存依赖，再复制前端源码构建。
- Python 阶段沿用当前 requirements 安装策略。
- 将 `/frontend/dist` 复制到 `/app/frontend/dist`。
- Compose 开发挂载不得覆盖镜像 dist；前端热更新使用本机 Vite proxy。

### Step 3：实现确定性 E2E

Playwright 通过 route/mock 提供符合 OpenAPI 的数据，覆盖：

1. 登录失败/成功、保护路由和退出。
2. 新建对话、回答、取消、幂等重试、数学和引用。
3. 对话重命名、归档和历史恢复。
4. 管理员知识 CRUD 和 revision 冲突。
5. 文档上传、任务状态、取消和重试。
6. 三个视口无页面横向溢出，导航可用，composer 不遮挡消息。

选择器优先 role、label 和稳定 test id，不依赖脆弱 CSS 层级。

### Step 4：增加 CI

`.github/workflows/quality.yml` 分为 frontend（api drift/format/lint/typecheck/unit/build/E2E）、backend（PostgreSQL/pgvector + 完整 runtime）和 container（compose contract + 镜像构建）。

### Step 5：验证并提交

```powershell
docker compose run --rm app pytest -q tests/test_compose_contract.py
Set-Location frontend
npm.cmd run format:check
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test -- --run
npm.cmd run build
npm.cmd run e2e
Set-Location ..
docker build -t mathrag:m6 .
git add Dockerfile docker-compose.yml tests/test_compose_contract.py .github/workflows/quality.yml frontend
git commit -m "build: package and verify vue frontend"
```

---

## Task 13：全量验收、视觉检查与文档收口

**文件：**

- 修改：`README.md`
- 新建：`docs/baselines/m6-frontend-acceptance.md`
- 新建：`docs/baselines/artifacts/m6-desktop.png`
- 新建：`docs/baselines/artifacts/m6-tablet.png`
- 新建：`docs/baselines/artifacts/m6-mobile.png`

### Step 1：执行前端门禁

```powershell
python scripts/export_openapi.py
Set-Location frontend
npm.cmd run api:check
npm.cmd run format:check
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test -- --run
npm.cmd run build
npm.cmd run e2e
Set-Location ..
```

所有命令为 0；不得通过降低 lint/typecheck 或测试范围换取通过。

### Step 2：执行完整 runtime 和 Compose 烟测

```powershell
docker compose run --rm app pytest -q -rs --ignore=tests/evaluation --ignore=tests/test_retrieval_baseline.py
```

构建并启动最终服务，至少烟测 health、auth/CSRF、RAG 新建/继续/取消、知识 CRUD、上传、任务列表/取消/重试、前端嵌套路由刷新和未知 API JSON 404。

### Step 3：执行真实浏览器视觉检查

用 Playwright 在 `1440x900`、`1024x768`、`390x844` 截图，并人工确认：

- 无页面横向溢出、文本截断和控件重叠；
- 移动抽屉、dialog、菜单不越界且 focus 正确恢复；
- 公式不空白、不裁切正文，长公式局部滚动；
- composer、loading、错误和取消状态不造成布局跳动；
- 状态色可区分且页面不呈单一蓝/紫色；
- 管理页密度适合重复操作。

acceptance 文档记录命令、测试数量、镜像 digest、截图、限制和 M7 移交项。

### Step 4：更新 README

写明本机后端 + Vite 开发、OpenAPI 生成/drift、前端质量命令、Compose 生产构建，以及 M6 单 worker/进程内任务限制。

### Step 5：停止容器并检查工作树

```powershell
docker compose down
docker compose ps --all
git diff --check
git status --short
```

`docker compose ps --all` 必须为空；不得使用 `down -v`，数据库卷保留。工作树只允许用户已有且未纳入本阶段的 `tmp/`。

### Step 6：提交验收材料

```powershell
git add README.md docs/baselines frontend
git commit -m "docs: record m6 frontend acceptance"
```

---

## 3. 错误处理矩阵

| 场景 | UI 行为 | 自动重试 | request id |
|---|---|---:|---|
| 401 session 失效 | 清用户态，跳登录并保留安全站内 next | 否 | 可复制 |
| 403 权限不足 | 页内提示；管理路由回 `/chat` | 否 | 显示 |
| 404 资源不存在 | 保留壳层并显示不存在 | 否 | 显示 |
| 409 revision 冲突 | 保留草稿，提供重载/重新应用 | 否 | 显示 |
| `RAG_REQUEST_IN_PROGRESS` | 保留问题和原 client id，提供重试 | 否 | 显示 |
| `RAG_CANCELLED` | 标记取消，再发生成新 client id | 否 | 显示 |
| AbortError | 标记取消，不显示网络故障 | 否 | 不要求 |
| 422 字段错误 | 映射字段，无法映射则页级提示 | 否 | 显示 |
| 5xx/网络错误 | 保留数据和输入，显式重试 | 否 | 有则显示 |
| job polling 失败 | 保留最后数据，下一周期单次重试 | 是 | 有则显示 |

---

## 4. M6 完成定义

- [ ] 9 条路由可直接进入并刷新，权限守卫符合角色。
- [ ] 所有 feature 请求经过统一 API client，没有裸 `fetch`。
- [ ] OpenAPI 与 TS 类型可重复生成，CI 检测漂移。
- [ ] 登录、退出、session 过期和双 CSRF cookie 名通过测试。
- [ ] 新建/继续对话、取消、幂等重试、历史恢复使用持久化 API。
- [ ] KaTeX 本地打包，恶意 HTML 只按文本显示。
- [ ] 知识 CRUD、revision 冲突、上传、任务列表/取消/重试可用。
- [ ] 旧 `app/api/chat.py`、`app/frontend/*` 和 `/api/chat` 已删除。
- [ ] SPA fallback 不吞 API、健康检查、OpenAPI 和静态文件 404。
- [ ] Node build 与 Python runtime 分离，最终镜像无 `node_modules`。
- [ ] frontend format、lint、typecheck、unit、build、E2E 全通过。
- [ ] 完整 backend runtime 通过并记录在 acceptance 文档。
- [ ] 三个目标视口完成截图和人工布局检查。
- [ ] 所有 Compose 容器停止，数据库卷保留。

## 5. 明确不在 M6 实现

- Redis/Celery 或其他持久化任务队列。
- Uvicorn/Gunicorn 多 worker 摄取协调。
- OCR、复杂 PDF 版面理解、文件病毒扫描。
- 全局生产限流、反向代理和公网 TLS。
- RAG run 历史读取 API 与旧消息引用补录。
- Pinia、SSR、微前端、WebSocket、流式 token。

以上进入 M7 并发、可靠性和生产加固，不在 M6 以半成品形式扩张范围。

## 6. 实施顺序与检查点

任务按 1 至 13 顺序执行。Task 1-2 建立类型和请求边界；Task 3 补齐 jobs 契约；Task 4-10 逐 feature 交付；Task 11 才删除旧入口；Task 12-13 完成生产构建和验收。

每个任务结束必须满足：目标测试通过、相关回归通过、`git diff --check` 通过、代码复核无阻断问题，并形成一个语义清晰的提交。Task 11 和 Task 13 是强制全量回归检查点；失败时必须先定位根因，不得将测试标记为 skip 后继续。
