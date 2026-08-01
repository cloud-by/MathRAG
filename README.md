# MathRAG

MathRAG 是一个面向数学问答场景的 RAG 原型系统，基于 **FastAPI + PostgreSQL/pgvector + OpenAI-Compatible Embedding API + DeepSeek/OpenAI-Compatible Chat API** 构建。

系统会先从本地结构化数学知识库中检索相关知识，再让大模型生成结构化回答，并通过浏览器前端展示答案、步骤、参考知识、检索规划和可继续追问的问题。

---

## 功能概览

- **认证与角色**：服务端 Session Cookie、`student`/`teacher`/`admin` 角色、CSRF 与显式 CORS 来源校验。
- **持久化数学 RAG 问答**：`/api/v1/chat` 保存用户问题、回答、运行状态和引用快照，并支持幂等重试。
- **用户会话隔离**：Conversation、Message 和 RAG Run 查询始终按当前用户过滤。
- **Agentic 检索规划**：先由 LLM 将用户问题改写为 1~4 条检索子问题，再合并检索结果。
- **pgvector 向量检索**：PostgreSQL 是在线知识的唯一事实数据源，按状态、可见性和模型精确过滤。
- **结构化知识库**：JSONL 种子知识经过可重复导入与 reindex 进入 PostgreSQL/pgvector。
- **知识管理**：管理员可通过带 revision 的 CRUD API 创建、更新和归档知识，非管理员只读取 public+ready 数据。
- **统一摄取**：管理员上传 PDF 或通过网页/PDF CLI 创建可轮询、取消和重试的导入任务；新数据不再追加 JSONL。
- **公式渲染**：前端通过 KaTeX 渲染 `\(...\)` 与 `\[...\]` 公式。
- **LLM JSON 修复**：对模型输出中常见 LaTeX 反斜杠转义问题做容错修复。
- **Docker 部署**：提供 `Dockerfile` 与 `docker-compose.yml`。
- **测试覆盖**：提供 API、RAG、导入器相关 pytest 用例。

---

## 项目结构

```text
MathRAG/
├── app/
│   ├── api/                  # FastAPI 路由注册
│   ├── core/                 # 配置、日志与 SPA 静态托管
│   ├── schemas/              # Pydantic 请求/响应模型
│   ├── modules/auth/         # 服务端 Session、CSRF 与角色依赖
│   ├── modules/conversations/# 用户隔离的会话与消息
│   ├── modules/knowledge/    # pgvector 模型、仓储、检索与 reindex 服务
│   ├── modules/ingestion/    # 安全上传、文档/任务状态机与统一摄取流水线
│   ├── modules/rag/          # RAG 运行、引用快照与持久化聊天
│   ├── modules/users/        # 用户模型、仓储与管理服务
│   ├── services/             # LLM、RAG、导入器
│   └── utils/                # Prompt 构建、文本清洗、数学后处理
├── frontend/                 # Vue 3 + TypeScript + Vite 单页应用
├── data/
│   ├── raw/                  # 原始知识库 JSONL
│   ├── processed/            # 预处理后的 chunk JSONL
│   └── index/                # 仅供历史 evaluation/回滚审计的冻结工件
├── scripts/                  # 构建、导入、验证、调试脚本
├── tests/                    # pytest 测试
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # 生产依赖，不包含 FAISS
├── requirements-evaluation.txt
├── run.py
└── README.md
```

核心数据流：

```mermaid
flowchart LR
  A["data/raw/math_knowledge_seed.jsonl"] --> B["scripts.build_kb"]
  B --> C["data/processed/kb_chunks.jsonl"]
  C --> D["scripts.import_legacy_knowledge"]
  D --> E["PostgreSQL knowledge tables"]
  E --> F["scripts.reindex_knowledge"]
  F --> V["pgvector embeddings"]
  Q["用户问题"] --> P["LLM 检索规划"]
  P --> R["KnowledgeSearchService"]
  V --> R
  R --> L["LLM 生成结构化答案"]
  L --> U["Web 前端 / API 响应"]
```

---

## 环境要求

推荐环境：

- Python 3.11+
- Node.js 24.11.1 与 npm 11+
- PostgreSQL 及 pgvector 扩展
- 可用的 OpenAI-Compatible Embedding API
- 可用的 DeepSeek 或 OpenAI-Compatible Chat API
- 如需 Docker 部署：Docker / Docker Compose

安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

Linux / macOS：

```bash
source .venv/bin/activate
pip install -r requirements.lock.txt
```

---

## 配置 `.env`

可以从示例文件复制：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

示例配置：

```env
# App
APP_NAME=MathRAG MVP
APP_HOST=127.0.0.1
APP_PORT=8000
APP_ENV=development
APP_WORKERS=1
DEBUG=true
SESSION_SECRET=
SESSION_TTL_SECONDS=604800
ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

# PostgreSQL
DATABASE_URL=postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag
TEST_DATABASE_URL=postgresql+asyncpg://mathrag:mathrag-dev-only@127.0.0.1:5432/mathrag_test

# Embedding，要求兼容 OpenAI embeddings 接口
EMBEDDING_API_KEY=sk-xxxx
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
EMBEDDING_BATCH_SIZE=10
EMBEDDING_TIMEOUT=60
EMBEDDING_NORMALIZE=true

# LLM，要求兼容 OpenAI chat.completions 接口
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-reasoner
LLM_TIMEOUT=600
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.2
LLM_RETURN_REASONING=false

# Retrieval
TOP_K=3

# Ingestion
UPLOAD_DIR=data/uploads
FRONTEND_DIST_DIR=frontend/dist
MAX_UPLOAD_BYTES=10485760
MAX_PDF_PAGES=200
MAX_INGESTION_TEXT_CHARS=200000
INGESTION_CHUNK_CHARS=4000
```

说明：

- `EMBEDDING_DIMENSIONS` 固定为 `1024`，必须与数据库列及实际 embedding 模型一致。
- 在线检索只读取 PostgreSQL/pgvector，不读取 `data/index` 下的历史工件。
- development 使用 `mathrag_session`/`mathrag_csrf` Cookie，可在本机 HTTP 调试；`SESSION_SECRET` 留空时使用进程内开发值，不能用于共享环境。
- staging/production 的 `SESSION_SECRET` 必须至少 32 个 UTF-8 字节，`ALLOWED_ORIGINS` 必须显式配置且不能包含 `*`。
- staging/production 使用 `__Host-mathrag_session`/`__Host-mathrag_csrf`，Cookie 带 `Secure; SameSite=Lax; Path=/`，Session Cookie 额外带 `HttpOnly`。
- `SESSION_TTL_SECONDS` 必须大于 0，默认 604800 秒（7 天）。
- `UPLOAD_DIR` 是受控 PDF 根目录；Compose 中固定为持久卷 `/app/data/uploads`。数据库只保存相对路径，API 不返回路径。
- `FRONTEND_DIST_DIR` 指向 Vue 生产构建产物；本地默认是 `frontend/dist`，容器内固定为镜像中的 `/app/frontend/dist`。
- `MAX_UPLOAD_BYTES`、`MAX_PDF_PAGES`、`MAX_INGESTION_TEXT_CHARS` 和 `INGESTION_CHUNK_CHARS` 都必须大于 0。
- `/api/v1/chat` 从数据库加载历史，不接受客户端提供的 `history`。

---

## 快速启动

先启动数据库、执行迁移，并按下文导入和 reindex 知识数据：

```bash
docker compose up -d postgres
alembic upgrade head
python -m scripts.import_legacy_knowledge
python -m scripts.reindex_knowledge
```

迁移完成后，通过交互式密码输入创建首个管理员或教师；密码不会出现在命令行参数和 shell 历史中：

```powershell
python -m scripts.create_user --username admin --role admin
python -m scripts.create_user --username teacher01 --email teacher01@example.local --role teacher
```

### 账号与角色

- 管理员可以创建和管理学生、教师、管理员。
- 教师只能创建并管理自己创建的学生。
- 学生没有账号管理入口。
- 新建账号和重置密码后，用户必须在首次登录时修改临时密码。
- 系统不提供公开注册和账号物理删除。

然后分别启动 FastAPI 和 Vite。第一个终端运行：

```bash
python run.py
```

或使用开发模式：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

第二个终端运行：

```powershell
Set-Location frontend
npm.cmd ci
npm.cmd run dev
```

Vite 将 `/api`、`/health` 和 `/openapi.json` 代理到 `127.0.0.1:8000`。开发页面使用 <http://127.0.0.1:5173/>；FastAPI 的 `/` 只在已有 `frontend/dist` 时提供生产 SPA。

访问：

- Vue 开发页：<http://127.0.0.1:5173/>
- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

---

## 知识库格式

原始知识库文件：

```text
data/raw/math_knowledge_seed.jsonl
```

每一行是一个 JSON 对象，字段顺序固定为：

```text
id, category, title, keywords, content, example, steps, difficulty
```

示例：

```json
{"id":"k0001","category":"quadratic_equation","title":"因式分解法解一元二次方程","keywords":["一元二次方程","因式分解"],"content":"当方程可以写成 \\(ab=0\\) 的形式时，可令每个因式分别为 0 来求解。","example":"解方程 \\(x^2+4x+3=0\\)。","steps":["把方程因式分解为 \\((x+1)(x+3)=0\\)。","分别令 \\(x+1=0\\)、\\(x+3=0\\)。","得到 \\(x=-1\\) 或 \\(x=-3\\)。"],"difficulty":"easy"}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 知识点 ID，格式如 `k0001` |
| `category` | string | 分类，建议使用中文或 snake_case |
| `title` | string | 知识点标题 |
| `keywords` | string[] | 关键词，不能为空 |
| `content` | string | 核心知识内容，不能为空 |
| `example` | string | 示例，可为空字符串 |
| `steps` | string[] | 理解或解题步骤，不能为空 |
| `difficulty` | string | `easy` / `medium` / `hard` |

当前新版 schema **不再使用** `stage`、`course`、`prerequisites`。

公式规范：

- 行内公式：`\(...\)`
- 块级公式：`\[...\]`
- 不建议继续新增 `$...$` 或 `$$...$$`

---

## 导入知识库并构建 pgvector 向量

### 1. 校验种子知识库

```bash
python -m scripts.validate_seed_jsonl
```

指定输入和错误输出：

```bash
python -m scripts.validate_seed_jsonl \
  --input data/raw/math_knowledge_seed.jsonl \
  --error-output data/raw/seed_validate_errors.jsonl
```

### 2. 构建 chunk

```bash
python -m scripts.build_kb
```

自定义路径：

```bash
python -m scripts.build_kb \
  --input data/raw/math_knowledge_seed.jsonl \
  --output data/processed/kb_chunks.jsonl
```

生成字段包括：

- `chunk_id`
- `source_id`
- `category`
- `title`
- `keywords`
- `content`
- `example`
- `steps`
- `difficulty`
- `source_line`
- `retrieval_text`
- `answer_context`
- `metadata`

### 3. 执行数据库迁移

```bash
alembic upgrade head
```

### 4. 幂等导入历史知识

```bash
python -m scripts.import_legacy_knowledge
```

该命令读取 raw seed 与 processed chunk，写入 PostgreSQL；重复执行会按已有状态跳过，不维护第二套在线索引。

### 5. 生成或刷新 pgvector 向量

```bash
python -m scripts.reindex_knowledge
```

reindex 会批量调用 Embedding Provider，并把当前模型下成功的 chunk 更新为 `ready`。重复执行时只处理仍需更新的记录。

---

## 知识摄取

M5 的在线上传和两个 CLI 都调用同一套 `IngestionService`，最终写入 PostgreSQL/pgvector。`data/raw` 与 `data/processed` 仅保留历史迁移输入；新知识不会写入 JSONL。

### 从公开数学站点导入

`--requested-by` 必须是数据库中的 active admin 用户名。命令同步等待任务完成，失败时返回非零退出码。

```bash
python -m scripts.import_math_knowledge \
  --sources wikipedia wikibooks \
  --keywords derivative \
  --limit-per-source 2 \
  --category calculus \
  --max-chunk-chars 6000 \
  --delay-seconds 1.0 \
  --requested-by admin
```

`--sources` 可选 `proofwiki`、`planetmath`、`wikibooks`、`wikipedia`；`--keywords` 与 `--requested-by` 必填。该 CLI 不再提供 `--output` 或 `--error-output`。

### 从本地 PDF 导入

将 PDF 放入 `data/data_lake/`，再执行：

```bash
python -m scripts.import_pdf_knowledge \
  --data-dir data/data_lake \
  --max-chunks 20 \
  --category "高中数学" \
  --requested-by admin
```

`--no-recursive` 可关闭子目录扫描；`--max-chunks` 在统一流水线中表示本次最多创建的 PDF 导入任务数。每个文件都经过扩展名、MIME、magic bytes、大小、页数、加密和文本有效性校验。该 CLI 不再生成 text/error/seed JSONL。

### 文本抽取预览

`POST /api/knowledge/extract` 只保留管理员预览能力。`save=false` 返回抽取结果；`save=true` 固定返回 410，不能绕过统一摄取流水线写入 JSONL。

---

## 公式与 LaTeX 处理

项目统一推荐使用 KaTeX 分隔符：

- 行内：`\(...\)`
- 块级：`\[...\]`

如果旧数据中有 `$...$` 或 `$$...$$`，可以先 dry-run：

```bash
python -m scripts.normalize_latex_delimiters
```

确认后写回并自动备份：

```bash
python -m scripts.normalize_latex_delimiters --write
```

指定输入/输出：

```bash
python -m scripts.normalize_latex_delimiters \
  --input data/raw/math_knowledge_seed.jsonl \
  --output data/raw/math_knowledge_seed.normalized.jsonl \
  --write
```

如果 chunk 元数据、数量、顺序或 `retrieval_text` 发生变化，应重新构建 processed 输入、幂等导入并 reindex：

```bash
python -m scripts.build_kb
python -m scripts.import_legacy_knowledge
python -m scripts.reindex_knowledge
```

---

## 调试脚本

### 仅测试检索

```bash
python -m scripts.demo_query --question "x^2+4x+3=0 怎么解？" --show-context
```

交互模式：

```bash
python -m scripts.demo_query --interactive --show-context
```

### 测试完整 RAG 链路

```bash
python -m scripts.test_rag \
  --question "x^2+4x+3=0 怎么解？" \
  --show-references
```

打印完整 JSON：

```bash
python -m scripts.test_rag \
  --question "x^2+4x+3=0 怎么解？" \
  --show-full-json
```

### 离线 FAISS/pgvector 对账

生产依赖与 Docker 镜像不安装 FAISS。只有需要复核冻结历史工件时才安装 evaluation 依赖：

```bash
pip install -r requirements-evaluation.lock.txt
python -m scripts.evaluate_pgvector_retrieval \
  --fixture tests/fixtures/retrieval_questions.json \
  --output docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json \
  --replace-existing
```

evaluation 使用同一批 query vectors 对账只读 legacy FAISS 与 pgvector；不得把历史 FAISS 工件重新接回在线请求路径。

### 回滚

发布前必须在同一维护窗口保存 PostgreSQL 与 `upload_data`，两者组成同一个恢复点；只恢复其中一项会造成 document 元数据与 PDF 文件不一致：

```bash
docker compose exec -T postgres pg_dump -U mathrag -d mathrag -Fc > mathrag.dump
docker compose exec -T mathrag tar -C /app/data/uploads -czf - . > upload_data.tar.gz
```

M5 回滚时先停止文档上传、知识写入和导入 CLI，等待运行中任务结束或明确失败，再停止应用并备份上述两项。优先回滚应用镜像；只有确认不再需要 M5 的 document/job/知识 revision 数据后，才在备份数据库上验证 `alembic downgrade 0004_create_identity_conversation_rag_tables`。恢复时先恢复 PostgreSQL，再将匹配的 `upload_data.tar.gz` 解压到空上传卷，最后执行 `alembic upgrade head` 并检查 live/ready。

同时保留上一版本容器镜像和冻结的 `data/index` 工件。发生检索回归时按以下顺序执行：

1. 先按入口网关/负载均衡平台的 runbook 停止新流量，并暂停知识写入入口。
2. 停止当前应用，数据库保持运行：

```bash
docker compose stop mathrag
```

3. 优先部署上一版本镜像：

```bash
export MATHRAG_ROLLBACK_IMAGE="<上一版本镜像>"
docker pull "${MATHRAG_ROLLBACK_IMAGE}"
docker image tag "${MATHRAG_ROLLBACK_IMAGE}" mathrag:local
docker compose up -d --no-build mathrag
```

没有可用镜像时，可从 M2 固定基点构建；最终 baseline 会记录验收时的精确回滚 SHA：

```bash
git switch --detach cd77635
docker build --pull=false -t mathrag:local .
docker compose up -d --no-build mathrag
```

4. 冻结的旧 FAISS 工件只读使用，禁止重建或覆盖：

```bash
chmod a-w data/index/faiss.index data/index/id_map.json
```

5. 使用回滚环境变量验证存活和就绪状态，不在文档中固化真实地址：

```bash
export MATHRAG_BASE_URL="<回滚环境地址>"
curl -fsS "${MATHRAG_BASE_URL}/health/live"
curl -fsS "${MATHRAG_BASE_URL}/health/ready"
```

两个 health 检查都成功后才恢复入口流量和知识写入。禁止同一在线版本长期并行 FAISS 与 pgvector 两条检索路径；数据迁移需要回滚时按 Alembic 版本和数据库备份单独执行，不修改冻结的 evaluation 工件。

---

## API 示例

### 健康检查

```http
GET /health
```

响应：

```json
{
  "status": "ok",
  "app_name": "MathRAG MVP"
}
```

### 登录、CSRF 与当前用户

登录会设置 HttpOnly Session Cookie 和前端可读的 CSRF Cookie：

```http
POST /api/v1/auth/login
Content-Type: application/json
Origin: http://localhost:8000

{"username":"alice","password":"<交互输入的密码>"}
```

使用 curl 时可保存 Cookie，再从 cookie jar 读取 CSRF 值并发送修改请求。下面的 `<csrf-cookie-value>` 只是占位符：

```bash
curl -c cookies.txt -H "Origin: http://localhost:8000" \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"<password>"}' \
  http://127.0.0.1:8000/api/v1/auth/login

curl -b cookies.txt -H "Origin: http://localhost:8000" \
  -H "X-CSRF-Token: <csrf-cookie-value>" \
  -H "Content-Type: application/json" \
  -d '{"title":"新对话"}' \
  http://127.0.0.1:8000/api/v1/conversations
```

PowerShell 可用同一个 `WebRequestSession` 保存 Cookie：

```powershell
$mathragSession = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$mathragOrigin = 'http://localhost:8000'
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/auth/login' `
  -WebSession $mathragSession -Headers @{Origin=$mathragOrigin} `
  -ContentType 'application/json' `
  -Body '{"username":"alice","password":"<password>"}'
$mathragCsrf = $mathragSession.Cookies.GetCookies('http://127.0.0.1:8000')['mathrag_csrf'].Value
```

`GET /api/v1/auth/me` 返回当前用户；`POST /api/v1/auth/logout` 需要同样的 Origin 与 CSRF 头，并撤销服务端 Session。

### 知识 CRUD

登录用户可读取知识；普通用户的列表和详情只返回 `public+ready`，管理员可读取全部状态和可见性。创建、更新和归档只允许管理员，并要求 Origin 与 CSRF：

```text
GET    /api/v1/knowledge-items?status=ready&visibility=public&category=代数&page=1&page_size=20
POST   /api/v1/knowledge-items
GET    /api/v1/knowledge-items/{item_id}
PATCH  /api/v1/knowledge-items/{item_id}
DELETE /api/v1/knowledge-items/{item_id}?revision={revision}
```

创建请求包含 `category/title/keywords/content/example/steps/difficulty/visibility`。更新请求只提交需要修改的字段，但必须携带当前 `revision`；并发修改时旧 revision 返回 `KNOWLEDGE_REVISION_CONFLICT`/409。DELETE 是带 revision 的逻辑归档，归档后不会再被 RAG 检索。

### PDF 上传与摄取任务

管理员使用登录阶段获得的同一 Session 和 CSRF 值上传 PDF：

```powershell
$mathragUpload = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/v1/documents' `
  -WebSession $mathragSession -Headers @{Origin=$mathragOrigin; 'X-CSRF-Token'=$mathragCsrf} `
  -Form @{file=Get-Item 'C:\data\lesson.pdf'; category='高中数学'}
$mathragJobId = $mathragUpload.job.id
Invoke-RestMethod -Method Get `
  -Uri "http://127.0.0.1:8000/api/v1/ingestion-jobs/$mathragJobId" `
  -WebSession $mathragSession
```

上传成功返回 202，响应包含安全的 `document` 与初始 `job`，不包含存储路径。客户端轮询 job，直到 `completed`、`failed` 或 `cancelled`；`progress` 范围为 0~100。管理接口为：

```text
GET  /api/v1/documents?status=ready&page=1&page_size=20
GET  /api/v1/ingestion-jobs/{job_id}
POST /api/v1/ingestion-jobs/{job_id}/cancel
POST /api/v1/ingestion-jobs/{job_id}/retry
```

cancel 只接受 pending，retry 只接受 failed；两个写操作均要求管理员 Origin 与 CSRF。retry 复用原 job/document/item/chunk，不重复抽取知识。`error_code` 是稳定分类，`error_message` 只包含脱敏摘要。

### Conversation 与持久化问答

先创建会话，再为每次逻辑请求生成一个 UUID 作为 `client_request_id`：

```http
POST /api/v1/chat
Content-Type: application/json
Origin: http://localhost:8000
X-CSRF-Token: <csrf-cookie-value>

{
  "conversation_id": "<conversation-uuid>",
  "client_request_id": "<request-uuid>",
  "question": "x^2+4x+3=0 怎么解？",
  "top_k": 3
}
```

网络重试必须复用原 `client_request_id`。已完成请求返回相同 message/run ID 和回答；仍在运行返回 `RAG_REQUEST_IN_PROGRESS`/409；已失败或取消的请求返回已保存的稳定错误，不再次调用外部服务。会话接口包括：

成功响应保留 `question/answer/steps/used_knowledge/related_questions/references/agentic_plan/reasoning_content`，并增加 `conversation_id`、`question_message_id`、`answer_message_id`、`rag_run_id` 和 `client_request_id`。

```text
GET    /api/v1/conversations
POST   /api/v1/conversations
GET    /api/v1/conversations/{conversation_id}
PATCH  /api/v1/conversations/{conversation_id}
DELETE /api/v1/conversations/{conversation_id}
GET    /api/v1/conversations/{conversation_id}/messages
```

错误响应携带 `error.code` 和 `error.request_id`。失败问题会保留 user/completed 消息，助手占位消息变为 failed；客户端可读取消息列表并使用同一 request ID 重放，运维侧使用 request ID 与返回的资源 ID 关联日志和数据库记录。

### 知识抽取

该接口仅允许管理员携带 Session、Origin 和 CSRF 进行预览。`save` 默认 `false`；`save=true` 固定返回 410，不能再追加 JSONL。

```http
POST /api/knowledge/extract
Content-Type: application/json
```

请求：

```json
{
  "text": "函数 y=kx+b 且 k 不等于 0 时称为一次函数。",
  "category": "函数",
  "save": false
}
```

响应：

```json
{
  "records": [
    {
      "id": "k0001",
      "category": "函数",
      "title": "一次函数的概念",
      "keywords": ["一次函数", "函数"],
      "content": "形如 \\(y=kx+b\\) 且 \\(k\\ne0\\) 的函数称为一次函数。",
      "example": "例如 \\(y=2x+3\\) 是一次函数。",
      "steps": ["识别表达式是否为 \\(y=kx+b\\)。", "检查 \\(k\\ne0\\)。"],
      "difficulty": "easy"
    }
  ],
  "saved_count": 0,
  "next_steps": []
}
```

---

## 前端说明

前端位于 `frontend/`，使用 Vue 3、TypeScript、Vite、Vue Router 和本地打包的 KaTeX。主要页面包括：

- 登录、持久化问答、历史恢复、重命名和归档对话；
- 管理员和教师用户管理、临时密码强制修改；
- 管理员知识 CRUD、revision 冲突处理、PDF 上传与摄取任务管理；
- 安全的本地 KaTeX 公式渲染、结构化错误与 request id；
- 桌面、平板和移动端响应式导航。

API 类型由 FastAPI OpenAPI 生成。后端契约变化后，在项目根目录运行：

```powershell
python scripts/export_openapi.py
Set-Location frontend
npm.cmd run api:check
```

`api:check` 会重新生成 `src/api/schema.d.ts`，并在 `openapi.json` 或类型文件存在漂移时失败。功能代码只能通过统一 API client 发请求；Session 使用 HttpOnly Cookie，CSRF 令牌从同源 Cookie 读取。

前端完整质量门禁：

```powershell
Set-Location frontend
npm.cmd run format:check
npm.cmd run lint
npm.cmd run typecheck
npm.cmd test -- --run
npm.cmd run build
npm.cmd run e2e
```

---

## Docker 部署

### 本地数据库开发

```powershell
Copy-Item .env.example .env
# 启动任何 Compose 服务前，先在 .env 中填写至少 32 字节的 SESSION_SECRET，
# 并按实际前端地址填写非通配的 ALLOWED_ORIGINS。
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

### 完整 Compose 启动

`docker-compose.yml` 会先用固定 Node 版本构建 Vue，再将 `frontend/dist` 复制到不含 Node.js 与 `node_modules` 的 Python runtime 镜像。本地 Compose 构建和运行使用 `mathrag:local` 镜像，并启动固定版本的 PostgreSQL/pgvector。首次创建数据库卷时，先执行迁移，再启动应用：

```powershell
Copy-Item .env.example .env
# Compose 的 mathrag 服务固定按 production 校验；先填写 SESSION_SECRET 和 ALLOWED_ORIGINS。
docker compose up -d postgres
.\.venv\Scripts\alembic.exe upgrade head
docker compose up -d --build mathrag
docker compose ps
```

生产容器由 FastAPI 同源提供 Vue SPA，访问 <http://127.0.0.1:8000/>。前端深层路由可直接刷新；未知 `/api/*` 保持 JSON 404，不会回落到 SPA。

M6 固定使用一个应用 worker。摄取任务由进程内 `BackgroundTasks` 执行，不具备跨进程恢复和分布式调度能力；多 worker、持久队列、OCR、限流与公网 TLS 留待 M7。

查看日志：

```bash
docker compose logs -f mathrag postgres
```

服务只绑定本机回环地址：

```text
127.0.0.1:8000 -> container:8000
127.0.0.1:5432 -> container:5432
```

停止服务：

```bash
docker compose down
```

---

## 测试

仅安装 `requirements.lock.txt` 的纯 runtime 环境不收集 FAISS evaluation 用例：

```bash
pytest -q --ignore=tests/evaluation --ignore=tests/test_retrieval_baseline.py
```

运行完整测试前安装包含 FAISS 的 evaluation 锁：

```bash
pip install -r requirements-evaluation.lock.txt
pytest -q
```

测试主要覆盖：

- `/api/v1/auth` Cookie、CSRF、角色与会话撤销
- Conversation owner 隔离、消息分页与归档
- `/api/v1/chat` 两段短事务、幂等重放、失败终态和引用快照
- `/api/knowledge/extract` 保存/预览逻辑
- `/api/v1/knowledge-items` 权限过滤、revision 冲突、创建/更新/归档与 RAG 可见性
- `/api/v1/documents` 安全上传，以及 ingestion job 的状态机、取消、失败收口和幂等重试
- RAG 多查询规划与 Knowledge Search 批量检索
- PostgreSQL/pgvector 导入、reindex、检索与运行时依赖边界
- 数学知识导入器
- PDF 知识导入器

---

## 常见问题

### `ModuleNotFoundError: No module named 'app'`

请在项目根目录运行命令，并优先使用模块方式：

```bash
python -m scripts.build_kb
python -m scripts.import_legacy_knowledge
python -m scripts.reindex_knowledge
```

### `ModuleNotFoundError: No module named 'dotenv'`

说明当前 Python 环境没有安装项目依赖。先激活虚拟环境并安装依赖：

```bash
pip install -r requirements.lock.txt
```

### `LLM_API_KEY` 或 `EMBEDDING_API_KEY` 缺失

检查 `.env` 是否存在，并确认 key 名称正确：

```env
LLM_API_KEY=...
EMBEDDING_API_KEY=...
```

### pgvector 检索没有返回知识

先确认数据库迁移已到最新版本，再幂等导入并 reindex：

```bash
alembic upgrade head
python -m scripts.import_legacy_knowledge
python -m scripts.reindex_knowledge
```

同时确认当前 Embedding 模型、固定 1024 维契约和数据库中 chunk 的 `ready` 状态一致。

### 公式没有渲染

检查：

1. `frontend/dist` 是否由当前源码成功构建，KaTeX 是否随产物一起加载。
2. 公式是否使用 `\(...\)` 或 `\[...\]`。
3. JSON 字符串里的反斜杠是否正确转义。

---

## 开发约定

- 新知识库记录只使用新版字段：`id/category/title/keywords/content/example/steps/difficulty`。
- 新增公式统一使用 KaTeX LaTeX 分隔符。
- 修改 `math_knowledge_seed.jsonl` 后，应依次执行校验、构建 chunk、幂等导入和 reindex。
- 在线代码只依赖 PostgreSQL/pgvector；FAISS 仅允许出现在 evaluation 依赖和离线对账脚本中。
- 临时 PDF、依赖、备份文件已通过 `.gitignore` / `.dockerignore` 排除。

---

## License

本项目主要用于教学、演示与研究原型。
