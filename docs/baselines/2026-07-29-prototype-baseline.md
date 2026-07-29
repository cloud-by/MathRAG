# MathRAG M0 原型基线报告

- 日期：2026-07-29
- 基线提交：`ca837f1`
- 工作分支：`codex/m0-baseline`
- 目标：证明现有原型可在固定环境中安装、测试、运行、检索和渲染，并为 M1 留下可复跑证据。

## 结论

M0 基线通过。Python 3.11 环境可复现安装；原有 20 项测试通过；M0 收口后全量 39 项测试通过。真实 `/health`、`/api/chat`、`/api/knowledge/extract` 均返回 200；固定 26 题的 FAISS Top-3 期望命中为 26/26；桌面与移动端页面能够展示真实回答、3 条参考知识和 KaTeX 公式。

M0 未引入 PostgreSQL、业务表、认证或 pgvector 在线检索，也未改变 `/api/chat` 和 FAISS 生产行为。

## 环境

| 项目 | 实际值 |
|---|---|
| 操作系统 | Windows，PowerShell |
| Python | CPython 3.11.9 |
| uv | 0.11.24 |
| 虚拟环境 | `.venv` |
| 完整依赖数 | 43 |
| 依赖快照 | `requirements.lock.txt` |
| 依赖快照 SHA-256 | `b53f553cb3b1045ff7a40908c5b977e1ce0feb591da36c32ced020b543795db1` |

环境创建与安装命令：

```powershell
uv venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

首次仅运行原仓库测试时结果为 `20 passed, 1 warning`；加入 M0 基线契约后最终结果为 `39 passed, 1 warning`。

## 数据与索引

| 工件 | 数量/属性 | 工作树文件 SHA-256 |
|---|---:|---|
| `data/raw/math_knowledge_seed.jsonl` | 26 条 | `2593f45081b11ab4ae280d1a7fb107791b3099c364f3813f215a73fa7369d062` |
| `data/processed/kb_chunks.jsonl` | 26 条 | `a0334a626d7e54ce04a447861af1616da26ad8b012d81f6720aa1d404539e5aa` |
| `data/index/id_map.json` | 26 项 | `6fe97be89ad8398d4ed636545b4d7939b5832f93c5602a2d65b4a40781cf8331` |
| `data/index/faiss.index` | 26 向量，1024 维，Inner Product | `e2520504ff2b392bbb56aea792046a752a217a4abf75ca8dec516fd219149192` |

Windows 工作树的 seed 原始字节使用 CRLF，原始哈希会随 Git 换行策略变化。固定题集因此使用规范化 UTF-8/LF 内容哈希：

```text
b87355849f828ae219ba4e03315436d65a1fce749db96740ae645a74c231e4b0
```

该值与 Git 中 LF blob 一致，可跨 Windows/Linux 复跑。固定问题集 `tests/fixtures/retrieval_questions.json` 包含 26 道中文问题，题号、字段、`top_k=3` 和期望 legacy ID 均执行严格校验。

## 测试结果

最终命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

结果：

```text
39 passed, 1 warning
```

新增覆盖：

- 固定问题集与 seed 绑定、未知字段/ID、中文问题和规范题号校验。
- Top-3 输出字段、期望命中率、非有限分数拒绝和敏感元数据排除。
- CLI 输出、UTC 时间、工件哈希和非标准 JSON 防护。
- `run.py` 使用 `settings.DEBUG`、`.env.example` 关键字段和 Docker 文档契约。

唯一警告来自 FastAPI TestClient 导入链：Starlette 提示当前 `httpx` 适配已弃用并建议迁移到 `httpx2`。M0 不升级整套 Web 测试依赖，该项进入后续依赖维护。

## 真实检索基线

执行命令：

```powershell
.\.venv\Scripts\python.exe -m scripts.capture_retrieval_baseline `
  --fixture tests\fixtures\retrieval_questions.json `
  --output docs\baselines\artifacts\faiss-top3-2026-07-29.json
```

注意：必须以 `python -m` 从项目根目录运行；直接执行 `python scripts\capture_retrieval_baseline.py` 不会自动把项目根目录加入模块搜索路径。

结果：

| 指标 | 结果 |
|---|---:|
| 问题数 | 26 |
| 期望知识进入 Top-3 | 26 |
| 期望命中率 | 100% |
| 开始时间 UTC | `2026-07-29T08:43:06.132082Z` |
| 结束时间 UTC | `2026-07-29T08:43:14.585285Z` |
| Embedding 模型 | `text-embedding-v4` |
| 维度 | 1024 |
| Provider origin 指纹 | `ec1f29107de2c3a3df64b95fe366d7c640c5a610fdc5b6804611630835b4dafc` |

输出工件 SHA-256：`09d206784542f0687c32b5e5edb879b4068f3f2a6ca972f8cf5684ba183d2c9a`。

## API 烟雾测试

服务命令：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765
```

| 接口 | 输入 | 结果 |
|---|---|---|
| `GET /health` | 无 | 200，`status=ok`，`app_name=MathRAG MVP` |
| `POST /api/chat` | 正弦定义与恒等式，`top_k=3` | 200，答案 212 字符，3 条引用 |
| `POST /api/knowledge/extract` | 正弦函数短文本，`save=false` | 200，抽取 1 条，`saved_count=0` |

响应工件：

- `docs/baselines/artifacts/chat-response-2026-07-29.json`，SHA-256 `6c3c4241c3301aa51784e522c46393ba7b5be16edd2dd484bfcf91e9e291ef54`
- `docs/baselines/artifacts/knowledge-extract-preview-2026-07-29.json`，SHA-256 `fc692e6948e8249e2f5cf3dd0da17baa6bb2a37b790d8a4bbe578562b38cd086`

对所有 JSON 基线工件执行 `api_key|authorization|password|secret|bearer|sk-*` 扫描，未发现匹配。`.env` 未纳入 Git，也未输出到报告。

## 浏览器与公式渲染

使用 Playwright CLI 在真实页面提交同类正弦问题并等待 `status=已完成`：

- 回答区域文本长度：682 字符。
- KaTeX 节点：122 个。
- 参考知识容器：3 个直接子项。
- 390×844 视口：无横向溢出。
- 唯一控制台错误：`/favicon.ico` 返回 404，不影响问答和公式渲染。

截图：

| 状态 | 尺寸 | 文件 | SHA-256 |
|---|---:|---|---|
| 桌面空闲态 | 1440×1000 | `docs/baselines/assets/frontend-idle-desktop.png` | `1895fb843af083f5aebd384da48a7d06081ae70ce5535dd6faf358916dac0d35` |
| 桌面回答态 | 1440×3398 | `docs/baselines/assets/frontend-answer-desktop.png` | `8f78a9d781445ad7c34c08f4bcb9d94be6fae704bdd16a4ead67ddf682f37580` |
| 移动回答态 | 390×6006 | `docs/baselines/assets/frontend-answer-mobile.png` | `128720d16cbe4785856ec7f69cd7b5eaefa1273983acaa490471e783551f9722` |

## M0 修复与决策

- 修复 `run.py` 读取不存在的 `APP_DEBUG`，现在由 `settings.DEBUG` 控制 reload。
- `.env.example` 补齐应用、调试和 Top-K 配置，示例密钥保持假值。
- README 区分远端 Compose 镜像与当前源码的本地 `docker build`/`docker run`。
- 建立 43 项精确版本依赖快照。
- ADR-0001 接受总体架构为实施基线，并冻结 M1 数据库版本、安全会话、CSRF、上传、RPO/RTO 和性能目标。
- 生成仅覆盖 M1 数据库基础设施的代码级 TDD 计划。

## 已知问题与后续归属

- Starlette/httpx 弃用警告：在 M1 依赖安装后单独验证兼容组合，不在 M0 盲目升级。
- `/favicon.ico` 404：前端静态资源阶段处理。
- 当前 CORS 为通配来源且允许凭据：认证 Cookie 上线前按 ADR-0001 改为显式来源。
- 当前前端 KaTeX 依赖公网 CDN：Vue/Vite 等价迁移时改为本地打包。
- 当前 LLM/Embedding 调用与 Gunicorn 超时层级尚未统一：M7 处理。
- 锁文件只有精确版本，没有 wheel hash：当前能复建版本集合，但尚未达到供应链制品哈希锁定。
- 本轮未执行 Docker 镜像构建或 Compose 健康测试；这属于 M1 数据库底座的明确验收项。

## M0 验收清单

- [x] Python 3.11 隔离环境与依赖快照。
- [x] 原有 20 项测试通过，最终 39 项测试通过。
- [x] `/health`、真实 Chat、只预览不写入的知识抽取通过。
- [x] 26 道固定问题与 legacy ID 契约。
- [x] FAISS Top-3 真实结果、API 响应和三张页面截图。
- [x] 规范化数据哈希、工件哈希和敏感字段扫描。
- [x] ADR-0001 与 M1 代码级 TDD 计划。
- [x] 未迁移业务数据、未切换检索、未修改现有 API 契约。
