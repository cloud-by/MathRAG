# M1 数据库基础设施基线报告

## 基线信息

- 验证日期：2026-07-30（Asia/Shanghai）
- 分支：`codex/m1-database-foundation`
- 验证起点提交：`c0c1544d29fd043d60efcfbd75bd5d07f863b990`
- 验证范围：M1 配置契约、异步数据库基础设施、Compose、健康检查、Alembic 和 FAISS 回归
- 敏感信息：报告未记录 `.env` 内容、数据库连接串、密码或 API 密钥

## 版本基线

| 组件 | 实际版本 |
|---|---|
| 本地 Python | 3.11.9 |
| 容器 Python | 3.11.9 |
| FastAPI | 0.140.13 |
| SQLAlchemy | 2.0.51 |
| asyncpg | 0.31.0 |
| Alembic | 1.18.5 |
| pgvector Python | 0.5.0 |
| PyYAML | 6.0.3 |
| PostgreSQL Server | 18.4 |
| pgvector 扩展 | 0.8.5 |
| PostgreSQL 镜像 | `pgvector/pgvector:0.8.5-pg18-bookworm` |

应用基础镜像固定为 `python:3.11.9-slim`。构建时解析的基础镜像摘要为 `sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317`。

## Compose 验证

验证时两个服务均为 healthy：

| 服务 | 镜像 | 端口绑定 | 状态 |
|---|---|---|---|
| `postgres` | `pgvector/pgvector:0.8.5-pg18-bookworm` | `127.0.0.1:5432` | healthy |
| `mathrag` | `mathrag:local` | `127.0.0.1:8000` | healthy |

- `docker compose config --quiet`：通过
- `docker compose build --pull mathrag`：通过
- 应用 Worker 数量：1
- PG18 数据卷挂载点：`/var/lib/postgresql`
- 测试数据库初始化：`mathrag_test` 存在
- 收尾状态：已执行 `docker compose down`，测试容器和网络已移除，数据库卷保留

PG18 不再支持把持久卷直接挂载到旧路径 `/var/lib/postgresql/data`。真实启动验证发现该问题后，已增加契约测试并改为挂载父目录。

## 迁移验证

- 主库 `alembic current`：`0001_enable_vector_extension (head)`
- 测试库回环：`base -> head -> base -> head`
- 回环结果：1 项集成测试通过
- `head` 状态：pgvector 扩展版本为 0.8.5
- `base` 状态：pgvector 扩展不存在
- 应用启动不执行 `create_all()`，不自动修改 schema

## 健康检查

| 场景 | `/health/live` | `/health/ready` | ready 检查结果 |
|---|---:|---:|---|
| 数据库正常 | 200 | 200 | `config=ok`、`database=ok`、`pgvector=0.8.5` |
| 数据库停止 | 200 | 503 | `config=ok`、`database=unavailable`、`pgvector=unknown` |
| 数据库恢复 | 200 | 200 | 自动恢复为正常结果 |

失败响应不包含连接串、密码或底层数据库异常文本。合法的传入 `X-Request-ID` 会原样返回；不合法或缺失的值由应用生成。

## 自动化测试

- 迁移与健康 API 定向测试：5 passed
- Compose 契约测试：3 passed
- 全量测试：66 passed，1 warning
- `git diff --check`：通过

唯一未解决警告：Starlette TestClient 使用 `httpx` 的弃用提示。该警告在 M0 已存在，本阶段没有新增警告类型。

## FAISS 回归

- 固定题集：26 题
- 期望命中：26/26
- 命中率：100%
- 数据集 SHA-256：`b87355849f828ae219ba4e03315436d65a1fce749db96740ae645a74c231e4b0`
- 结果文件：`docs/baselines/artifacts/faiss-top3-m1-regression.json`

M1 新模块未引用 FAISS、`id_map` 或 `kb_chunks`，检索仍由现有 FAISS 路径负责。

## 反模式检查

- 未发现 `create_all(`。
- 未发现跨请求创建的全局 `AsyncSession(` 实例。
- 未发现 M1 业务表、`__tablename__` 或 `mapped_column`。
- `app/modules` 与 `app/infrastructure` 未引用 FAISS、`id_map` 或 processed JSONL。
- 数据库 Engine 和 sessionmaker 延迟创建；模块导入不连接数据库。

## 阶段边界

M1 只建立数据库底座和 pgvector 扩展，不包含知识、用户、会话、导入任务等业务表，也没有把聊天或检索流量切换到 PostgreSQL。首次数据库初始化仍需显式执行 `alembic upgrade head`；应用 readiness 会在迁移缺失时保持未就绪。
