# ADR-0001：M0 工程与架构实施基线

- 状态：已接受
- 决策日期：2026-07-29
- 适用范围：MathRAG M1-M7
- 依据：`2026-07-29-mathrag-project-architecture-design.md`

## 背景

当前仓库是可运行的单用户 MathRAG 原型，在线知识链路由 JSONL、FAISS 与 `id_map.json` 组成。后续目标是演进为 Vue 3 + TypeScript 前端、模块化单体 FastAPI 后端和 PostgreSQL + pgvector 唯一在线事实数据源。

在进入数据库改造前，必须冻结可复现环境、安全边界和阶段验收指标，避免各阶段按临时偏好选择版本或修改跨模块契约。

## 总体架构决策

接受 2026-07-29 总体架构设计作为 M1-M7 的实施基线。文档中的“设计草案”从本 ADR 接受之日起视为已批准的实施输入；后续冲突由编号更高的 ADR 显式替代，不以未记录的代码偏差改变架构。

采用以下边界：

- Vue 3 + TypeScript + Vite 前端，生产环境与 API 同源。
- 模块化单体 FastAPI，不在第一阶段拆分微服务。
- PostgreSQL + pgvector 是目标在线唯一事实数据源。
- 初始部署为一个异步 Web Worker 和一个 PostgreSQL 实例。
- Redis、Celery、独立向量数据库、Kubernetes 不在 M1-M7 初始范围。
- M3 切换前保留现有 FAISS 在线路径；M1 不改聊天或检索实现。

## 运行时与依赖版本

### 当前原型基线

- Python：3.11.9
- 依赖快照：仓库根目录 `requirements.lock.txt`
- 固定安装命令：`uv pip install --python .venv/Scripts/python.exe -r requirements.lock.txt`
- `requirements.txt` 继续表达直接依赖；锁文件表达 M0 已验证的完整环境。

### M1 数据库组合

M1 使用以下精确版本，升级必须单独通过空库迁移、健康检查和全量回归：

- PostgreSQL：18.4
- pgvector 扩展：0.8.5
- Compose 数据库镜像：`pgvector/pgvector:0.8.5-pg18-bookworm`
- SQLAlchemy：2.0.51
- Alembic：1.18.5
- asyncpg：0.31.0
- pgvector Python 包：0.5.0

选择稳定版本，不采用 SQLAlchemy 2.1 预发布版本。PostgreSQL 与镜像使用不可变的完整标签，不使用 `latest`、`pg18` 等浮动标签。

M1 完成后重新生成完整锁文件，并在报告中记录 Python、PostgreSQL server、扩展和 Python 包的实际版本。

## 数据库与迁移约束

- 只有 Alembic migration 可以修改生产 schema；应用启动不得调用 `create_all()`。
- Web 请求使用一个进程级 `AsyncEngine`，每个请求创建独立 `AsyncSession`。
- `AsyncSession` 不跨请求共享；Repository 不调用 `commit()`。
- Service 决定提交点；异常路径回滚，请求结束关闭 Session。
- 不在等待 LLM、Embedding 或文件处理期间持有数据库事务。
- M1 的首个 migration 只启用 `vector` 扩展，不创建业务表。
- M1 必须验证空库 `upgrade head` 以及 `downgrade base -> upgrade head` 回环。

## 认证与会话基线

该部分在 M4 实现，但从 M0 起冻结契约：

- 密码哈希：`pwdlib[argon2]==0.3.0`，使用 `PasswordHash.recommended()`；数据库仅存哈希。
- 会话：服务端随机会话，不使用 JWT 作为浏览器登录态。
- 原始令牌：`secrets.token_urlsafe(32)` 生成，至少 256 bit 随机性。
- 数据库存储：仅存原始令牌的 SHA-256，不存原始值。
- 生产 Cookie 名：`__Host-mathrag_session`。
- 生产 Cookie 属性：`HttpOnly; Secure; SameSite=Lax; Path=/`，不设置 `Domain`。
- 会话 TTL：7 天；退出、禁用用户、密码重置时撤销相关会话。
- 本地 HTTP 开发允许通过显式 `APP_ENV=development` 使用非 `__Host-` Cookie 和 `Secure=false`；生产配置不得继承该降级。

## CSRF 与跨域基线

- 所有依赖 Cookie 身份且会修改状态的请求执行 CSRF 校验。
- 使用与当前会话绑定并由服务端签名的 double-submit token。
- 前端从非 HttpOnly 的 CSRF Cookie 读取 token，并发送 `X-CSRF-Token` 请求头。
- 后端同时验证签名、会话绑定、`Origin`；缺少 `Origin` 时验证 `Referer`。
- token 比较使用常量时间函数。
- `GET`、`HEAD`、`OPTIONS` 不改变业务状态。
- 生产 CORS 只接受 `ALLOWED_ORIGINS` 明确列出的来源；启用凭据时禁止 `*`。

## 上传与导入边界

- 在线单文件上限：20 MiB。
- 在线 PDF 页数上限：200 页。
- 单次在线请求最多 5 个文件。
- 同一用户同时运行的在线导入任务最多 2 个。
- 扩展名、MIME、文件签名、大小和 PDF 页数均需校验。
- 保存名由服务端生成 UUID；数据库只保存受控根目录下的相对路径。
- 大批量 PDF、网页和历史知识迁移继续使用管理员 CLI，不在 Web Worker 内无界执行。

## 可用性与恢复目标

- 数据库与上传目录备份目标 RPO：24 小时。
- 单实例恢复目标 RTO：4 小时。
- M7 前必须在独立临时环境完成至少一次数据库与上传目录恢复演练。
- `/health/live` 只表示进程存活，数据库异常时仍返回 200。
- `/health/ready` 验证关键配置、数据库连接和 `vector` 扩展；任一失败返回 503。
- 外部 LLM/Embedding 故障不影响 live；是否使 ready 降级由 M7 在运维策略中记录，M1 不将外部 API 纳入 ready。

## 性能与隔离门槛

- 首版验收负载：10 个并发用户提交聊天。
- 会话串用、权限泄漏和跨用户引用泄漏必须为 0。
- 不含外部 LLM/Embedding 时间的 API 与数据库部分 P95 不高于 300 ms。
- M3 的本地 pgvector 精确 Top-K 检索 P95 不高于 100 ms。
- 未经实测证明，不增加 Worker、不引入 Redis、不创建 HNSW 索引。

## 日志与敏感信息

- 日志和基线工件不得写入 API Key、Cookie、Authorization、密码、会话原始令牌或连接串凭据。
- Embedding 基线只记录模型、维度、归一化、距离方式和脱敏后的 provider origin SHA-256。
- 默认不记录完整聊天正文和知识正文；使用 request ID、记录 ID、错误码和分段耗时关联诊断。

## 结果

正面结果：后续阶段拥有一致的版本、安全和验收依据；M1 可以专注数据库底座，不必重复讨论跨阶段契约。

代价：依赖升级、Cookie/CSRF 方案和容量目标的改变必须更新 ADR 并重新验证相关阶段，不能静默漂移。

## 参考

- [PostgreSQL versioning policy](https://www.postgresql.org/support/versioning/)
- [pgvector Docker image tags](https://hub.docker.com/r/pgvector/pgvector/tags)
- [SQLAlchemy asyncio](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [pgvector Python SQLAlchemy examples](https://github.com/pgvector/pgvector-python#sqlalchemy)
- [FastAPI password hashing example](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [OWASP CSRF Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html)
