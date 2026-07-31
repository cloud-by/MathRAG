# MathRAG M4 认证、会话与 RAG 持久化验收基线

## 1. 验收结论

M4 已完成服务端 Session 认证、`admin`/`user` 角色、CSRF 与显式 CORS、严格按用户隔离的 Conversation/Message，以及两段短事务的 RAG 运行、幂等键和引用快照持久化。最终全量测试 `514 passed, 0 skipped, 1 warning`；空测试库迁移回环、Compose 构建、live/ready 探针和容器清理均通过。

验收时间：2026-07-31（Asia/Shanghai）。

## 2. 版本与提交

| 项目 | 值 |
|---|---|
| M4 基点 / M3 回滚提交 | `bcf8e9d` |
| M4 任务 1-9 最后提交 | `402637d3f612cfa77e93c37de87c629c27e2cb79` |
| 最终验收与文档 | 本文所在提交 |
| 开发分支 | `codex/m4-auth-conversation-persistence` |
| Python | `3.11.9` |
| FastAPI / SQLAlchemy / asyncpg | `0.140.13` / `2.0.51` / `0.31.0` |
| PostgreSQL / pgvector Server / Python | `18.4` / `0.8.5` / `0.5.0` |
| pwdlib | `0.3.0` |
| runtime / evaluation 锁 | 53 / 54 个包 |
| runtime 镜像 | `mathrag:local` |
| runtime 镜像 ID | `sha256:75ebb90180e9a73cffd9aeac8570e5989f4e8902299ed0f59291d8ab961a6d85` |

evaluation 锁唯一额外包仍为 `faiss-cpu`；在线 import graph 和生产镜像不加载 FAISS。

## 3. 数据库与迁移

专用测试库先通过主库隔离守卫，再执行：

```text
alembic downgrade base
alembic upgrade head
alembic current
alembic check
```

结果：

- 空库完整升级至 `0004_create_identity_conversation_rag_tables (head)`。
- `alembic check` 输出 `No new upgrade operations detected.`。
- downgrade 依次移除 M4、M3、M2、M1 schema，upgrade 可完整恢复。
- M4 新增 `users`、`user_sessions`、`conversations`、`messages`、`rag_runs`、`rag_references`，并给 `knowledge_items` 增加 nullable `owner_id`。
- 跨模块外键模型可独立导入，reindex、检索和 CLI 不再依赖 `app.main` 的导入顺序。

## 4. 测试证据

| 分层 | 结果 |
|---|---:|
| auth/conversation/rag unit | 63 passed |
| API | 19 passed，1 个既有 warning |
| integration | 53 passed |
| 最终全量 | 514 passed，0 skipped，1 个既有 warning |

唯一 warning 为 Starlette 对当前 `httpx` TestClient 适配层的弃用提示；M4 未新增 warning。

关键行为证据：

- tx1 提交 `user/completed`、空正文 `assistant/pending`、`rag_run/running` 后关闭 Session，才进入 planner/Embedding/pgvector/LLM。
- tx2 使用 CAS 完成 assistant/run 并写引用；成功、failed、cancelled 均有稳定终态测试。
- tx2 异常使用新 Session 尝试失败收口；外部异常正文、连接串和供应商响应不进入 API 或数据库错误字段。
- 相同 `(conversation_id, client_request_id)` 在 running 时返回 409，completed 时返回相同 IDs/响应；并发测试只产生一条 run、两条 message，executor 调用一次。
- 修改或删除知识 chunk 后，历史引用 snapshot 内容保持不变，`chunk_id` 外键变为 NULL。

## 5. 隔离与安全

| 操作 | A 访问 A | A 访问 B | 未登录 |
|---|---:|---:|---:|
| Conversation GET/list | 允许 | 404 / 列表不含 B | 401 |
| PATCH/DELETE | 允许 | 404 | 401 |
| Message list | 允许 | 404 | 401 |
| `/api/v1/chat` | 允许 | 404 | 401 |
| admin dependency | admin 允许 | user 403 | 401 |

- 数据库只保存 Session token 的 SHA-256，不保存原始 token；公开 schema 不包含 `password_hash` 或 `token_hash`。
- production Cookie 名为 `__Host-mathrag_session` 和 `__Host-mathrag_csrf`；两者均为 `Secure; SameSite=Lax; Path=/`，Session Cookie 额外为 `HttpOnly`。
- 所有 Cookie 身份的状态修改接口校验签名 double-submit CSRF、Session 绑定和 Origin/Referer。
- production Compose 必须显式提供至少 32 字节的 `SESSION_SECRET` 和非通配 `ALLOWED_ORIGINS`；缺失时在启动前失败。
- `/api/knowledge/extract` 仅管理员可预览；`save=true` 在外部调用和文件访问前返回 410，不返回本机路径或供应商错误正文。

## 6. API 工作流

真实 PostgreSQL + fake provider 的完整 API 流程已通过：login -> Cookie/CSRF -> create conversation -> `/api/v1/chat` -> message list -> logout -> me 401。数据库对应一条 user message、一条 completed assistant message、一条 completed run 和一条引用快照。

旧 `/api/chat` 仅在 development 保持 M3 请求/响应兼容，带 `Deprecation: true` 和 successor `Link`；staging/production 固定返回 410。删除门为 M6 Vue 切换提交。

## 7. Compose 验收与最终状态

```text
mathrag-mathrag-1   Up (healthy)   127.0.0.1:8000->8000/tcp
mathrag-postgres-1  Up (healthy)   127.0.0.1:5432->5432/tcp
GET /health/live    200 {"status":"ok","app_name":"MathRAG MVP"}
GET /health/ready   200 config=ok,database=ok,pgvector=0.8.5
```

smoke 使用进程级临时非真实配置值，没有写入 `.env`、仓库或报告。验收后执行 `docker compose down`；最终 `docker compose ps --all` 列表为空。

## 8. 回滚与已知限制

1. 停止新登录和聊天写入，备份数据库。
2. 停止 M4 应用容器。
3. 回滚应用到 M3 固定提交 `bcf8e9d`。
4. 只有确认不再需要 M4 用户、Session、Conversation、Message 和 RAG 审计数据后，才执行 downgrade 到 `0003_enforce_vector_readiness`；该操作会删除 M4 表。
5. 验证 M3 `/health/live`、`/health/ready` 和旧 `/api/chat` 后再恢复流量。

M4 不包含 Vue 登录/会话界面、公开注册、Redis/JWT/OAuth、多 Worker 协调或同步 LLM SDK 的真正可中断 I/O。前端切换属于 M6，容量与 I/O 加固属于 M7。
