# MathRAG M5 知识管理与统一摄取验收基线

## 1. 验收结论

M5 已完成带权限和 revision 并发控制的知识管理 API、安全 PDF 上传、document/ingestion job 状态机、可重试摄取流水线，以及网页/PDF CLI 到统一 `IngestionService` 的切换。PostgreSQL/pgvector 仍是唯一在线知识事实源；M5 新数据不追加 seed、error 或 text chunk JSONL。

验收时间：2026-07-31（Asia/Shanghai）。

## 2. 版本与提交

| 项目 | 值 |
|---|---|
| M5 基点 / M4 main | `366383b73649fdf61adfdd8ef0f982fd75c0ef61` |
| M5 任务 1-10 最后提交 | `bcd9da3f7853c4cb47bbe50761df00c9d11d24b2` |
| 最终验收与文档 | 本文所在提交 |
| 开发分支 | `codex/m5-knowledge-ingestion-management` |
| Python | `3.11.9` |
| PostgreSQL / pgvector | `18.4` / `0.8.5` |
| runtime 镜像 | `mathrag:local` |

生产锁与镜像不安装 FAISS。runtime 全套按 README 固定忽略 `tests/evaluation` 和 `tests/test_retrieval_baseline.py`；离线 FAISS/pgvector 对账仍使用 `requirements-evaluation.lock.txt`，没有接回在线 import graph。

## 3. 数据库与迁移

专用 `mathrag_test` 测试库通过主库隔离守卫后完成 base/head 回环。Compose 主库只执行向前迁移和只读检查：

```text
alembic current
0005_create_documents_ingestion_jobs (head)

alembic check
No new upgrade operations detected.
```

M5 新增 `documents`、`ingestion_jobs`、`ingestion_job_items`，并为知识条目/chunk 增加 document、job、可见性、revision 与状态关联。迁移 head、ORM metadata 和数据库约束一致。

## 4. 测试证据

| 分层 | 结果 |
|---|---:|
| knowledge + ingestion unit | 274 passed |
| API 与认证时钟回归 | 49 passed，1 个既有 warning |
| M5 完整工作流 | 2 passed，1 个既有 warning |
| runtime 最终全量 | 572 passed，1 个既有 warning |

唯一允许的 warning 是 Starlette 对当前 `httpx` TestClient 适配层的弃用提示；M5 未新增 warning。

关键行为证据：

- 管理员 login -> CSRF PDF upload -> 后台摄取 -> job completed -> document/knowledge ready 全链路通过。
- 普通用户只能读取 `public+ready`，private 详情表现为 404，知识写接口返回 403。
- PATCH 和归档必须携带当前 revision；过期 revision 返回 `KNOWLEDGE_REVISION_CONFLICT`/409。
- 第一次 Embedding 超时后 job/document/item/chunk 均为 failed，错误摘要不包含供应商正文。
- 管理员 retry 后复用同一 job、document、item 和 chunk ID；条目与 chunk 数量不增加，LLM 抽取只执行一次，attempt_count 为 2。
- 归档知识不再被 pgvector 检索命中，历史 RAG 引用快照不受影响。
- Session 撤销时间使用数据库 created_at 下界，避免应用时钟轻微落后触发约束冲突。

## 5. 安全与事务边界

- 上传只接受 `.pdf` + `application/pdf` + `%PDF-`，并限制字节数、页数、加密状态、路径和有效文本。
- 文件先写受控根目录内的 `.part`，校验通过后原子发布；数据库创建失败只清理本次进程拥有的文件。
- API DTO 不返回 `storage_path`、绝对路径或重试载荷。
- 管理员写接口统一要求有效 Session、admin 角色、Origin 和 double-submit CSRF。
- LLM、Embedding、PDF/网页读取期间不持有业务 AsyncSession；每次状态写入使用短事务。
- job 通过状态 + attempt_count CAS 防止并发 claim、旧 worker 或重复 retry 覆盖新结果。
- 错误码和摘要使用稳定映射，不保存上游响应、密钥、数据库连接串或本机路径。

## 6. API 与 CLI 契约

知识管理：

```text
GET/POST /api/v1/knowledge-items
GET/PATCH/DELETE /api/v1/knowledge-items/{item_id}
```

文档与任务：

```text
POST /api/v1/documents
GET  /api/v1/documents
GET  /api/v1/ingestion-jobs/{job_id}
POST /api/v1/ingestion-jobs/{job_id}/cancel
POST /api/v1/ingestion-jobs/{job_id}/retry
```

网页和 PDF CLI 都要求 `--requested-by <active-admin-username>`，同步等待任务完成并通过统一 service 写 PostgreSQL/pgvector。旧 `--output`、`--error-output`、`--text-output`、`--append-text-output` 和 `--import-to-knowledge` 已移除。

## 7. Compose 验收与最终状态

Compose 使用进程级临时 `SESSION_SECRET`/`ALLOWED_ORIGINS`，没有修改或读取 `.env`。验证结果：

```text
mathrag-postgres-1  healthy
mathrag-mathrag-1   healthy
GET /health/live    200 {"status":"ok","app_name":"MathRAG MVP"}
GET /health/ready   200 config=ok,database=ok,pgvector=0.8.5
```

在 `upload_data` 创建唯一空验证目录，强制重建应用容器后目录仍存在；验证后已删除该目录。真实 PostgreSQL + fake provider 的上传 completed 和失败 retry 工作流由集成验收覆盖，Compose smoke 不调用真实外部 LLM/Embedding 服务。

验收结束后执行 `docker compose down`，不执行 `down -v`；最终 `docker compose ps --all` 必须为空。

## 8. 备份、回滚与已知限制

发布备份必须同时包含 PostgreSQL 和 `upload_data`，并标记为同一恢复点。回滚流程：

1. 停止文档上传、知识写入和两个导入 CLI，等待运行中任务收口。
2. 备份 PostgreSQL、`upload_data` 和当前应用镜像。
3. 优先只回滚应用镜像；恢复后校验 live/ready、登录、知识读取和历史 RAG。
4. 只有确认放弃全部 M5 document/job/revision 数据后，才在备份副本验证 downgrade 到 `0004_create_identity_conversation_rag_tables`。
5. 数据恢复必须使用同一时间点的数据库与上传卷，随后执行 `alembic upgrade head`。

M5 仍使用进程内 FastAPI BackgroundTasks，部署约束为单 worker；不包含 Celery/Redis、任务租约恢复、OCR、恶意文件扫描、前端管理界面或生产级容量压测。M6 应直接消费 M5 的 Cookie/CSRF、knowledge revision 和 ingestion job 轮询契约。
