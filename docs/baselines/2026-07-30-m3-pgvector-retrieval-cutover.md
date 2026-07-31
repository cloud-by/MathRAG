# MathRAG M3 pgvector 检索切换验收基线

## 1. 验收结论

M3 已完成在线检索从 FAISS 到 PostgreSQL + pgvector 的单向切换。固定 26 题全部命中，FAISS/pgvector Top-3 平均重合率为 1.0，pgvector 精确 SQL P95 为 10.848 ms；生产镜像不安装 FAISS，在线 import graph 不加载 FAISS、id_map 或 processed JSONL。

验收时间：2026-07-31（Asia/Shanghai）。

## 2. 版本与提交

| 项目 | 值 |
|---|---|
| M3 基点 / M2 回滚提交 | `cd77635ffc83a5c98c2887c40701da7281e2169b` |
| 验收代码提交 | `74f0a4521f667dbb7dabf4f4319c51da6cfe0eee` |
| 验收代码 tree | `5d30b27c6addc6963dc9bce39c77bf33b9565ba5` |
| artifact 提交 | `7f74af023fe3787d6af3ab001dc7a456efeeddd4` |
| Python | `3.11.9` |
| PostgreSQL | `18.4` |
| pgvector Server / Python | `0.8.5` / `0.5.0` |
| FastAPI / SQLAlchemy | `0.140.13` / `2.0.51` |
| runtime 镜像 | `mathrag:local` |
| runtime 镜像 ID | `sha256:1b68ec57ae2399b4041727415cf4f6fb2dc0d8edea6a80665ffdda6dbb8e8a58` |

runtime lock 固定 48 个包；evaluation lock 固定 49 个包，唯一额外包为 `faiss-cpu==1.14.3`，两份锁的其余包版本完全一致。

## 3. 数据库与可重入流程

现场命令中的连接信息均已脱敏：

```powershell
$env:DATABASE_URL='postgresql+asyncpg://mathrag:***@127.0.0.1:5432/mathrag'
$env:TEST_DATABASE_URL='postgresql+asyncpg://mathrag:***@127.0.0.1:5432/mathrag_test'
python -m alembic upgrade head
python -m alembic current
python -m alembic check
python -m scripts.import_legacy_knowledge
python -m scripts.import_legacy_knowledge
python -m scripts.reindex_knowledge
python -m scripts.reindex_knowledge
```

结果：

- Alembic current：`0003_enforce_vector_readiness (head)`。
- Alembic check：`No new upgrade operations detected.`。
- M3 首次物化：import 创建 26 items/26 chunks；reindex 选择并就绪 26 chunks。
- 最终双跑：两次 import 均为 `created=0, skipped=26, conflicts=0, failed=0`。
- import 输入与数据库规范化 SHA-256 均为 `82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5`。
- 最终两次 reindex 均为 `selected=0, skipped=26, ready=0, failed=0`，证明同模型 ready 数据可重入。

最终数据库摘要：

| 指标 | 结果 |
|---|---:|
| knowledge_items / public / ready | 26 / 26 / 26 |
| knowledge_chunks / ready | 26 / 26 |
| 非空向量 | 26 |
| Embedding 模型集合 | `text-embedding-v4`（26） |
| 向量维度 min / max | 1024 / 1024 |
| 测试库 items / chunks | 0 / 0 |

## 4. 检索质量证据

真实 Provider 与主库上的执行命令：

```powershell
python -m scripts.evaluate_pgvector_retrieval `
  --fixture tests/fixtures/retrieval_questions.json `
  --output docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json `
  --replace-existing
```

Artifact：`docs/baselines/artifacts/pgvector-faiss-m3-2026-07-30.json`

| 指标 | 门槛 | 实测 |
|---|---:|---:|
| 固定问题数 | 26 | 26 |
| pgvector Top-3 期望命中 | >= 24/26 | 26/26 |
| 期望命中率 | >= 0.90 | 1.0 |
| FAISS/pgvector Top-3 平均重合率 | >= 0.80 | 1.0 |
| pgvector SQL P50 | 记录值 | 8.153 ms |
| pgvector SQL P95 | <= 100 ms | 10.848 ms |

计时范围仅包含 Repository SQL；方法为 `top_k=3`、首题向量预热 1 次、正式计时 26 次。

证据哈希：

- artifact SHA-256：`6d97b8d00f0aaee195152e09ea27475dc23569fa779368ba3930beb35403ffa0`
- fixture SHA-256：`7225124ac0b7af83754401e8d77b9cccc5fb7da67e0cf0f853beaadfb72f49f5`
- seed SHA-256：`b87355849f828ae219ba4e03315436d65a1fce749db96740ae645a74c231e4b0`
- FAISS SHA-256：`e2520504ff2b392bbb56aea792046a752a217a4abf75ca8dec516fd219149192`
- id_map SHA-256：`6fe97be89ad8398d4ed636545b4d7939b5832f93c5602a2d65b4a40781cf8331`
- Provider origin SHA-256：`ec1f29107de2c3a3df64b95fe366d7c640c5a610fdc5b6804611630835b4dafc`

Artifact 不包含 Provider URL、API key、Authorization、完整向量或知识正文。

## 5. 测试与运行时边界

- 带独立 `TEST_DATABASE_URL` 的全量测试：`417 passed, 0 skipped, 1 warning`。
- Artifact/evaluation/runtime boundary 定向回归：`134 passed`。
- 运行时边界测试：`14 passed`。
- 唯一 warning 为既有 Starlette TestClient/httpx 弃用提示。
- runtime 镜像内探针：`faiss_present=False`。
- 在线 fresh import 不加载 `faiss`、旧 retriever、旧 vector store 或旧 embedding service。
- 已删除在线 FAISS 三个 service、`scripts/build_index.py` 和可覆盖冻结 id_map 的旧修复脚本。
- 冻结 FAISS/id_map 仅允许显式 evaluation 工具只读访问。

Docker 现场结果：

```text
mathrag-mathrag-1   Up (healthy)   127.0.0.1:8000->8000/tcp
mathrag-postgres-1  Up (healthy)   127.0.0.1:5432->5432/tcp
GET /health/live    200 {"status":"ok"}
GET /health/ready   200 config=ok,database=ok,pgvector=0.8.5
```

## 6. 回滚

1. 先在入口网关停止新流量，并暂停知识写入。
2. 执行 `docker compose stop mathrag`，保持数据库运行并保留备份。
3. 优先重新部署上一固定镜像；无可用镜像时，从 M2 提交 `cd77635ffc83a5c98c2887c40701da7281e2169b` 构建。
4. 冻结 `data/index/faiss.index` 与 `data/index/id_map.json`，只读用于旧版本紧急回滚或审计，不得重建或覆盖。
5. 分别验证 `/health/live` 与 `/health/ready`，成功后再恢复流量和写入。
6. 禁止在同一在线版本长期并行 FAISS 与 pgvector 两条检索路径。
7. 如需回滚数据库 schema，先停止写入并从备份恢复，或在确认兼容性后单独执行 Alembic downgrade；不得把数据库回滚与在线双路径混用。

README 的“回滚”章节提供了镜像标签、M2 构建和健康检查命令模板。

## 7. 最终边界

M3 不包含用户/owner 权限模型、HNSW/IVFFlat、Redis、Celery 或多 Worker 协调。private 数据继续由 SQL 过滤为不可在线检索；近似索引只有在后续数据量与 P95 证明确有需要时再单独设计和迁移。
