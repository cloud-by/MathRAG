# M2 旧知识迁移验收基线

- 验收时间：2026-07-30 18:27:30 +08:00
- 分支：`codex/m2-legacy-knowledge-migration`
- 实现提交：`15485ab7ba27603006cd4611106b768e47efb8f6`（`fix: harden legacy import boundaries`）
- 运行环境：Python 3.11.9、PostgreSQL 18.4、pgvector 0.8.5。

## 可复制验收命令

以下命令均从仓库根目录执行，并且本基线唯一的 Python 解释器是 `.\.venv\Scripts\python.exe`（Python 3.11.9）；裸系统 `python` 不能作为本基线的解释器。连接串由安全配置或当前 PowerShell 进程注入，本文档不记录 URL 或口令。主库占位符为 `<redacted-main-url>`，其数据库名必须是 `mathrag`；测试库占位符为 `<redacted-test-url>`，其数据库名必须是 `mathrag_test`，且两者必须不同。

```powershell
# 由安全配置提供实际值；不要把实际 URL 或口令写入日志、文档或提交。
$env:DATABASE_URL = '<redacted-main-url>'
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic check
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge
.\.venv\Scripts\python.exe -m scripts.import_legacy_knowledge

$env:TEST_DATABASE_URL = '<redacted-test-url>'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\test_retrieval_baseline.py -q
.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_import_rollback.py -q
```

## 主库验收

导入前以只读 SQL 核对主库：`knowledge_items=0`、`knowledge_chunks=0`。因此未清空、覆盖或重置任何主库数据；M2 导入后的 26 条记录保留为完成态。

迁移命令 `.\.venv\Scripts\python.exe -m alembic upgrade head` 成功；关键输出为 `Context impl PostgresqlImpl.` 与 `Will assume transactional DDL.`。随后：

```text
$ .\.venv\Scripts\python.exe -m alembic current
0002_create_knowledge_tables (head)

$ .\.venv\Scripts\python.exe -m alembic check
No new upgrade operations detected.
```

离线 CLI 连续运行两次均为 exit 0，stderr 均为空；stdout 各为唯一的一行 JSON：

```json
{"conflicts":0,"created":26,"database_chunks":26,"database_items":26,"database_sha256":"82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5","failed":0,"input_chunks":26,"input_items":26,"input_sha256":"82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5","skipped":0}
{"conflicts":0,"created":0,"database_chunks":26,"database_items":26,"database_sha256":"82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5","failed":0,"input_chunks":26,"input_items":26,"input_sha256":"82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5","skipped":26}
```

主库只读 SQL 复核结果如下，所有数值均满足预期。

| 核对项 | 结果 |
| --- | ---: |
| items / chunks | 26 / 26 |
| 非空且去重的 `legacy_id` | 26 / 26 |
| `status=indexing` items | 26 |
| `visibility=public` 且 `revision=1` items | 26 |
| `status=pending` chunks | 26 |
| `embedding IS NULL` / `embedding_model IS NULL` chunks | 26 / 26 |
| 每 item 恰一 chunk / `chunk_index=0` | 26 / 26 |
| 含 `legacy_chunk_id`、`legacy_source_id`、`source_line` 的 metadata | 26 |

chunk metadata 的键集合为 `chunk_index,difficulty,has_example,has_steps,legacy_chunk_id,legacy_source_id,source_file,source_line`。用 loader 加载两份源 JSONL 后，通过 `KnowledgeRepository.list_legacy_items_ordered()` 和 `bundle_from_model()` 按 `legacy_id` 逐条比较 26 个 `persistent_payload()`，结果完全相等；源集合与数据库集合 digest 均为 `82a76468c817454de1b87c825488db6b31e6778f9d058f9a8345d7c67590d4c5`。

输入文件 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `data/raw/math_knowledge_seed.jsonl` | `2593f45081b11ab4ae280d1a7fb107791b3099c364f3813f215a73fa7369d062` |
| `data/processed/kb_chunks.jsonl` | `a0334a626d7e54ce04a447861af1616da26ad8b012d81f6720aa1d404539e5aa` |

## 回归证据

- 显式配置 `TEST_DATABASE_URL` 指向专用测试库后运行全量 `.\.venv\Scripts\python.exe -m pytest -q`：`134 passed, 1 warning in 13.64s`，无 skipped。唯一警告是既有 `StarletteDeprecationWarning`（`starlette.testclient` 与 httpx）；测试完成后的测试库计数为 `knowledge_items=0`、`knowledge_chunks=0`。
- `.\.venv\Scripts\python.exe -m pytest tests\test_retrieval_baseline.py -q`：`16 passed in 0.15s`。该数字是测试用例数，不是固定题数量。本次现场读取同一 fixture 的 26 题，并调用当前 `retrieve(question, top_k=3)`，得到 `expected_hit_count=26`、命中率 `1.0`，即 26/26；本次没有写出新的 M2 FAISS artifact。
- 可审计的 FAISS 回归证据使用既有 `docs/baselines/artifacts/faiss-top3-m1-regression.json`：artifact 自身 SHA-256 为 `b7077178a1a8b23127a5ca2b392bb913c9249f9b7b844d8acda3a0e98fb7692d`，其中记录的 26/26 结果及以下三项 hash 均与本次当前文件一致：`data/index/faiss.index`=`e2520504ff2b392bbb56aea792046a752a217a4abf75ca8dec516fd219149192`，`data/index/id_map.json`=`6fe97be89ad8398d4ed636545b4d7939b5832f93c5602a2d65b4a40781cf8331`，`data/processed/kb_chunks.jsonl`=`a0334a626d7e54ce04a447861af1616da26ad8b012d81f6720aa1d404539e5aa`。该 artifact 仅含模型和 provider-origin 的 SHA-256 等安全元数据，不含 secret。现场重跑完整 artifact 需要 embedding 配置及网络，不能把该重跑误称为本次新产物。
- `.\.venv\Scripts\python.exe -m pytest tests\integration\knowledge\test_import_rollback.py -q`：`1 passed in 0.62s`。
- TDD 覆盖重点：schema migration 升降级往返、Repository 查询与载荷恢复、真实 PostgreSQL 中已 flush 首条后遇冲突的整批 rollback、UTF-8 loader/跨文件边界、CLI 双跑幂等和字段级无损。
- 受保护 diff 以 `1882088d82eda83c4d77c2205292205314361c27`（`main` 基线）和实现提交 `15485ab7ba27603006cd4611106b768e47efb8f6` 为准，执行 `git diff --exit-code 1882088d82eda83c4d77c2205292205314361c27 15485ab7ba27603006cd4611106b768e47efb8f6 -- app/api/chat.py app/services/retriever.py data/raw/math_knowledge_seed.jsonl data/processed/kb_chunks.jsonl` 为空；两份 JSONL 的 `Get-FileHash` 与上表一致。

## 运行边界与回滚

M2 是离线单写者迁移：在线 `/api/chat` 仍只走旧 FAISS，未双写；embedding 和 `ready` 状态由 M3 承担。本阶段不实际对主库执行回滚。

实际 `POST /api/knowledge/extract` 在 `save=true` 时只调用 `append_records()` 追加 legacy raw JSONL；它不双写 PostgreSQL，也不自动重建 processed JSONL 或 FAISS。因此迁移验收窗口必须冻结该保存写入。验收后若新增知识，必须按现有流程先重新构建 processed，再导入 PostgreSQL、重建 FAISS，并重新核对输入 SHA、三项 FAISS hash 和集合 digest，避免 raw、PostgreSQL、FAISS 三套状态漂移。

回滚命令只适用于当前 migration 图且 `alembic current` 恰为 `0002_create_knowledge_tables`。执行前必须停写，以安全客户端核对 `current_database()` 为目标库、核对 `alembic current=0002_create_knowledge_tables`，并完成且验证可恢复备份；未来 head 大于 `0002` 时禁止照抄此命令。

```text
.\.venv\Scripts\python.exe -m alembic -c alembic.ini downgrade 0001_enable_vector_extension
```

该操作会删除两张 M2 知识表，不会删除 vector 扩展、JSONL 或 FAISS 文件。降级后须确认 `alembic current=0001_enable_vector_extension`、两张表不存在且 vector 扩展仍在；恢复时使用 `upgrade head`，再从备份恢复或重跑导入，并重新执行本验收。

已知非阻塞项：M2 保持离线单写者；测试库须串行专用；仍有既有 Starlette deprecation 警告。
