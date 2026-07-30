# M2 旧知识迁移验收基线

- 验收时间：2026-07-30 18:27:30 +08:00
- 分支：`codex/m2-legacy-knowledge-migration`
- 实现提交：`15485ab7ba27603006cd4611106b768e47efb8f6`（`fix: harden legacy import boundaries`）
- 运行环境：Python 3.11.9、PostgreSQL 18.4、pgvector 0.8.5。

## 主库验收

导入前以只读 SQL 核对主库：`knowledge_items=0`、`knowledge_chunks=0`。因此未清空、覆盖或重置任何主库数据；M2 导入后的 26 条记录保留为完成态。

迁移命令 `python -m alembic upgrade head` 成功；关键输出为 `Context impl PostgresqlImpl.` 与 `Will assume transactional DDL.`。随后：

```text
$ python -m alembic current
0002_create_knowledge_tables (head)

$ python -m alembic check
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

- 显式配置 `TEST_DATABASE_URL` 指向专用测试库后运行全量 `python -m pytest -q`：`134 passed, 1 warning in 13.64s`，无 skipped。唯一警告是既有 `StarletteDeprecationWarning`（`starlette.testclient` 与 httpx）。
- `python -m pytest tests\\test_retrieval_baseline.py -q`：`16 passed in 0.15s`。另以同一固定 fixture 调用当前 FAISS `retrieve(..., top_k=3)`：fixture 为 26 题，`expected_hit_count=26`、命中率 `1.0`，即 26/26。
- `python -m pytest tests\\integration\\knowledge\\test_import_rollback.py -q`：`1 passed in 0.62s`。
- TDD 覆盖重点：schema migration 升降级往返、Repository 查询与载荷恢复、真实 PostgreSQL 中已 flush 首条后遇冲突的整批 rollback、UTF-8 loader/跨文件边界、CLI 双跑幂等和字段级无损。
- `git diff main -- app/api/chat.py app/services/retriever.py data/raw/math_knowledge_seed.jsonl data/processed/kb_chunks.jsonl` 为空；两份 JSONL 的 `Get-FileHash` 与上表一致。

## 运行边界与回滚

M2 是离线单写者迁移：在线 `/api/chat` 仍只走旧 FAISS，未双写；embedding 和 `ready` 状态由 M3 承担。本阶段不实际对主库执行回滚。

如确需回滚，命令仅为：

```text
python -m alembic -c alembic.ini downgrade 0001_enable_vector_extension
```

该操作会删除两张 M2 知识表，必须先备份；不会删除 vector 扩展、JSONL 或 FAISS 文件。

已知非阻塞项：M2 保持离线单写者；测试库须串行专用；仍有既有 Starlette deprecation 警告。
