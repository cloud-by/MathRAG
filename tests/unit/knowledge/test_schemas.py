"""旧知识迁移输入契约测试。"""

import pytest
from pydantic import ValidationError

from app.modules.knowledge.schemas import (
    LegacyImportSummary,
    LegacyKnowledgeBundle,
    LegacyKnowledgeChunkInput,
    LegacyKnowledgeItemInput,
    collection_sha256,
)


def item_data(**changes: object) -> dict[str, object]:
    """构造一条有效的旧知识条目。"""
    data: dict[str, object] = {
        "id": "legacy-algebra-1",
        "category": "algebra",
        "title": "一元一次方程",
        "keywords": ["方程", "代数"],
        "content": "移项并合并同类项。",
        "example": "2x + 1 = 5",
        "steps": ["移项", "求解"],
        "difficulty": "easy",
    }
    data.update(changes)
    return data


def chunk_data(**changes: object) -> dict[str, object]:
    """构造一条有效的旧知识分块。"""
    data: dict[str, object] = {
        "chunk_id": "legacy-algebra-1-0",
        "source_id": "legacy-algebra-1",
        "category": "algebra",
        "title": "一元一次方程",
        "keywords": ["方程", "代数"],
        "content": "移项并合并同类项。",
        "example": "2x + 1 = 5",
        "steps": ["移项", "求解"],
        "difficulty": "easy",
        "source_line": 1,
        "retrieval_text": "一元一次方程：移项并合并同类项。",
        "answer_context": "先移项，再求解。",
        "metadata": {"origin": "legacy"},
    }
    data.update(changes)
    return data


def bundle_data(**chunk_changes: object) -> LegacyKnowledgeBundle:
    """构造一组匹配的条目和分块。"""
    return LegacyKnowledgeBundle(
        item=LegacyKnowledgeItemInput(**item_data()),
        chunk=LegacyKnowledgeChunkInput(**chunk_data(**chunk_changes)),
    )


def test_bundle_rejects_mismatched_source_id() -> None:
    """分块来源必须对应同一条旧知识。"""
    with pytest.raises(ValidationError, match="source_id"):
        bundle_data(source_id="another-source")


def test_bundle_rejects_mismatched_title() -> None:
    """跨文件的可持久化知识字段必须完全一致。"""
    with pytest.raises(ValidationError, match="title"):
        bundle_data(title="不同标题")


def test_sha256_ignores_metadata_key_order_but_detects_content_change() -> None:
    """摘要对 JSON 键顺序稳定，并能识别内容变化。"""
    first = bundle_data(metadata={"origin": "legacy", "page": 2})
    reordered = bundle_data(metadata={"page": 2, "origin": "legacy"})
    changed = bundle_data(metadata={"origin": "legacy", "page": 3})

    assert first.sha256() == reordered.sha256()
    assert first.sha256() != changed.sha256()


def test_collection_sha256_is_independent_of_bundle_order() -> None:
    """集合摘要按旧条目 ID 排序。"""
    first = bundle_data()
    second = LegacyKnowledgeBundle(
        item=LegacyKnowledgeItemInput(**item_data(id="legacy-geometry-1", title="三角形")),
        chunk=LegacyKnowledgeChunkInput(
            **chunk_data(
                chunk_id="legacy-geometry-1-0",
                source_id="legacy-geometry-1",
                title="三角形",
            )
        ),
    )

    assert collection_sha256([first, second]) == collection_sha256([second, first])


def test_input_models_reject_extra_fields() -> None:
    """旧输入不得静默接收未知字段。"""
    with pytest.raises(ValidationError):
        LegacyKnowledgeItemInput(**item_data(unexpected="value"))

    with pytest.raises(ValidationError):
        LegacyKnowledgeChunkInput(**chunk_data(unexpected="value"))


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"created": -1}, "created"),
        ({"input_sha256": "A" * 64}, "input_sha256"),
        ({"database_sha256": "not-a-sha"}, "database_sha256"),
    ],
)
def test_summary_rejects_invalid_counts_and_digests(
    changes: dict[str, object], match: str
) -> None:
    """导入摘要仅接受非负计数和规范 SHA-256。"""
    data: dict[str, object] = {
        "input_items": 1,
        "input_chunks": 1,
        "created": 1,
        "skipped": 0,
        "conflicts": 0,
        "failed": 0,
        "database_items": 1,
        "database_chunks": 1,
        "input_sha256": "a" * 64,
        "database_sha256": "b" * 64,
    }
    data.update(changes)

    with pytest.raises(ValidationError, match=match):
        LegacyImportSummary(**data)


def test_persistent_payload_contains_only_migration_fields() -> None:
    """持久化载荷不包含运行期状态，并补充旧来源元数据。"""
    bundle = bundle_data(metadata={"origin": "legacy", "status": "source-value"})

    assert bundle.persistent_payload() == {
        "item": item_data(),
        "chunk": {
            "chunk_index": 0,
            "retrieval_text": "一元一次方程：移项并合并同类项。",
            "answer_context": "先移项，再求解。",
            "metadata": {
                "origin": "legacy",
                "status": "source-value",
                "legacy_chunk_id": "legacy-algebra-1-0",
                "legacy_source_id": "legacy-algebra-1",
                "source_line": 1,
            },
        },
    }
