"""旧知识 JSONL 加载器的单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.modules.knowledge.errors import DuplicateLegacyIdError, LegacyKnowledgeInputError
from app.modules.knowledge.schemas import collection_sha256


def _item(legacy_id: str = "k0001", **changes: object) -> dict[str, object]:
    """构造一条符合真实旧知识 schema 的原始记录。"""
    value: dict[str, object] = {
        "id": legacy_id,
        "category": "代数",
        "title": "一元一次方程",
        "keywords": ["方程"],
        "content": "基础内容",
        "example": "x + 1 = 2",
        "steps": ["移项", "求解"],
        "difficulty": "easy",
    }
    value.update(changes)
    return value


def _chunk(legacy_id: str = "k0001", **changes: object) -> dict[str, object]:
    """构造一条符合真实旧知识 schema 的处理后记录。"""
    item = _item(legacy_id)
    value: dict[str, object] = {
        "chunk_id": f"{legacy_id}_chunk_0",
        "source_id": legacy_id,
        **{key: item[key] for key in ("category", "title", "keywords", "content", "example", "steps", "difficulty")},
        "source_line": 1,
        "retrieval_text": "检索文本",
        "answer_context": "回答上下文",
        "metadata": {"source_file": "raw.jsonl"},
    }
    value.update(changes)
    return value


def _write_jsonl(path: Path, *records: object) -> None:
    """使用 UTF-8 写入测试 JSONL，并保留空行表达。"""
    lines = [record if isinstance(record, str) else json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_load_legacy_bundles_accepts_valid_pairs_and_ignores_blank_lines(tmp_path: Path) -> None:
    """有效记录按 legacy_id 稳定排序，空白行不计入输入。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, "", _item("k0002"), "   ", _item("k0001"))
    _write_jsonl(chunk_path, _chunk("k0001"), "\t", _chunk("k0002"))

    bundles = load_legacy_bundles(raw_path, chunk_path)

    assert [bundle.item.id for bundle in bundles] == ["k0001", "k0002"]
    assert [bundle.chunk_index for bundle in bundles] == [0, 0]
    assert len(collection_sha256(bundles)) == 64


def test_bad_json_reports_path_and_line_without_raw_content(tmp_path: Path) -> None:
    """坏 JSON 的诊断包含定位但绝不回显原始知识正文。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    secret = "绝不可泄露的知识正文"
    _write_jsonl(raw_path, _item(), f'{{"content":"{secret}"')
    _write_jsonl(chunk_path, _chunk())

    with pytest.raises(LegacyKnowledgeInputError) as raised:
        load_legacy_bundles(raw_path, chunk_path)

    detail = str(raised.value)
    assert "raw.jsonl:2" in detail
    assert secret not in detail


def test_schema_error_reports_field_path_without_input_value(tmp_path: Path) -> None:
    """Schema 诊断只含字段路径、类型和消息，不含输入正文。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    secret = "输入值不得出现"
    _write_jsonl(raw_path, _item(content=secret, difficulty="impossible"))
    _write_jsonl(chunk_path, _chunk())

    with pytest.raises(LegacyKnowledgeInputError) as raised:
        load_legacy_bundles(raw_path, chunk_path)

    detail = str(raised.value)
    assert "raw.jsonl:1" in detail
    assert "difficulty" in detail
    assert secret not in detail
    assert "input_value" not in detail


@pytest.mark.parametrize(
    ("raw_records", "chunk_records", "error_type", "expected"),
    [
        ((_item(), _item()), (_chunk(),), DuplicateLegacyIdError, "k0001"),
        ((_item(),), (_chunk(), _chunk("k0002", source_id="k0001")), DuplicateLegacyIdError, "k0001"),
        ((_item(), _item("k0002")), (_chunk(), _chunk("k0002", chunk_id="k0001_chunk_0")), LegacyKnowledgeInputError, "k0001_chunk_0"),
    ],
)
def test_duplicate_identifiers_are_rejected_stably(
    tmp_path: Path,
    raw_records: tuple[dict[str, object], ...],
    chunk_records: tuple[dict[str, object], ...],
    error_type: type[Exception],
    expected: str,
) -> None:
    """raw ID、chunk source_id 与 chunk_id 三类重复均必须显式拒绝。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, *raw_records)
    _write_jsonl(chunk_path, *chunk_records)

    with pytest.raises(error_type, match=expected):
        load_legacy_bundles(raw_path, chunk_path)


@pytest.mark.parametrize(
    ("raw_records", "chunk_records", "expected"),
    [((_item("k0001"),), (_chunk("k0002"),), "missing_chunks=k0001, orphan_chunks=k0002"),
        ((_item("k0001"), _item("k0002")), (_chunk("k0001"),), "missing_chunks=k0002"),
        ((_item("k0001"),), (_chunk("k0001"), _chunk("k0002")), "orphan_chunks=k0002")],
)
def test_missing_and_orphan_identifier_sets_are_reported_stably(
    tmp_path: Path,
    raw_records: tuple[dict[str, object], ...],
    chunk_records: tuple[dict[str, object], ...],
    expected: str,
) -> None:
    """两文件 ID 集合不一致时，诊断包含稳定排序的集合。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, *raw_records)
    _write_jsonl(chunk_path, *chunk_records)

    with pytest.raises(LegacyKnowledgeInputError, match=expected):
        load_legacy_bundles(raw_path, chunk_path)


def test_cross_file_mismatch_is_wrapped_without_knowledge_body(tmp_path: Path) -> None:
    """跨文件字段不一致时，Pydantic 错误必须转为安全的输入错误。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    secret = "不应出现在错误中的正文"
    _write_jsonl(raw_path, _item(content=secret))
    _write_jsonl(chunk_path, _chunk(title="不一致的标题", content=secret))

    with pytest.raises(LegacyKnowledgeInputError) as raised:
        load_legacy_bundles(raw_path, chunk_path)

    detail = str(raised.value)
    assert "k0001" in detail
    assert str(raw_path) in detail
    assert str(chunk_path) in detail
    assert secret not in detail


def test_processed_step_prefixes_are_normalized_only_in_memory(tmp_path: Path) -> None:
    """历史“步骤 N：”前缀可精确剥离，持久化真值仍来自 raw。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, _item(steps=["移项", "求解"]))
    _write_jsonl(chunk_path, _chunk(steps=["步骤1： 移项", "步骤2:求解"]))

    bundles = load_legacy_bundles(raw_path, chunk_path)

    assert bundles[0].item.steps == ["移项", "求解"]
    assert bundles[0].chunk.steps == ["移项", "求解"]


def test_equal_steps_without_prefix_remain_valid(tmp_path: Path) -> None:
    """没有历史前缀时，相同 steps 可直接配对。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, _item())
    _write_jsonl(chunk_path, _chunk())

    assert load_legacy_bundles(raw_path, chunk_path)[0].item.id == "k0001"


def test_step_prefix_normalization_rejects_remaining_difference(tmp_path: Path) -> None:
    """剥离精确历史前缀后仍不同的 steps 必须安全失败。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    _write_jsonl(raw_path, _item(steps=["移项", "求解"]))
    _write_jsonl(chunk_path, _chunk(steps=["步骤1：移项", "步骤2：错误步骤"]))

    with pytest.raises(LegacyKnowledgeInputError) as raised:
        load_legacy_bundles(raw_path, chunk_path)

    assert "legacy_id=k0001" in str(raised.value)


def test_real_legacy_files_load_26_sorted_bundles() -> None:
    """仓库内现存 UTF-8 JSONL 可无损加载为 26 条有序 bundle。"""
    from app.core.config import settings
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    bundles = load_legacy_bundles(settings.RAW_KB_PATH, settings.PROCESSED_KB_PATH)

    assert len(bundles) == 26
    assert [bundle.item.id for bundle in bundles] == sorted(bundle.item.id for bundle in bundles)
    assert len(collection_sha256(bundles)) == 64


def test_invalid_utf8_is_wrapped_with_path_and_line(tmp_path: Path) -> None:
    """UTF-8 解码错误应转换为领域输入错误且不得泄露字节正文。"""
    from app.modules.knowledge.legacy_loader import load_legacy_bundles

    raw_path = tmp_path / "raw.jsonl"
    chunk_path = tmp_path / "chunks.jsonl"
    raw_path.write_bytes(b"\xff\xfe")
    _write_jsonl(chunk_path, _chunk())

    with pytest.raises(LegacyKnowledgeInputError) as raised:
        load_legacy_bundles(raw_path, chunk_path)

    assert "raw.jsonl:1" in str(raised.value)
