"""以 UTF-8 安全读取既有知识 JSONL 的迁移加载器。"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.modules.knowledge.errors import DuplicateLegacyIdError, LegacyKnowledgeInputError
from app.modules.knowledge.schemas import (
    LegacyKnowledgeBundle,
    LegacyKnowledgeChunkInput,
    LegacyKnowledgeItemInput,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
_STEP_PREFIX = re.compile(r"^步骤\d+[：:]\s*")


def _validation_detail(error: ValidationError) -> str:
    """将 Pydantic 错误压缩为不含输入值的稳定诊断。"""
    parts: list[str] = []
    for detail in error.errors(include_input=False):
        location = ".".join(str(part) for part in detail["loc"]) or "record"
        parts.append(f"{location}: {detail['type']}: {detail['msg']}")
    return "; ".join(parts)


def _decode_error_line(error: UnicodeError) -> int:
    """尽力从 UTF-8 解码异常中恢复一基行号。"""
    raw = getattr(error, "object", b"")
    start = getattr(error, "start", 0)
    if isinstance(raw, bytes):
        return raw[:start].count(b"\n") + 1
    return 1


def _read_jsonl(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    """以 UTF-8 读取 JSONL，忽略空行并安全封装所有输入错误。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise LegacyKnowledgeInputError(f"{path}:{_decode_error_line(exc)}: UTF-8 解码失败") from exc
    except OSError as exc:
        raise LegacyKnowledgeInputError(f"{path}:1: 无法读取输入文件") from exc

    records: list[ModelT] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(model_type.model_validate_json(line))
        except ValidationError as exc:
            raise LegacyKnowledgeInputError(
                f"{path}:{line_number}: {_validation_detail(exc)}"
            ) from exc
    return records


def _duplicate_ids(values: list[str]) -> list[str]:
    """返回排序后的重复标识，避免依赖输入顺序。"""
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _normalize_processed_steps(
    raw_steps: list[str], processed_steps: list[str], legacy_id: str
) -> list[str]:
    """仅兼容历史构建器为每个步骤添加的一次显示序号前缀。"""
    if raw_steps == processed_steps:
        return processed_steps
    if len(raw_steps) != len(processed_steps):
        raise LegacyKnowledgeInputError(f"legacy_id={legacy_id}: 处理后 steps 数量与原始记录不一致")

    normalized = [_STEP_PREFIX.sub("", step, count=1) for step in processed_steps]
    if normalized != raw_steps:
        raise LegacyKnowledgeInputError(f"legacy_id={legacy_id}: 处理后 steps 与原始记录不一致")
    return raw_steps


def _bundle_or_input_error(
    item: LegacyKnowledgeItemInput,
    chunk: LegacyKnowledgeChunkInput,
    raw_path: Path,
    chunk_path: Path,
) -> LegacyKnowledgeBundle:
    """构造跨文件 bundle，并将验证失败隔离为安全输入错误。"""
    normalized_steps = _normalize_processed_steps(item.steps, chunk.steps, item.id)
    bundle_chunk = chunk.model_copy(update={"steps": normalized_steps})
    try:
        return LegacyKnowledgeBundle(item=item, chunk=bundle_chunk, chunk_index=0)
    except ValidationError as exc:
        raise LegacyKnowledgeInputError(
            f"legacy_id={item.id}; raw_path={raw_path}; chunk_path={chunk_path}; "
            f"{_validation_detail(exc)}"
        ) from exc


def load_legacy_bundles(raw_path: Path, chunk_path: Path) -> list[LegacyKnowledgeBundle]:
    """加载并校验旧知识的原始记录与处理后分块，返回按 ID 排序的 bundle。"""
    items = _read_jsonl(raw_path, LegacyKnowledgeItemInput)
    chunks = _read_jsonl(chunk_path, LegacyKnowledgeChunkInput)

    duplicate_item_ids = _duplicate_ids([item.id for item in items])
    if duplicate_item_ids:
        raise DuplicateLegacyIdError(f"发现重复的原始旧知识 ID: {', '.join(duplicate_item_ids)}")

    duplicate_source_ids = _duplicate_ids([chunk.source_id for chunk in chunks])
    if duplicate_source_ids:
        raise DuplicateLegacyIdError(f"发现重复的处理后 source_id: {', '.join(duplicate_source_ids)}")

    duplicate_chunk_ids = _duplicate_ids([chunk.chunk_id for chunk in chunks])
    if duplicate_chunk_ids:
        raise LegacyKnowledgeInputError(f"发现重复的处理后 chunk_id: {', '.join(duplicate_chunk_ids)}")

    items_by_id = {item.id: item for item in items}
    chunks_by_source_id = {chunk.source_id: chunk for chunk in chunks}
    missing = sorted(set(items_by_id) - set(chunks_by_source_id))
    orphan = sorted(set(chunks_by_source_id) - set(items_by_id))
    if missing or orphan:
        details: list[str] = []
        if missing:
            details.append(f"missing_chunks={', '.join(missing)}")
        if orphan:
            details.append(f"orphan_chunks={', '.join(orphan)}")
        raise LegacyKnowledgeInputError(
            f"旧知识 ID 集合不一致: {', '.join(details)}"
        )

    return [
        _bundle_or_input_error(items_by_id[legacy_id], chunks_by_source_id[legacy_id], raw_path, chunk_path)
        for legacy_id in sorted(items_by_id)
    ]
