"""旧知识迁移输入与稳定摘要契约。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_CONSISTENT_FIELDS = (
    "category",
    "title",
    "keywords",
    "content",
    "example",
    "steps",
    "difficulty",
)


class LegacyKnowledgeItemInput(BaseModel):
    """原始 JSONL 中的一条旧知识条目。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    keywords: list[str]
    content: str = Field(min_length=1)
    example: str
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]


class LegacyKnowledgeChunkInput(BaseModel):
    """处理后 JSONL 中的一条旧知识分块。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=255)
    keywords: list[str]
    content: str = Field(min_length=1)
    example: str
    steps: list[str]
    difficulty: Literal["easy", "medium", "hard"]
    source_line: int = Field(ge=1)
    retrieval_text: str = Field(min_length=1)
    answer_context: str = Field(min_length=1)
    metadata: dict[str, object]


class LegacyKnowledgeBundle(BaseModel):
    """一条原始旧知识及其对应处理分块。"""

    model_config = ConfigDict(extra="forbid")

    item: LegacyKnowledgeItemInput
    chunk: LegacyKnowledgeChunkInput
    chunk_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_consistency(self) -> LegacyKnowledgeBundle:
        """确保跨文件记录描述的是同一份知识。"""
        if self.item.id != self.chunk.source_id:
            raise ValueError("source_id 必须与 item.id 一致")

        for field_name in _CONSISTENT_FIELDS:
            if getattr(self.item, field_name) != getattr(self.chunk, field_name):
                raise ValueError(f"{field_name} 必须在 item 与 chunk 间完全一致")
        return self

    def persistent_payload(self) -> dict[str, object]:
        """返回用于持久化和幂等比较的稳定载荷。"""
        metadata = {
            **self.chunk.metadata,
            "legacy_chunk_id": self.chunk.chunk_id,
            "legacy_source_id": self.chunk.source_id,
            "source_line": self.chunk.source_line,
        }
        return {
            "item": self.item.model_dump(mode="json"),
            "chunk": {
                "chunk_index": self.chunk_index,
                "retrieval_text": self.chunk.retrieval_text,
                "answer_context": self.chunk.answer_context,
                "metadata": metadata,
            },
        }

    def sha256(self) -> str:
        """计算此迁移载荷的规范 SHA-256 摘要。"""
        return _sha256_json(self.persistent_payload())


def collection_sha256(bundles: Sequence[LegacyKnowledgeBundle]) -> str:
    """按旧条目 ID 排序后计算迁移集合的稳定摘要。"""
    payloads = [bundle.persistent_payload() for bundle in sorted(bundles, key=lambda bundle: bundle.item.id)]
    return _sha256_json(payloads)


class LegacyImportSummary(BaseModel):
    """旧知识迁移执行后的可审计摘要。"""

    model_config = ConfigDict(extra="forbid")

    input_items: int = Field(ge=0)
    input_chunks: int = Field(ge=0)
    created: int = Field(ge=0)
    skipped: int = Field(ge=0)
    conflicts: int = Field(ge=0)
    failed: int = Field(ge=0)
    database_items: int = Field(ge=0)
    database_chunks: int = Field(ge=0)
    input_sha256: str = Field(pattern=_DIGEST_PATTERN)
    database_sha256: str = Field(pattern=_DIGEST_PATTERN)


def _sha256_json(value: object) -> str:
    """使用固定 JSON 编码生成 SHA-256 摘要。"""
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
