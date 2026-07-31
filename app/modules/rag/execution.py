"""脱离 ORM 和 HTTP 的可持久化 RAG 执行结果。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from app.modules.knowledge.search import KnowledgeSearchHit


@dataclass(frozen=True)
class ReferenceSnapshot:
    rank: int
    chunk_id: UUID
    score: float
    snapshot: dict[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot", deepcopy(self.snapshot))


@dataclass(frozen=True)
class RAGExecution:
    question: str
    answer: str
    steps: tuple[str, ...]
    used_knowledge: tuple[str, ...]
    related_questions: tuple[str, ...]
    hits: tuple[KnowledgeSearchHit, ...]
    strategy: str
    retrieval_queries: tuple[str, ...]
    top_k: int
    llm_model: str
    embedding_model: str
    reasoning_content: str | None
    model_metadata: dict[str, object]
    agentic_plan_queries: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_metadata", deepcopy(self.model_metadata))

    def to_public_response(self) -> dict[str, object]:
        references = [
            hit.to_reference(rank=rank)
            for rank, hit in enumerate(self.hits, start=1)
        ]
        response: dict[str, object] = {
            "question": self.question,
            "answer": self.answer,
            "steps": list(self.steps),
            "used_knowledge": list(self.used_knowledge),
            "related_questions": list(self.related_questions),
            "references": references,
            "agentic_plan": {
                "strategy": self.strategy,
                "retrieval_queries": list(self.agentic_plan_queries),
            },
        }
        if self.reasoning_content is not None:
            response["reasoning_content"] = self.reasoning_content
        return response

    def to_reference_snapshots(self) -> tuple[ReferenceSnapshot, ...]:
        snapshots: list[ReferenceSnapshot] = []
        for rank, hit in enumerate(self.hits, start=1):
            snapshots.append(
                ReferenceSnapshot(
                    rank=rank,
                    chunk_id=hit.database_chunk_id,
                    score=hit.score,
                    snapshot={
                        "source_id": hit.legacy_source_id,
                        "category": hit.category,
                        "title": hit.title,
                        "keywords": list(hit.keywords),
                        "content": hit.content,
                        "example": hit.example,
                        "steps": list(hit.steps),
                        "difficulty": hit.difficulty,
                        "answer_context": hit.answer_context,
                        "retrieval_text": hit.retrieval_text,
                        "metadata": deepcopy(hit.metadata),
                    },
                )
            )
        return tuple(snapshots)


class RAGExecutor(Protocol):
    async def execute(
        self,
        *,
        question: str,
        history: Sequence[dict[str, str]],
        top_k: int | None,
    ) -> RAGExecution: ...


def safe_model_metadata(
    *,
    finish_reason: object,
    usage: object,
) -> dict[str, object]:
    """只保留稳定 finish reason 和数值 token usage。"""
    metadata: dict[str, object] = {}
    if isinstance(finish_reason, str) and finish_reason:
        metadata["finish_reason"] = finish_reason
    for field_name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _read_field(usage, field_name)
        if type(value) is int and value >= 0:
            metadata[field_name] = value
    return metadata


def _read_field(source: object, field_name: str) -> object:
    if isinstance(source, Mapping):
        return source.get(field_name)
    return getattr(source, field_name, None)
