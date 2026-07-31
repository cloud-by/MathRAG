"""可持久化 RAG 执行结果测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from app.modules.knowledge.search import KnowledgeSearchHit
from app.modules.rag.execution import RAGExecution
from app.services import rag_pipeline as rag_pipeline_module
from app.services.rag_pipeline import RAGPipeline


def make_hit() -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        database_chunk_id=UUID("00000000-0000-0000-0000-000000000123"),
        legacy_chunk_id="k0001-chunk-0",
        legacy_source_id="k0001",
        category="algebra",
        title="一元二次方程",
        keywords=("方程",),
        content="知识正文",
        example="示例",
        steps=("步骤一",),
        difficulty="easy",
        answer_context="回答上下文",
        retrieval_text="检索文本",
        source_line=1,
        metadata={"stable": 1},
        distance=0.2,
    )


def test_execute_preserves_database_uuid_and_public_legacy_reference(monkeypatch) -> None:
    planner_histories: list[list[dict[str, str]]] = []

    class Planner:
        def create_plan(self, *, question, history):
            planner_histories.append(list(history or []))
            return SimpleNamespace(
                strategy="多查询",
                retrieval_queries=[" 方程 定义 ", "方程 定义"],
            )

    class Search:
        embedding_model = "embedding-test"

        async def search(self, queries, *, top_k):
            assert queries == ["当前问题", "方程 定义"]
            assert top_k == 2
            return [make_hit()]

    raw_response = SimpleNamespace(
        model="llm-test",
        choices=[SimpleNamespace(finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    def fake_chat_json(*, messages):
        assert messages
        return SimpleNamespace(
            data={
                "answer": "回答",
                "steps": ["第一步"],
                "used_knowledge": ["一元二次方程"],
                "related_questions": ["追问"],
            },
            reasoning_content="内部推理",
            raw_response=raw_response,
        )

    monkeypatch.setattr(rag_pipeline_module, "chat_json", fake_chat_json)
    pipeline = RAGPipeline(knowledge_search=Search(), planner=Planner())  # type: ignore[arg-type]

    execution = asyncio.run(
        pipeline.execute(
            question="当前问题",
            history=[
                {"role": "user", "content": "历史问题"},
                {"role": "assistant", "content": "历史回答"},
                {"role": "user", "content": "当前问题"},
            ],
            top_k=2,
        )
    )

    assert isinstance(execution, RAGExecution)
    assert execution.hits[0].database_chunk_id == UUID(
        "00000000-0000-0000-0000-000000000123"
    )
    assert execution.retrieval_queries == ("当前问题", "方程 定义")
    assert planner_histories == [
        [
            {"role": "user", "content": "历史问题"},
            {"role": "assistant", "content": "历史回答"},
        ]
    ]
    assert execution.llm_model == "llm-test"
    assert execution.embedding_model == "embedding-test"
    assert execution.model_metadata == {
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    public = execution.to_public_response()
    assert public["references"][0]["chunk_id"] == "k0001-chunk-0"
    assert "database_chunk_id" not in str(public)
    assert public["agentic_plan"] == {
        "strategy": "多查询",
        "retrieval_queries": [" 方程 定义 ", "方程 定义"],
    }
    snapshot = execution.to_reference_snapshots()[0]
    assert snapshot.chunk_id == UUID("00000000-0000-0000-0000-000000000123")
    assert snapshot.snapshot["source_id"] == "k0001"
    assert snapshot.snapshot["content"] == "知识正文"
    with pytest.raises(Exception):
        execution.answer = "篡改"  # type: ignore[misc]
