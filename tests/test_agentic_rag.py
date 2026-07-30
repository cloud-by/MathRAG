from __future__ import annotations

import asyncio
import inspect
import threading
from types import SimpleNamespace
from typing import Any, Dict, List
from uuid import UUID

import pytest

from app.services.agentic_rag import QueryPlanner
from app.modules.knowledge.search import KnowledgeSearchHit
from app.services import rag_pipeline as rag_pipeline_module
from app.services.rag_pipeline import RAGPipeline, chat_with_rag
from scripts import test_rag as test_rag_cli


def _make_hit(
    database_chunk_id: UUID,
    *,
    legacy_chunk_id: str,
    title: str,
    distance: float,
) -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        database_chunk_id=database_chunk_id,
        legacy_chunk_id=legacy_chunk_id,
        legacy_source_id=legacy_chunk_id,
        category="concept",
        title=title,
        keywords=("测试",),
        content="内容",
        example="示例",
        steps=("步骤1", "步骤2"),
        difficulty="easy",
        answer_context="上下文",
        retrieval_text="检索文本",
        source_line=1,
        metadata={},
        distance=distance,
    )


def test_query_planner_fallback_to_original_question(monkeypatch) -> None:
    def _raise_chat_json(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("app.services.agentic_rag.chat_json", _raise_chat_json)

    planner = QueryPlanner()
    plan = planner.create_plan(question="什么是二次函数？", history=[])

    assert plan.strategy.startswith("规划失败")
    assert plan.retrieval_queries == ["什么是二次函数？"]


def test_rag_pipeline_and_chat_wrapper_are_async() -> None:
    assert inspect.iscoroutinefunction(RAGPipeline.chat)
    assert inspect.iscoroutinefunction(chat_with_rag)


def test_reset_rag_pipeline_discards_cached_pipeline(monkeypatch) -> None:
    reset = getattr(rag_pipeline_module, "reset_rag_pipeline", None)
    assert callable(reset)

    monkeypatch.setattr(rag_pipeline_module, "_rag_pipeline", SimpleNamespace())
    reset()

    assert rag_pipeline_module._rag_pipeline is None


def test_rag_pipeline_uses_dedicated_error_for_user_input() -> None:
    error_type = getattr(rag_pipeline_module, "RAGInputError", None)
    assert isinstance(error_type, type)
    assert issubclass(error_type, ValueError)

    pipeline = RAGPipeline(knowledge_search=SimpleNamespace())

    with pytest.raises(error_type, match="question 不能为空"):
        asyncio.run(pipeline.chat(question="   "))

    with pytest.raises(error_type, match="top_k 必须是 1 到 10 的整数"):
        asyncio.run(pipeline.chat(question="测试问题", top_k=0))


def test_test_rag_cli_awaits_pipeline_for_default_output(
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[str, list[Dict[str, Any]], int]] = []

    async def fake_chat_with_rag(
        question: str,
        history: list[Dict[str, Any]],
        top_k: int,
    ) -> Dict[str, Any]:
        calls.append((question, history, top_k))
        return {
            "question": question,
            "answer": "测试回答",
            "steps": [],
            "used_knowledge": [],
            "related_questions": [],
            "references": [],
        }

    monkeypatch.setattr(
        test_rag_cli,
        "parse_args",
        lambda: SimpleNamespace(
            question="测试问题",
            top_k=2,
            show_references=False,
            show_full_json=False,
        ),
    )
    monkeypatch.setattr(test_rag_cli, "chat_with_rag", fake_chat_with_rag)

    test_rag_cli.main()

    assert calls == [("测试问题", [], 2)]
    output = capsys.readouterr().out
    assert "问题：测试问题" in output
    assert "回答：\n测试回答" in output


def test_test_rag_cli_awaits_pipeline_for_full_json(
    monkeypatch,
    capsys,
) -> None:
    async def fake_chat_with_rag(
        question: str,
        history: list[Dict[str, Any]],
        top_k: int,
    ) -> Dict[str, Any]:
        return {"question": question, "answer": "完整结果", "references": []}

    monkeypatch.setattr(
        test_rag_cli,
        "parse_args",
        lambda: SimpleNamespace(
            question="完整输出",
            top_k=3,
            show_references=False,
            show_full_json=True,
        ),
    )
    monkeypatch.setattr(test_rag_cli, "chat_with_rag", fake_chat_with_rag)

    test_rag_cli.main()

    payload = capsys.readouterr().out
    assert '"question": "完整输出"' in payload
    assert '"answer": "完整结果"' in payload


def test_rag_pipeline_uses_one_batched_knowledge_search(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    worker_threads: Dict[str, int] = {}

    class MockPlanner:
        strategy = "拆分核心概念和别名"
        retrieval_queries = ["二次函数 定义", "抛物线 图像"]

        def create_plan(self, question: str, history: List[Dict[str, Any]] | None = None) -> "MockPlanner":
            worker_threads["planner"] = threading.get_ident()
            return self

    class FakeKnowledgeSearchService:
        def __init__(self) -> None:
            self.calls: list[tuple[list[str], int]] = []

        async def search(
            self,
            queries: List[str],
            *,
            top_k: int,
        ) -> list[KnowledgeSearchHit]:
            self.calls.append((list(queries), top_k))
            return [
                _make_hit(
                    UUID("00000000-0000-0000-0000-000000000002"),
                    legacy_chunk_id="c2",
                    title="二次函数图像",
                    distance=0.12,
                ),
                _make_hit(
                    UUID("00000000-0000-0000-0000-000000000001"),
                    legacy_chunk_id="c1",
                    title="二次函数定义",
                    distance=0.40,
                ),
            ]

    search = FakeKnowledgeSearchService()

    def mock_chat_json(*, messages: List[Dict[str, str]]) -> SimpleNamespace:
        assert messages
        worker_threads["llm"] = threading.get_ident()
        return SimpleNamespace(
            data={
                "answer": "二次函数图像是抛物线。",
                "steps": ["识别形式", "观察图像"],
                "used_knowledge": ["二次函数图像"],
                "related_questions": ["顶点怎么求？", "开口方向如何判断？"],
            }
        )

    monkeypatch.setattr(rag_pipeline_module, "chat_json", mock_chat_json)

    pipeline = RAGPipeline(
        knowledge_search=search,
        planner=MockPlanner(),
    )
    result = asyncio.run(
        pipeline.chat(question="什么是二次函数", history=[], top_k=2)
    )

    assert search.calls == [
        (["什么是二次函数", "二次函数 定义", "抛物线 图像"], 2)
    ]
    assert result["agentic_plan"]["retrieval_queries"] == [
        "二次函数 定义",
        "抛物线 图像",
    ]
    assert result["references"] == [
        {
            "rank": 1,
            "score": 0.88,
            "index": None,
            "chunk_id": "c2",
            "source_id": "c2",
            "category": "concept",
            "title": "二次函数图像",
            "keywords": ["测试"],
            "content": "内容",
            "example": "示例",
            "steps": ["步骤1", "步骤2"],
            "difficulty": "easy",
            "answer_context": "上下文",
            "retrieval_text": "检索文本",
            "source_line": 1,
            "metadata": {},
        },
        {
            "rank": 2,
            "score": 0.6,
            "index": None,
            "chunk_id": "c1",
            "source_id": "c1",
            "category": "concept",
            "title": "二次函数定义",
            "keywords": ["测试"],
            "content": "内容",
            "example": "示例",
            "steps": ["步骤1", "步骤2"],
            "difficulty": "easy",
            "answer_context": "上下文",
            "retrieval_text": "检索文本",
            "source_line": 1,
            "metadata": {},
        },
    ]
    assert result["used_knowledge"] == ["二次函数图像"]
    assert worker_threads.keys() == {"planner", "llm"}
    assert worker_threads["planner"] != event_loop_thread
    assert worker_threads["llm"] != event_loop_thread
    assert "app.services.retriever" not in inspect.getsource(rag_pipeline_module)
