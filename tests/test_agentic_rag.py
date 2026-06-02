from __future__ import annotations

from typing import Any, Dict, List

from app.services.agentic_rag import QueryPlanner
from app.services.rag_pipeline import RAGPipeline


def _mock_ref(chunk_id: str, score: float, title: str, rank: int = 1) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": score,
        "index": rank - 1,
        "chunk_id": chunk_id,
        "source_id": chunk_id,
        "category": "concept",
        "stage": "junior_secondary",
        "course": "初中代数",
        "title": title,
        "keywords": ["测试"],
        "content": "内容",
        "example": "示例",
        "steps": ["步骤1", "步骤2"],
        "prerequisites": ["前置"],
        "difficulty": "easy",
        "answer_context": "上下文",
        "retrieval_text": "检索文本",
        "source_line": 1,
        "metadata": {},
    }


def test_query_planner_fallback_to_original_question(monkeypatch) -> None:
    def _raise_chat_json(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("app.services.agentic_rag.chat_json", _raise_chat_json)

    planner = QueryPlanner()
    plan = planner.create_plan(question="什么是二次函数？", history=[])

    assert plan.strategy.startswith("规划失败")
    assert plan.retrieval_queries == ["什么是二次函数？"]


def test_rag_pipeline_uses_multi_query_retrieval(monkeypatch) -> None:
    class MockPlanner:
        strategy = "拆分核心概念和别名"
        retrieval_queries = ["二次函数 定义", "抛物线 图像"]

        def create_plan(self, question: str, history: List[Dict[str, Any]] | None = None) -> "MockPlanner":
            return self

    def _mock_retrieve(question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if "定义" in question:
            return [
                _mock_ref("c1", 0.60, "二次函数定义", 1),
                _mock_ref("c2", 0.55, "二次函数图像", 2),
            ]
        return [
            _mock_ref("c2", 0.88, "二次函数图像", 1),
            _mock_ref("c3", 0.51, "抛物线性质", 2),
        ]

    class MockLLMResponse:
        data = {
            "answer": "二次函数图像是抛物线。",
            "steps": ["识别形式", "观察图像"],
            "used_knowledge": ["二次函数图像"],
            "related_questions": ["顶点怎么求？", "开口方向如何判断？"],
        }

    monkeypatch.setattr("app.services.rag_pipeline.get_query_planner", lambda: MockPlanner())
    monkeypatch.setattr("app.services.rag_pipeline.retrieve", _mock_retrieve)
    monkeypatch.setattr("app.services.rag_pipeline.chat_json", lambda messages: MockLLMResponse())

    pipeline = RAGPipeline()
    result = pipeline.chat(question="什么是二次函数", history=[], top_k=2)

    assert result["agentic_plan"]["retrieval_queries"] == ["二次函数 定义", "抛物线 图像"]
    assert len(result["references"]) == 2
    assert result["references"][0]["chunk_id"] == "c2"
    assert result["references"][0]["score"] == 0.88
    assert result["used_knowledge"] == ["二次函数图像"]