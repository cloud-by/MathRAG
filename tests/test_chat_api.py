from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from app.main import app
from app.modules.knowledge.errors import EmbeddingUnavailableError


client = TestClient(app)


def build_mock_reference(rank: int = 1) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": 0.987654,
        "index": None,
        "chunk_id": f"k000{rank}_chunk_0",
        "source_id": f"k000{rank}",
        "category": "concept",
        "title": f"测试知识点{rank}",
        "keywords": ["代数式", "表达式", "字母表示数"],
        "content": "这是一个用于测试的知识点内容。",
        "example": "例如 3x+2 是一个代数式。",
        "steps": ["步骤1：识别结构", "步骤2：理解含义"],
        "difficulty": "easy",
        "answer_context": "【测试知识点】\n这是一个用于回答的上下文。",
        "retrieval_text": "类别：concept\n标题：测试知识点",
        "source_line": rank,
        "metadata": {
            "source_file": "math_knowledge_seed.jsonl",
            "chunk_index": 0,
            "difficulty": "easy",
        },
    }


def build_mock_result() -> Dict[str, Any]:
    return {
        "question": "什么是代数式？",
        "answer": "代数式是由数、字母和运算符号组成的式子，用来表示数量关系。",
        "steps": [
            "先观察式子中是否包含字母和运算符号。",
            "再判断它是否没有等号，从而区分代数式与方程。"
        ],
        "used_knowledge": ["测试知识点1"],
        "related_questions": ["代数式和方程有什么区别？", "什么叫同类项？"],
        "references": [build_mock_reference(1), build_mock_reference(2)],
        "agentic_plan": {
            "strategy": "拆分核心概念并补充同义表达",
            "retrieval_queries": ["代数式 定义", "代数式 与 方程 区别"],
        },
        "reasoning_content": "这是测试用的推理内容。",
    }


def test_chat_success_returns_complete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        assert question == "什么是代数式？"
        assert isinstance(history, list)
        assert top_k == 3
        return build_mock_result()

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={
            "question": "什么是代数式？",
            "history": [
                {"role": "user", "content": "先前问题1"},
                {"role": "assistant", "content": "先前回答1"},
            ],
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "什么是代数式？"
    assert "代数式" in data["answer"]
    assert isinstance(data["steps"], list)
    assert len(data["steps"]) >= 2
    assert data["used_knowledge"] == ["测试知识点1"]
    assert len(data["related_questions"]) == 2
    assert isinstance(data["references"], list)
    assert len(data["references"]) == 2
    assert data["agentic_plan"]["strategy"] == "拆分核心概念并补充同义表达"
    assert data["agentic_plan"]["retrieval_queries"] == ["代数式 定义", "代数式 与 方程 区别"]

    ref = data["references"][0]
    assert ref["chunk_id"] == "k0001_chunk_0"
    assert ref["source_id"] == "k0001"
    assert ref["category"] == "concept"
    assert ref["title"] == "测试知识点1"
    assert ref["difficulty"] == "easy"
    assert ref["index"] is None
    assert ref["keywords"] == ["代数式", "表达式", "字母表示数"]
    assert ref["steps"] == ["步骤1：识别结构", "步骤2：理解含义"]
    assert "reasoning_content" in data


def test_chat_history_is_passed_as_plain_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        captured["question"] = question
        captured["history"] = history
        captured["top_k"] = top_k
        return build_mock_result()

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    payload = {
        "question": "平方差公式是什么？",
        "history": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，我是数学助教。"},
            {"role": "user", "content": "平方差公式是什么？"},
        ],
        "top_k": 5,
    }

    response = client.post("/api/chat", json=payload)

    assert response.status_code == 200
    assert captured["question"] == "平方差公式是什么？"
    assert captured["top_k"] == 5
    assert captured["history"] == payload["history"]


def test_chat_rejects_empty_question() -> None:
    response = client.post(
        "/api/chat",
        json={
            "question": "   ",
            "history": [],
            "top_k": 3,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_chat_rejects_invalid_top_k() -> None:
    response = client.post(
        "/api/chat",
        json={
            "question": "什么是函数？",
            "history": [],
            "top_k": 0,
        },
    )

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_openapi_describes_pgvector_and_legacy_index_contract() -> None:
    document = client.get("/openapi.json").json()

    assert document["info"]["description"] == (
        "基于 FastAPI + PostgreSQL/pgvector + 大模型 API 的数学 RAG 问答原型系统"
    )
    index_schema = document["components"]["schemas"]["ReferenceItem"]["properties"][
        "index"
    ]
    assert index_schema["description"] == "旧版兼容字段；pgvector 路径为空"


def test_lifespan_preserves_app_error_while_attempting_both_cleanups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main

    events: list[str] = []

    async def dispose_provider() -> None:
        events.append("embedding")
        raise RuntimeError("embedding cleanup failed")

    async def dispose_database() -> None:
        events.append("database")
        raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(validate_runtime=lambda: events.append("validate")),
    )
    monkeypatch.setattr(main, "dispose_embedding_provider", dispose_provider)
    monkeypatch.setattr(main, "dispose_engine", dispose_database)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="application failed"):
            async with main.lifespan(main.app):
                events.append("body")
                raise RuntimeError("application failed")

    asyncio.run(exercise())

    assert events == ["validate", "body", "embedding", "database"]


def test_chat_returns_400_when_pipeline_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise ValueError("question 不能为空")

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={
            "question": "测试问题",
            "history": [],
            "top_k": 3,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "question 不能为空"


def test_chat_returns_503_without_leaking_database_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise SQLAlchemyError("postgresql://user:password@host/database")

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "知识检索暂不可用。"
    assert "password" not in response.text


def test_chat_returns_502_without_leaking_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise EmbeddingUnavailableError("https://provider.invalid?api_key=secret")

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "向量服务暂不可用。"
    assert "api_key" not in response.text
    assert "secret" not in response.text


def test_chat_catch_all_is_fixed_and_does_not_expose_faiss_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise FileNotFoundError("data/index/faiss.index?token=secret")

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={
            "question": "测试问题",
            "history": [],
            "top_k": 3,
        },
    )

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "系统内部错误。"
    assert "faiss.index" not in response.text
    assert "secret" not in response.text


def test_chat_response_reference_schema_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        result = build_mock_result()
        result["references"] = [build_mock_reference(1)]
        return result

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={
            "question": "导数的几何意义是什么？",
            "history": [],
            "top_k": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()

    ref = data["references"][0]
    required_fields = {
        "rank",
        "score",
        "index",
        "chunk_id",
        "source_id",
        "category",
        "title",
        "keywords",
        "content",
        "example",
        "steps",
        "difficulty",
        "answer_context",
        "retrieval_text",
        "source_line",
        "metadata",
    }

    assert required_fields.issubset(set(ref.keys()))
