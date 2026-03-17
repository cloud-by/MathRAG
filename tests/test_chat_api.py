from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def build_mock_reference(rank: int = 1) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": 0.987654,
        "index": rank - 1,
        "chunk_id": f"k000{rank}_chunk_0",
        "source_id": f"k000{rank}",
        "category": "concept",
        "stage": "junior_secondary",
        "course": "初中代数",
        "title": f"测试知识点{rank}",
        "keywords": ["代数式", "表达式", "字母表示数"],
        "content": "这是一个用于测试的知识点内容。",
        "example": "例如 3x+2 是一个代数式。",
        "steps": ["步骤1：识别结构", "步骤2：理解含义"],
        "prerequisites": ["用字母表示数"],
        "difficulty": "easy",
        "answer_context": "【测试知识点】\n这是一个用于回答的上下文。",
        "retrieval_text": "学段：初中\n课程：初中代数\n标题：测试知识点",
        "source_line": rank,
        "metadata": {
            "source_file": "math_knowledge_seed.jsonl",
            "chunk_index": 0,
            "stage": "junior_secondary",
            "course": "初中代数",
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
        "reasoning_content": "这是测试用的推理内容。",
    }


def test_chat_success_returns_complete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
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

    ref = data["references"][0]
    assert ref["chunk_id"] == "k0001_chunk_0"
    assert ref["source_id"] == "k0001"
    assert ref["category"] == "concept"
    assert ref["stage"] == "junior_secondary"
    assert ref["course"] == "初中代数"
    assert ref["title"] == "测试知识点1"
    assert ref["difficulty"] == "easy"
    assert ref["keywords"] == ["代数式", "表达式", "字母表示数"]
    assert ref["steps"] == ["步骤1：识别结构", "步骤2：理解含义"]
    assert ref["prerequisites"] == ["用字母表示数"]
    assert "reasoning_content" in data


def test_chat_history_is_passed_as_plain_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
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


def test_chat_returns_400_when_pipeline_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
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


def test_chat_returns_500_when_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise FileNotFoundError("data/index/faiss.index")

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
    assert "系统文件缺失" in data["detail"]


def test_chat_response_reference_schema_is_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
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
        "stage",
        "course",
        "title",
        "keywords",
        "content",
        "example",
        "steps",
        "prerequisites",
        "difficulty",
        "answer_context",
        "retrieval_text",
        "source_line",
        "metadata",
    }

    assert required_fields.issubset(set(ref.keys()))