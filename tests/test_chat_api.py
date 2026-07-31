from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIError, APIStatusError
from sqlalchemy.exc import SQLAlchemyError

from app.main import app
from app.core.config import settings as app_settings
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
    assert response.headers["Deprecation"] == "true"
    assert response.headers["Link"] == '</api/v1/chat>; rel="successor-version"'


def test_legacy_chat_is_gone_outside_development(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def mock_chat_with_rag(**kwargs):
        nonlocal called
        called = True
        return build_mock_result()

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)
    monkeypatch.setattr(
        "app.api.chat.settings",
        replace(
            app_settings,
            APP_ENV="production",
            SESSION_SECRET="s" * 32,
            ALLOWED_ORIGINS=("https://mathrag.example",),
        ),
    )

    response = client.post(
        "/api/chat",
        json={"question": "什么是代数式？", "history": [], "top_k": 3},
    )

    assert response.status_code == 410
    assert response.json() == {"detail": "旧聊天接口已停用，请使用 /api/v1/chat。"}
    assert called is False


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


def test_lifespan_rebuilds_rag_dependencies_and_preserves_app_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main
    from app.services import rag_pipeline as rag_pipeline_module

    events: list[str] = []
    built_searches: list[object] = []
    provider_calls = 0
    database_calls = 0

    def build_search() -> object:
        search = object()
        built_searches.append(search)
        events.append("build")
        return search

    def reset_pipeline() -> None:
        events.append("reset")
        rag_pipeline_module._rag_pipeline = None

    async def dispose_provider() -> None:
        nonlocal provider_calls
        provider_calls += 1
        events.append(f"embedding-{provider_calls}")
        if provider_calls == 2:
            raise RuntimeError("embedding cleanup failed")

    async def dispose_database() -> None:
        nonlocal database_calls
        database_calls += 1
        events.append(f"database-{database_calls}")
        if database_calls == 2:
            raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(rag_pipeline_module, "_rag_pipeline", None)
    monkeypatch.setattr(
        rag_pipeline_module,
        "build_knowledge_search_service",
        build_search,
    )
    monkeypatch.setattr(
        rag_pipeline_module,
        "get_query_planner",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(validate_runtime=lambda: events.append("validate")),
    )
    monkeypatch.setattr(main, "reset_rag_pipeline", reset_pipeline)
    monkeypatch.setattr(main, "dispose_embedding_provider", dispose_provider)
    monkeypatch.setattr(main, "dispose_engine", dispose_database)

    async def exercise() -> tuple[object, object]:
        async with main.lifespan(main.app):
            first = rag_pipeline_module.get_rag_pipeline()
            events.append("body-1")

        with pytest.raises(RuntimeError, match="application failed"):
            async with main.lifespan(main.app):
                second = rag_pipeline_module.get_rag_pipeline()
                events.append("body-2")
                raise RuntimeError("application failed")

        return first, second

    first, second = asyncio.run(exercise())

    assert first is not second
    assert len(built_searches) == 2
    assert events == [
        "validate",
        "build",
        "body-1",
        "reset",
        "embedding-1",
        "database-1",
        "validate",
        "build",
        "body-2",
        "reset",
        "embedding-2",
        "database-2",
    ]


def test_chat_returns_400_when_pipeline_raises_rag_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services import rag_pipeline as rag_pipeline_module

    error_type = getattr(rag_pipeline_module, "RAGInputError", None)
    assert isinstance(error_type, type)

    async def mock_chat_with_rag(question: str, history: List[Dict[str, str]] | None = None, top_k: int | None = None) -> Dict[str, Any]:
        raise error_type("question 不能为空")

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


def test_chat_generic_value_error_is_fixed_500_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "llm-parse-sensitive-marker"

    async def mock_chat_with_rag(
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        raise ValueError(marker)

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "系统内部错误。"
    assert marker not in response.text


def test_chat_non_dict_pipeline_result_is_fixed_500(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def mock_chat_with_rag(
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Any:
        return ["not", "an", "object"]

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "系统内部错误。"


def test_chat_response_validation_error_is_fixed_500_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "response-validation-sensitive-marker"

    async def mock_chat_with_rag(
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        result = build_mock_result()
        result["references"][0]["score"] = marker
        return result

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "系统内部错误。"
    assert marker not in response.text


def test_chat_api_status_error_is_fixed_502_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "upstream-status-sensitive-marker"
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")
    response = httpx.Response(
        500,
        request=request,
        json={"error": {"message": marker}},
    )

    async def mock_chat_with_rag(
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        raise APIStatusError(marker, response=response, body=response.json())

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    api_response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert api_response.status_code == 502
    assert api_response.json()["detail"] == "大模型 API 返回错误。"
    assert marker not in api_response.text


def test_chat_api_error_is_fixed_502_without_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "upstream-api-sensitive-marker"
    request = httpx.Request("POST", "https://llm.invalid/v1/chat/completions")

    async def mock_chat_with_rag(
        question: str,
        history: List[Dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        raise APIError(marker, request, body=None)

    monkeypatch.setattr("app.api.chat.chat_with_rag", mock_chat_with_rag)

    response = client.post(
        "/api/chat",
        json={"question": "测试问题", "history": [], "top_k": 3},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "大模型 API 调用失败。"
    assert marker not in response.text


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
