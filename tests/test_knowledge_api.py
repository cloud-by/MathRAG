from __future__ import annotations

from typing import List

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.modules.auth.dependencies import require_admin_csrf
from app.schemas.knowledge import KnowledgeRecord


client = TestClient(app)


@pytest.fixture(autouse=True)
def override_admin_csrf_dependency():
    app.dependency_overrides[require_admin_csrf] = lambda: object()
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_admin_csrf, None)


def build_record(item_id: str = "k9999") -> KnowledgeRecord:
    return KnowledgeRecord(
        id=item_id,
        category="函数",
        title="一次函数的概念",
        keywords=["一次函数", "函数", "正比例函数"],
        content="一次函数通常形如 y=kx+b，其中 k 不等于 0，表示两个变量之间的线性关系。",
        example="例如 y=2x+3 是一次函数。",
        steps=["识别表达式是否为 y=kx+b 的形式", "检查 k 是否不等于 0"],
        difficulty="easy",
    )


def test_extract_knowledge_rejects_legacy_save_before_external_call(monkeypatch) -> None:
    extraction_called = False

    def mock_extract_knowledge_records(text: str, category: str | None = None) -> List[KnowledgeRecord]:
        nonlocal extraction_called
        extraction_called = True
        return [build_record()]

    monkeypatch.setattr("app.api.knowledge.extract_knowledge_records", mock_extract_knowledge_records)

    response = client.post(
        "/api/knowledge/extract",
        json={
            "text": "一次函数一般形如 y=kx+b。",
            "category": "函数",
            "save": True,
        },
    )

    assert response.status_code == 410
    assert response.json() == {"detail": "旧 JSONL 写入能力已停用。"}
    assert extraction_called is False


def test_extract_knowledge_can_preview_without_saving(monkeypatch) -> None:
    records = [build_record()]
    def mock_extract_knowledge_records(text: str, category: str | None = None) -> List[KnowledgeRecord]:
        return records

    monkeypatch.setattr("app.api.knowledge.extract_knowledge_records", mock_extract_knowledge_records)

    response = client.post(
        "/api/knowledge/extract",
        json={
            "text": "一次函数一般形如 y=kx+b。",
            "save": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["saved_count"] == 0
    assert data["records"][0]["title"] == "一次函数的概念"
    assert data["next_steps"] == []


def test_extract_knowledge_rejects_empty_text() -> None:
    response = client.post("/api/knowledge/extract", json={"text": "   "})

    assert response.status_code == 422
