from __future__ import annotations

from typing import List

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.knowledge import KnowledgeRecord


client = TestClient(app)


def build_record(item_id: str = "k9999") -> KnowledgeRecord:
    return KnowledgeRecord(
        id=item_id,
        category="函数",
        stage="junior_secondary",
        course="初中数学",
        title="一次函数的概念",
        keywords=["一次函数", "函数", "正比例函数"],
        content="一次函数通常形如 y=kx+b，其中 k 不等于 0，表示两个变量之间的线性关系。",
        example="例如 y=2x+3 是一次函数。",
        steps=["识别表达式是否为 y=kx+b 的形式", "检查 k 是否不等于 0"],
        prerequisites=["变量", "代数式"],
        difficulty="easy",
    )


def test_extract_knowledge_saves_records(monkeypatch) -> None:
    records = [build_record()]

    def mock_extract_knowledge_records(text: str, stage: str | None = None, course: str | None = None, category: str | None = None) -> List[KnowledgeRecord]:
        assert text == "一次函数一般形如 y=kx+b。"
        assert stage == "junior_secondary"
        assert course == "初中数学"
        return records

    def mock_append_records(items) -> int:
        assert list(items) == records
        return 1

    monkeypatch.setattr("app.api.knowledge.extract_knowledge_records", mock_extract_knowledge_records)
    monkeypatch.setattr("app.api.knowledge.append_records", mock_append_records)

    response = client.post(
        "/api/knowledge/extract",
        json={
            "text": "一次函数一般形如 y=kx+b。",
            "stage": "junior_secondary",
            "course": "初中数学",
            "save": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["saved_count"] == 1
    assert data["records"][0]["id"] == "k9999"
    assert data["records"][0]["title"] == "一次函数的概念"
    assert any("python -m scripts.build_kb" in step for step in data["next_steps"])


def test_extract_knowledge_can_preview_without_saving(monkeypatch) -> None:
    records = [build_record()]
    append_called = False

    def mock_extract_knowledge_records(text: str, stage: str | None = None, course: str | None = None, category: str | None = None) -> List[KnowledgeRecord]:
        return records

    def mock_append_records(items) -> int:
        nonlocal append_called
        append_called = True
        return 1

    monkeypatch.setattr("app.api.knowledge.extract_knowledge_records", mock_extract_knowledge_records)
    monkeypatch.setattr("app.api.knowledge.append_records", mock_append_records)

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
    assert append_called is False


def test_extract_knowledge_rejects_empty_text() -> None:
    response = client.post("/api/knowledge/extract", json={"text": "   "})

    assert response.status_code == 422
