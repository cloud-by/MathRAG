from __future__ import annotations

import json

from app.schemas.knowledge import SEED_FIELD_ORDER, KnowledgeRecord
from app.services.math_knowledge_importer import SOURCE_REGISTRY, TextChunk, SourceDocument, discover_documents, transform_chunk


def build_record(item_id: str = "k0001") -> KnowledgeRecord:
    return KnowledgeRecord(
        id=item_id,
        category="微积分",
        stage="undergraduate",
        course="微积分",
        title="导数",
        keywords=["导数", "变化率"],
        content="导数刻画函数在某一点附近的瞬时变化率。",
        example="例如 f(x)=x^2 的导数是 f'(x)=2x。",
        steps=["先明确函数表达式", "再使用导数定义或求导法则"],
        prerequisites=["函数", "极限"],
        difficulty="medium",
    )


def test_seed_record_dump_uses_exact_seed_fields() -> None:
    record = build_record()

    data = record.to_seed_dict()

    assert list(data.keys()) == SEED_FIELD_ORDER
    assert "records" not in data
    assert "saved_count" not in data
    assert "next_steps" not in data


def test_transform_chunk_writes_valid_records_only(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "seed.jsonl"
    error_path = tmp_path / "errors.jsonl"
    chunk = TextChunk(
        document=SourceDocument(
            source_name="wikipedia",
            source_url="https://example.test/derivative",
            title="Derivative",
            license="CC BY-SA",
            text="A derivative is an instantaneous rate of change.",
        ),
        chunk_index=0,
        text="A derivative is an instantaneous rate of change.",
    )

    class FakeResponse:
        data = {
            "items": [
                {
                    "category": "微积分",
                    "stage": "undergraduate",
                    "course": "微积分",
                    "title": "导数",
                    "keywords": ["导数", "变化率"],
                    "content": "导数刻画函数在某一点附近的瞬时变化率。",
                    "example": "例如 f(x)=x^2 的导数是 f'(x)=2x。",
                    "steps": ["先明确函数表达式", "再使用导数定义或求导法则"],
                    "prerequisites": ["函数", "极限"],
                    "difficulty": "medium",
                }
            ]
        }

    monkeypatch.setattr("app.services.math_knowledge_importer.chat_json", lambda **kwargs: FakeResponse())

    records = transform_chunk(chunk=chunk, output_path=output_path, error_path=error_path)

    assert len(records) == 1
    row = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert list(row.keys()) == SEED_FIELD_ORDER
    assert row["id"] == "k0001"
    assert row["title"] == "导数"
    assert not error_path.exists()


def test_transform_chunk_writes_invalid_outputs_to_error_file(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "seed.jsonl"
    error_path = tmp_path / "errors.jsonl"
    chunk = TextChunk(
        document=SourceDocument(
            source_name="proofwiki",
            source_url="https://example.test/theorem",
            title="Theorem",
            license="see source",
            text="A theorem statement.",
        ),
        chunk_index=0,
        text="A theorem statement.",
    )

    class FakeResponse:
        data = {"items": [{"title": "Missing fields"}]}

    monkeypatch.setattr("app.services.math_knowledge_importer.chat_json", lambda **kwargs: FakeResponse())

    records = transform_chunk(chunk=chunk, output_path=output_path, error_path=error_path)

    assert records == []
    assert not output_path.exists()
    error_row = json.loads(error_path.read_text(encoding="utf-8").strip())
    assert "timestamp" in error_row
    assert error_row["error_type"] == "chunk_transform"
    assert error_row["source_name"] == "proofwiki"
    assert "raw_response" in error_row


def test_discover_documents_skips_blocked_source_and_logs_error(monkeypatch, tmp_path) -> None:
    error_path = tmp_path / "errors.jsonl"

    class BlockedSource:
        def search(self, keyword: str, limit: int):
            raise RuntimeError("403 Client Error: Forbidden")

    monkeypatch.setitem(SOURCE_REGISTRY, "blocked_test", BlockedSource())

    documents = discover_documents(
        sources=["blocked_test"],
        keywords=["derivative"],
        limit_per_source=2,
        delay_seconds=0,
        error_path=error_path,
    )

    assert documents == []
    row = json.loads(error_path.read_text(encoding="utf-8").strip())
    assert "timestamp" in row
    assert row["error_type"] == "source_discovery"
    assert row["source_name"] == "blocked_test"
    assert "403" in row["error"]


def test_transform_chunk_rejects_english_records(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "seed.jsonl"
    error_path = tmp_path / "errors.jsonl"
    chunk = TextChunk(
        document=SourceDocument(
            source_name="wikipedia",
            source_url="https://example.test/derivative",
            title="Derivative",
            license="CC BY-SA",
            text="A derivative is an instantaneous rate of change.",
        ),
        chunk_index=0,
        text="A derivative is an instantaneous rate of change.",
    )

    class FakeResponse:
        data = {
            "items": [
                {
                    "category": "calculus",
                    "stage": "undergraduate",
                    "course": "Calculus",
                    "title": "Derivative",
                    "keywords": ["derivative", "rate of change"],
                    "content": "The derivative measures the instantaneous rate of change of a function.",
                    "example": "For f(x)=x^2, f'(x)=2x.",
                    "steps": ["Identify the function", "Apply a derivative rule"],
                    "prerequisites": ["function", "limit"],
                    "difficulty": "medium",
                }
            ]
        }

    monkeypatch.setattr("app.services.math_knowledge_importer.chat_json", lambda **kwargs: FakeResponse())

    records = transform_chunk(chunk=chunk, output_path=output_path, error_path=error_path)

    assert records == []
    assert not output_path.exists()
    error_row = json.loads(error_path.read_text(encoding="utf-8").strip())
    assert "timestamp" in error_row
    assert error_row["error_type"] == "chunk_transform"
    assert "not Chinese enough" in error_row["error"]
