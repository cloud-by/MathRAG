from __future__ import annotations

import json

from app.services.math_knowledge_importer import SourceDocument
from app.services.pdf_knowledge_importer import (
    build_pdf_text_chunks,
    import_pdf_knowledge,
    write_pdf_text_chunks,
)


def test_build_pdf_text_chunks_and_write_text_set(tmp_path) -> None:
    document = SourceDocument(
        source_name="local_pdf",
        source_url="D:/tmp/math.pdf",
        title="测试教材",
        license="local file",
        text="函数的概念\n\n函数描述两个变量之间的对应关系。" * 20,
    )
    chunks = build_pdf_text_chunks([document], max_chunk_chars=80)
    output_path = tmp_path / "pdf_text_chunks.jsonl"

    count = write_pdf_text_chunks(output_path, chunks)

    assert count == len(chunks)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["source_name"] == "local_pdf"
    assert rows[0]["source_title"] == "测试教材"
    assert "函数" in rows[0]["text"]
    assert rows[0]["text_length"] == len(rows[0]["text"])


def test_import_pdf_knowledge_extract_only(monkeypatch, tmp_path) -> None:
    document = SourceDocument(
        source_name="local_pdf",
        source_url="D:/tmp/math.pdf",
        title="测试教材",
        license="local file",
        text="导数刻画函数的瞬时变化率。",
    )
    text_output = tmp_path / "chunks.jsonl"
    seed_output = tmp_path / "seed.jsonl"
    error_output = tmp_path / "errors.jsonl"

    monkeypatch.setattr("app.services.pdf_knowledge_importer.load_pdf_documents", lambda **kwargs: [document])

    result = import_pdf_knowledge(
        text_output_path=text_output,
        output_path=seed_output,
        error_path=error_output,
        extract_only=True,
    )

    assert result.documents == 1
    assert result.text_chunks == 1
    assert result.saved_records == 0
    assert text_output.exists()
    assert not seed_output.exists()


def test_import_pdf_knowledge_can_call_transform(monkeypatch, tmp_path) -> None:
    document = SourceDocument(
        source_name="local_pdf",
        source_url="D:/tmp/math.pdf",
        title="测试教材",
        license="local file",
        text="导数刻画函数的瞬时变化率。",
    )
    calls = []

    monkeypatch.setattr("app.services.pdf_knowledge_importer.load_pdf_documents", lambda **kwargs: [document])

    def fake_transform_chunk(**kwargs):
        calls.append(kwargs)
        return [object()]

    monkeypatch.setattr("app.services.pdf_knowledge_importer.transform_chunk", fake_transform_chunk)

    result = import_pdf_knowledge(
        text_output_path=tmp_path / "chunks.jsonl",
        output_path=tmp_path / "seed.jsonl",
        error_path=tmp_path / "errors.jsonl",
        extract_only=False,
        stage="senior_secondary",
        course="高中数学",
        category="导数",
    )

    assert result.saved_records == 1
    assert len(calls) == 1
    assert calls[0]["stage"] == "senior_secondary"
    assert calls[0]["course"] == "高中数学"
    assert calls[0]["category"] == "导数"
