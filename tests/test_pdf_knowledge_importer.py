from __future__ import annotations

import json

from app.core.config import settings
from app.services.math_knowledge_importer import SourceDocument
from app.services.pdf_knowledge_importer import (
    build_pdf_text_chunks,
    extract_pdf_document,
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
        category="导数",
    )

    assert result.saved_records == 1
    assert len(calls) == 1
    assert calls[0]["category"] == "导数"


def test_extract_pdf_document_reuses_controlled_extractor_without_absolute_source(
    monkeypatch, tmp_path
) -> None:
    pdf_path = tmp_path / "private" / "lesson.pdf"
    pdf_path.parent.mkdir()
    pdf_path.write_bytes(b"%PDF-placeholder")

    class Result:
        text = "第 1 页\n函数定义"
        page_count = 1
        title = "函数教材"

    calls = []

    def fake_extract(path, *, max_pages):
        calls.append((path, max_pages))
        return Result()

    monkeypatch.setattr("app.services.pdf_knowledge_importer.extract_pdf_text", fake_extract)

    document = extract_pdf_document(pdf_path)

    assert calls == [(pdf_path, settings.MAX_PDF_PAGES)]
    assert document.title == "函数教材"
    assert document.text == "第 1 页\n函数定义"
    assert document.source_url == "lesson.pdf"
    assert str(pdf_path.resolve()) not in document.source_url
