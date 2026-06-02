from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.config import settings
from app.services.math_knowledge_importer import (
    SourceDocument,
    TextChunk,
    chunk_text,
    clean_plain_text,
    transform_chunk,
    write_jsonl,
)


DEFAULT_DATA_LAKE_DIR = settings.DATA_DIR / "data_lake"
DEFAULT_TEXT_OUTPUT = settings.PROCESSED_DATA_DIR / "pdf_text_chunks.jsonl"
DEFAULT_ERROR_OUTPUT = settings.RAW_DATA_DIR / "pdf_knowledge_import_errors.jsonl"


@dataclass(frozen=True)
class PDFExtractResult:
    documents: int
    text_chunks: int
    saved_records: int
    text_output: Path
    error_output: Path


def require_pypdf() -> Any:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction requires pypdf. Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return PdfReader


def iter_pdf_paths(data_dir: Path, recursive: bool = True) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"PDF data lake directory not found: {data_dir}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _metadata_value(metadata: Any, key: str) -> str:
    if not metadata:
        return ""
    try:
        value = metadata.get(key)
    except AttributeError:
        value = getattr(metadata, key.strip("/"), "")
    return str(value or "").strip()


def extract_pdf_document(path: Path) -> SourceDocument:
    PdfReader = require_pypdf()
    reader = PdfReader(str(path))
    metadata = getattr(reader, "metadata", None)
    title = _metadata_value(metadata, "/Title") or path.stem

    page_texts: List[str] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = f"\n[page {page_index} extraction failed: {exc}]\n"
        page_text = clean_plain_text(page_text)
        if page_text:
            page_texts.append(f"第 {page_index} 页\n{page_text}")

    text = clean_plain_text("\n\n".join(page_texts))
    return SourceDocument(
        source_name="local_pdf",
        source_url=str(path.resolve()),
        title=title,
        license="local file; check the original PDF license before redistribution",
        text=text,
    )


def load_pdf_documents(data_dir: Path = DEFAULT_DATA_LAKE_DIR, recursive: bool = True) -> List[SourceDocument]:
    documents: List[SourceDocument] = []
    for path in iter_pdf_paths(data_dir, recursive=recursive):
        document = extract_pdf_document(path)
        if document.text:
            documents.append(document)
    return documents


def build_pdf_text_chunks(
    documents: Iterable[SourceDocument],
    max_chunk_chars: int = 4000,
    max_chunks: int | None = None,
) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    for document in documents:
        for index, text in enumerate(chunk_text(document.text, max_chars=max_chunk_chars)):
            chunks.append(TextChunk(document=document, chunk_index=index, text=text))
            if max_chunks is not None and len(chunks) >= max_chunks:
                return chunks
    return chunks


def text_chunk_to_row(chunk: TextChunk) -> Dict[str, Any]:
    return {
        "source_name": chunk.document.source_name,
        "source_url": chunk.document.source_url,
        "source_title": chunk.document.title,
        "license": chunk.document.license,
        "chapter": chunk.document.chapter,
        "section": chunk.document.section,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "text_length": len(chunk.text),
    }


def write_pdf_text_chunks(path: Path, chunks: Iterable[TextChunk], append: bool = False) -> int:
    rows = [text_chunk_to_row(chunk) for chunk in chunks]
    if append:
        return write_jsonl(path, rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def import_pdf_knowledge(
    data_dir: Path = DEFAULT_DATA_LAKE_DIR,
    text_output_path: Path = DEFAULT_TEXT_OUTPUT,
    output_path: Path = settings.RAW_KB_PATH,
    error_path: Path = DEFAULT_ERROR_OUTPUT,
    recursive: bool = True,
    max_chunk_chars: int = 4000,
    max_chunks: int | None = None,
    extract_only: bool = True,
    append_text_output: bool = False,
    stage: str | None = None,
    course: str | None = None,
    category: str | None = None,
) -> PDFExtractResult:
    documents = load_pdf_documents(data_dir=data_dir, recursive=recursive)
    chunks = build_pdf_text_chunks(documents, max_chunk_chars=max_chunk_chars, max_chunks=max_chunks)
    write_pdf_text_chunks(text_output_path, chunks, append=append_text_output)

    saved_records = 0
    if not extract_only:
        for chunk in chunks:
            saved_records += len(
                transform_chunk(
                    chunk=chunk,
                    output_path=output_path,
                    error_path=error_path,
                    stage=stage,
                    course=course,
                    category=category,
                )
            )

    return PDFExtractResult(
        documents=len(documents),
        text_chunks=len(chunks),
        saved_records=saved_records,
        text_output=text_output_path,
        error_output=error_path,
    )
