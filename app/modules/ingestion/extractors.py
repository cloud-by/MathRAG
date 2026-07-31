"""受控 PDF 的纯解析与文本抽取。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from typing import Any, BinaryIO, Callable

from pypdf import PdfReader

from app.modules.ingestion.errors import (
    DocumentPdfEmptyError,
    DocumentPdfEncryptedError,
    DocumentPdfInvalidError,
    DocumentPdfPageCountError,
)


PDFSource = str | PathLike[str] | BinaryIO


@dataclass(frozen=True)
class ExtractedPDF:
    """PDF 抽取结果，仅包含后续导入需要的纯数据。"""

    text: str
    page_count: int
    title: str | None


def clean_extracted_text(text: str) -> str:
    """规范 PDF 提取文本中的空白，不改变可见内容顺序。"""
    normalized = text.replace("\u3000", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_pdf_text(
    source: PDFSource,
    *,
    max_pages: int,
    reader_factory: Callable[[PDFSource], Any] | None = None,
) -> ExtractedPDF:
    """解析 PDF 并返回清洗后的分页文本，所有解析错误均转换为稳定异常。"""
    if max_pages <= 0:
        raise ValueError("max_pages 必须大于 0")
    factory = PdfReader if reader_factory is None else reader_factory
    try:
        reader = factory(source)
        encrypted = bool(reader.is_encrypted)
    except Exception:
        raise DocumentPdfInvalidError() from None
    if encrypted:
        raise DocumentPdfEncryptedError()

    try:
        page_count = len(reader.pages)
    except Exception:
        raise DocumentPdfInvalidError() from None
    if page_count < 1 or page_count > max_pages:
        raise DocumentPdfPageCountError()

    page_texts: list[str] = []
    for page_index in range(page_count):
        try:
            page = reader.pages[page_index]
            text = clean_extracted_text(page.extract_text() or "")
        except Exception:
            raise DocumentPdfInvalidError() from None
        if text:
            page_texts.append(f"第 {page_index + 1} 页\n{text}")

    combined = clean_extracted_text("\n\n".join(page_texts))
    if not combined:
        raise DocumentPdfEmptyError()
    return ExtractedPDF(
        text=combined,
        page_count=page_count,
        title=_read_title(reader),
    )


def _read_title(reader: Any) -> str | None:
    """尽力读取标题元数据；异常或空值都不影响正文处理。"""
    try:
        metadata = reader.metadata
        value = metadata.get("/Title") if metadata else None
    except Exception:
        return None
    title = clean_extracted_text(str(value or ""))
    return title or None
