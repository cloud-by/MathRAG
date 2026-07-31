from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.modules.ingestion.errors import (
    DocumentPdfEmptyError,
    DocumentPdfEncryptedError,
    DocumentPdfInvalidError,
    DocumentPdfPageCountError,
)
from app.modules.ingestion.extractors import extract_pdf_text


def _pdf_bytes(*, pages: int = 1, text: str | None = "Algebra lesson", password: str | None = None) -> bytes:
    output = BytesIO()
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        if text is not None:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            font_ref = writer._add_object(font)
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): font_ref}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password is not None:
        writer.encrypt(password)
    writer.write(output)
    return output.getvalue()


def test_extract_pdf_text_parses_real_pdf_and_cleans_text(tmp_path: Path) -> None:
    path = tmp_path / "lesson.pdf"
    path.write_bytes(_pdf_bytes(text="Algebra   lesson"))

    result = extract_pdf_text(path, max_pages=10)

    assert result.page_count == 1
    assert result.text == "第 1 页\nAlgebra lesson"
    assert isinstance(result.text, str)
    with pytest.raises(AttributeError):
        result.text = "changed"  # type: ignore[misc]


def test_extract_pdf_text_rejects_corrupt_pseudo_pdf_without_leaking_details(tmp_path: Path) -> None:
    path = tmp_path / "private-location.pdf"
    path.write_bytes(b"%PDF-this-is-not-a-real-pdf")

    with pytest.raises(DocumentPdfInvalidError) as caught:
        extract_pdf_text(path, max_pages=10)

    assert caught.value.code == "DOCUMENT_PDF_INVALID"
    assert str(path) not in caught.value.message
    assert "EOF" not in caught.value.message


def test_extract_pdf_text_rejects_zero_pages(tmp_path: Path) -> None:
    path = tmp_path / "zero.pdf"
    path.write_bytes(_pdf_bytes(pages=0))

    with pytest.raises(DocumentPdfPageCountError) as caught:
        extract_pdf_text(path, max_pages=10)

    assert caught.value.code == "DOCUMENT_PDF_PAGE_COUNT_INVALID"


def test_extract_pdf_text_rejects_page_limit(tmp_path: Path) -> None:
    path = tmp_path / "many.pdf"
    path.write_bytes(_pdf_bytes(pages=2, text=None))

    with pytest.raises(DocumentPdfPageCountError) as caught:
        extract_pdf_text(path, max_pages=1)

    assert caught.value.code == "DOCUMENT_PDF_PAGE_COUNT_INVALID"


def test_extract_pdf_text_rejects_encrypted_pdf(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    path.write_bytes(_pdf_bytes(password="secret"))

    with pytest.raises(DocumentPdfEncryptedError) as caught:
        extract_pdf_text(path, max_pages=10)

    assert caught.value.code == "DOCUMENT_PDF_ENCRYPTED"


def test_extract_pdf_text_rejects_all_blank_pages(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    path.write_bytes(_pdf_bytes(text=None))

    with pytest.raises(DocumentPdfEmptyError) as caught:
        extract_pdf_text(path, max_pages=10)

    assert caught.value.code == "DOCUMENT_PDF_EMPTY"


def test_extract_pdf_text_maps_page_failure_to_stable_error() -> None:
    class BrokenPage:
        def extract_text(self) -> str:
            raise RuntimeError("sensitive parser detail")

    class FakeReader:
        is_encrypted = False
        pages = [BrokenPage()]
        metadata = None

    with pytest.raises(DocumentPdfInvalidError) as caught:
        extract_pdf_text(
            BytesIO(b"%PDF-fake"),
            max_pages=10,
            reader_factory=lambda _source: FakeReader(),
        )

    assert caught.value.code == "DOCUMENT_PDF_INVALID"
    assert "sensitive parser detail" not in caught.value.message
