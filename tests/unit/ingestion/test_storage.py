from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.modules.ingestion.errors import (
    DocumentMimeError,
    DocumentPathError,
    DocumentPdfEmptyError,
    DocumentPdfEncryptedError,
    DocumentPdfInvalidError,
    DocumentPdfPageCountError,
    DocumentStorageError,
    DocumentTooLargeError,
)
from app.modules.ingestion.storage import UploadStorage, resolve_stored_path, validate_original_name


class FakeUpload:
    def __init__(
        self,
        filename: str | None,
        content_type: str | None,
        content: bytes,
        *,
        read_sizes: list[int] | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._content = content
        self._offset = 0
        self.read_sizes = [] if read_sizes is None else read_sizes

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._offset >= len(self._content):
            return b""
        end = len(self._content) if size < 0 else self._offset + size
        chunk = self._content[self._offset : end]
        self._offset += len(chunk)
        return chunk


def _storage(
    root: Path,
    *,
    max_bytes: int = 4 * 1024 * 1024,
    max_pages: int = 10,
) -> UploadStorage:
    return UploadStorage(
        root=root,
        max_bytes=max_bytes,
        max_pages=max_pages,
        now=lambda: datetime(2026, 7, 31, tzinfo=timezone.utc),
        uuid_factory=lambda: UUID("12345678-1234-5678-1234-567812345678"),
    )


def _pdf_bytes(
    *,
    pages: int = 1,
    text: str | None = "Algebra lesson",
    password: str | None = None,
) -> bytes:
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


def _assert_no_upload_artifacts(root: Path) -> None:
    assert not list(root.rglob("*.part"))
    assert not list(root.rglob("*.pdf"))


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "   ",
        "..",
        "../escape.pdf",
        "folder/escape.pdf",
        r"folder\escape.pdf",
        "/absolute.pdf",
        r"C:\absolute.pdf",
    ],
)
def test_validate_original_name_rejects_paths_and_non_pdf(filename: str) -> None:
    with pytest.raises(DocumentPathError) as caught:
        validate_original_name(filename)

    assert caught.value.code == "DOCUMENT_PATH_INVALID"
    if filename:
        assert filename not in caught.value.message


def test_validate_original_name_accepts_pdf_case_insensitively() -> None:
    assert validate_original_name("  Lesson.PDF  ") == "Lesson.PDF"


@pytest.mark.parametrize("filename", ["notes.txt", ".pdf"])
def test_validate_original_name_maps_extension_errors_to_unsupported_mime(filename: str) -> None:
    with pytest.raises(DocumentMimeError) as caught:
        validate_original_name(filename)

    assert caught.value.code == "DOCUMENT_MIME_UNSUPPORTED"
    assert caught.value.status_code == 415


@pytest.mark.parametrize("relative_path", ["../escape.pdf", "/escape.pdf", r"..\escape.pdf", ""])
def test_resolve_stored_path_rejects_outside_root(tmp_path: Path, relative_path: str) -> None:
    with pytest.raises(DocumentPathError):
        resolve_stored_path(tmp_path / "uploads", relative_path)


def test_resolve_stored_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "uploads"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("当前平台不允许创建符号链接")

    with pytest.raises(DocumentPathError):
        resolve_stored_path(root, "linked/escape.pdf")


def test_save_upload_streams_to_uuid_path_and_returns_scalar_metadata(tmp_path: Path) -> None:
    content = _pdf_bytes(text="Algebra lesson")
    upload = FakeUpload("Lesson.PDF", "application/pdf", content)

    stored = asyncio.run(_storage(tmp_path).save_upload(upload))

    assert stored.relative_path == "2026/07/12345678-1234-5678-1234-567812345678.pdf"
    assert stored.original_name == "Lesson.PDF"
    assert stored.mime_type == "application/pdf"
    assert stored.size_bytes == len(content)
    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert resolve_stored_path(tmp_path, stored.relative_path).read_bytes() == content
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024]
    assert not list(tmp_path.rglob("*.part"))
    assert "filename" not in stored.__dict__


@pytest.mark.parametrize(
    ("content", "error_type", "code"),
    [
        (b"%PDF-corrupt", DocumentPdfInvalidError, "DOCUMENT_PDF_INVALID"),
        (
            _pdf_bytes(password="secret"),
            DocumentPdfEncryptedError,
            "DOCUMENT_PDF_ENCRYPTED",
        ),
        (
            _pdf_bytes(pages=0),
            DocumentPdfPageCountError,
            "DOCUMENT_PDF_PAGE_COUNT_INVALID",
        ),
        (
            _pdf_bytes(text=None),
            DocumentPdfEmptyError,
            "DOCUMENT_PDF_EMPTY",
        ),
    ],
)
def test_save_upload_validates_complete_pdf_before_atomic_replace(
    tmp_path: Path,
    content: bytes,
    error_type: type[Exception],
    code: str,
) -> None:
    with pytest.raises(error_type) as caught:
        asyncio.run(
            _storage(tmp_path, max_pages=10).save_upload(
                FakeUpload("invalid.pdf", "application/pdf", content)
            )
        )

    assert caught.value.code == code
    _assert_no_upload_artifacts(tmp_path)


def test_save_upload_rejects_pdf_over_page_limit_before_atomic_replace(tmp_path: Path) -> None:
    with pytest.raises(DocumentPdfPageCountError) as caught:
        asyncio.run(
            _storage(tmp_path, max_pages=1).save_upload(
                FakeUpload(
                    "many.pdf",
                    "application/pdf",
                    _pdf_bytes(pages=2, text=None),
                )
            )
        )

    assert caught.value.code == "DOCUMENT_PDF_PAGE_COUNT_INVALID"
    _assert_no_upload_artifacts(tmp_path)


@pytest.mark.parametrize(
    ("filename", "content_type", "content", "error_type", "code", "status_code"),
    [
        ("notes.pdf", "text/plain", b"%PDF-safe", DocumentMimeError, "DOCUMENT_MIME_UNSUPPORTED", 415),
        ("notes.txt", "application/pdf", b"%PDF-safe", DocumentMimeError, "DOCUMENT_MIME_UNSUPPORTED", 415),
        ("notes.pdf", "application/pdf", b"not-a-pdf", DocumentPdfInvalidError, "DOCUMENT_PDF_INVALID", 422),
    ],
)
def test_save_upload_rejects_invalid_metadata_or_magic_without_artifacts(
    tmp_path: Path,
    filename: str,
    content_type: str,
    content: bytes,
    error_type: type[Exception],
    code: str,
    status_code: int,
) -> None:
    upload = FakeUpload(filename, content_type, content)

    with pytest.raises(error_type) as caught:
        asyncio.run(_storage(tmp_path).save_upload(upload))

    assert caught.value.code == code
    assert caught.value.status_code == status_code
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_upload_stops_at_limit_and_removes_partial_file(tmp_path: Path) -> None:
    upload = FakeUpload("large.pdf", "application/pdf", b"%PDF-" + b"x" * 32)

    with pytest.raises(DocumentTooLargeError) as caught:
        asyncio.run(_storage(tmp_path, max_bytes=12).save_upload(upload))

    assert caught.value.code == "DOCUMENT_TOO_LARGE"
    assert caught.value.status_code == 413
    assert upload._offset == 13
    assert upload.read_sizes == [13]
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_upload_cancellation_removes_partial_file(tmp_path: Path) -> None:
    class CancellingUpload(FakeUpload):
        async def read(self, size: int = -1) -> bytes:
            if self._offset:
                raise asyncio.CancelledError
            self._offset = 1
            return b"%PDF-"

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            _storage(tmp_path).save_upload(
                CancellingUpload("cancelled.pdf", "application/pdf", b"")
            )
        )

    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_upload_maps_read_failure_without_leaking_details(tmp_path: Path) -> None:
    class BrokenUpload(FakeUpload):
        async def read(self, size: int = -1) -> bytes:
            if self._offset:
                raise RuntimeError("C:/private/parser-state")
            self._offset = 1
            return b"%PDF-"

    with pytest.raises(DocumentStorageError) as caught:
        asyncio.run(
            _storage(tmp_path).save_upload(
                BrokenUpload("broken.pdf", "application/pdf", b"")
            )
        )

    assert caught.value.code == "DOCUMENT_STORAGE_FAILED"
    assert "private" not in caught.value.message
    assert not list(tmp_path.rglob("*.part"))
    assert not list(tmp_path.rglob("*.pdf"))


def test_save_upload_maps_non_bytes_chunk_and_cleans_partial_file(tmp_path: Path) -> None:
    class InvalidChunkUpload(FakeUpload):
        async def read(self, size: int = -1) -> bytes:
            return "not-bytes"  # type: ignore[return-value]

    with pytest.raises(DocumentStorageError) as caught:
        asyncio.run(
            _storage(tmp_path).save_upload(
                InvalidChunkUpload("invalid.pdf", "application/pdf", b"")
            )
        )

    assert caught.value.code == "DOCUMENT_STORAGE_FAILED"
    _assert_no_upload_artifacts(tmp_path)


def test_save_upload_does_not_overwrite_or_delete_existing_uuid_target(tmp_path: Path) -> None:
    existing = tmp_path / "2026" / "07" / "12345678-1234-5678-1234-567812345678.pdf"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")

    with pytest.raises(DocumentStorageError):
        asyncio.run(
            _storage(tmp_path).save_upload(
                FakeUpload("new.pdf", "application/pdf", b"%PDF-new")
            )
        )

    assert existing.read_bytes() == b"existing"
    assert not list(tmp_path.rglob("*.part"))


def test_save_upload_atomically_rejects_final_created_during_publish(
    monkeypatch, tmp_path: Path
) -> None:
    original_link = os.link
    competing_content = b"created-by-another-call"

    def competing_link(source, target):
        Path(target).write_bytes(competing_content)
        return original_link(source, target)

    monkeypatch.setattr(os, "link", competing_link)

    with pytest.raises(DocumentStorageError) as caught:
        asyncio.run(
            _storage(tmp_path).save_upload(
                FakeUpload("race.pdf", "application/pdf", _pdf_bytes())
            )
        )

    final_path = tmp_path / "2026" / "07" / "12345678-1234-5678-1234-567812345678.pdf"
    assert caught.value.code == "DOCUMENT_STORAGE_FAILED"
    assert final_path.read_bytes() == competing_content
    assert not list(tmp_path.rglob("*.part"))


def test_save_upload_rolls_back_own_final_when_part_unlink_fails(
    monkeypatch, tmp_path: Path
) -> None:
    original_unlink = Path.unlink
    part_failed = False

    def fail_first_part_unlink(path: Path, *args, **kwargs):
        nonlocal part_failed
        if path.suffix == ".part" and not part_failed:
            part_failed = True
            raise OSError("private filesystem detail")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_first_part_unlink)

    with pytest.raises(DocumentStorageError) as caught:
        asyncio.run(
            _storage(tmp_path).save_upload(
                FakeUpload("unlink.pdf", "application/pdf", _pdf_bytes())
            )
        )

    assert caught.value.code == "DOCUMENT_STORAGE_FAILED"
    _assert_no_upload_artifacts(tmp_path)


def test_save_upload_maps_atomic_link_failure_and_cleans_part(
    monkeypatch, tmp_path: Path
) -> None:
    def fail_link(source, target):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(DocumentStorageError) as caught:
        asyncio.run(
            _storage(tmp_path).save_upload(
                FakeUpload("link.pdf", "application/pdf", _pdf_bytes())
            )
        )

    assert caught.value.code == "DOCUMENT_STORAGE_FAILED"
    _assert_no_upload_artifacts(tmp_path)


def test_save_upload_does_not_delete_stale_part_owned_by_another_call(tmp_path: Path) -> None:
    stale_part = tmp_path / "2026" / "07" / "12345678-1234-5678-1234-567812345678.part"
    stale_part.parent.mkdir(parents=True)
    stale_part.write_bytes(b"another-call")

    with pytest.raises(DocumentStorageError):
        asyncio.run(
            _storage(tmp_path).save_upload(
                FakeUpload("new.pdf", "application/pdf", _pdf_bytes())
            )
        )

    assert stale_part.read_bytes() == b"another-call"
    assert not list(tmp_path.rglob("*.pdf"))
