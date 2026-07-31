from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT, Settings


def test_ingestion_settings_have_safe_defaults() -> None:
    configured = Settings()

    assert configured.UPLOAD_DIR == PROJECT_ROOT / "data" / "uploads"
    assert configured.MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    assert configured.MAX_PDF_PAGES == 200
    assert configured.MAX_INGESTION_TEXT_CHARS == 200_000
    assert configured.INGESTION_CHUNK_CHARS == 4_000


def test_ingestion_settings_accept_explicit_values(tmp_path: Path) -> None:
    configured = Settings(
        UPLOAD_DIR=tmp_path / "uploads",
        MAX_UPLOAD_BYTES=10,
        MAX_PDF_PAGES=2,
        MAX_INGESTION_TEXT_CHARS=100,
        INGESTION_CHUNK_CHARS=50,
    )

    assert configured.UPLOAD_DIR == tmp_path / "uploads"
    assert configured.MAX_UPLOAD_BYTES == 10
    assert configured.MAX_PDF_PAGES == 2
    assert configured.MAX_INGESTION_TEXT_CHARS == 100
    assert configured.INGESTION_CHUNK_CHARS == 50


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("MAX_UPLOAD_BYTES", 0),
        ("MAX_PDF_PAGES", 0),
        ("MAX_INGESTION_TEXT_CHARS", 0),
        ("INGESTION_CHUNK_CHARS", 0),
    ],
)
def test_ingestion_numeric_limits_must_be_positive(
    field_name: str, value: int
) -> None:
    with pytest.raises(ValueError, match=field_name):
        Settings(**{field_name: value})
