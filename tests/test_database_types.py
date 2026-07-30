from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.infrastructure.database.types import UTCDateTime


def test_utc_datetime_rejects_naive_values() -> None:
    column_type = UTCDateTime()

    with pytest.raises(ValueError, match="时区"):
        column_type.process_bind_param(datetime(2026, 7, 29, 12, 0), None)


def test_utc_datetime_normalizes_offset_values() -> None:
    column_type = UTCDateTime()
    value = datetime(2026, 7, 29, 16, 0, tzinfo=timezone(timedelta(hours=8)))

    assert column_type.process_bind_param(value, None) == datetime(
        2026, 7, 29, 8, 0, tzinfo=UTC
    )
