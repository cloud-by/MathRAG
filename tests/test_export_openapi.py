from __future__ import annotations

from scripts.export_openapi import serialize_openapi_document


def test_openapi_serialization_is_deterministic_utf8_text() -> None:
    document = {
        "z_path": {"summary": "数学问答"},
        "a_path": {"responses": {"200": {"description": "成功"}}},
    }

    serialized = serialize_openapi_document(document)

    assert serialized.index('"a_path"') < serialized.index('"z_path"')
    assert "数学问答" in serialized
    assert "\\u" not in serialized
    assert "\r" not in serialized
    assert serialized.endswith("\n")
