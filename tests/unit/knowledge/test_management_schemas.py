"""知识管理公开模型测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.modules.ingestion.schemas import (
    DocumentAccepted,
    DocumentPage,
    DocumentRead,
    IngestionJobRead,
)
from app.modules.knowledge.management_schemas import (
    KnowledgeItemCreate,
    KnowledgeItemPage,
    KnowledgeItemRead,
    KnowledgeItemUpdate,
)


def _knowledge_create_data(**changes: object) -> dict[str, object]:
    data: dict[str, object] = {
        "category": "  algebra  ",
        "title": "  Quadratic equations  ",
        "keywords": [" roots ", "", "roots", "factoring"],
        "content": "  Factor the equation.  ",
        "example": "  x² - 1 = 0  ",
        "steps": ["  Factor  ", "", "Factor", "Solve"],
        "difficulty": "medium",
        "visibility": "public",
    }
    data.update(changes)
    return data


def test_knowledge_create_normalizes_strings_and_collection_values() -> None:
    request = KnowledgeItemCreate.model_validate(_knowledge_create_data())

    assert request.category == "algebra"
    assert request.title == "Quadratic equations"
    assert request.content == "Factor the equation."
    assert request.example == "x² - 1 = 0"
    assert request.keywords == ["roots", "factoring"]
    assert request.steps == ["Factor", "Solve"]


@pytest.mark.parametrize("field_name", ["category", "title", "content"])
def test_knowledge_create_requires_nonempty_core_fields(field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        KnowledgeItemCreate.model_validate(_knowledge_create_data(**{field_name: "  "}))


@pytest.mark.parametrize("field_name", ["keywords", "steps"])
def test_knowledge_create_requires_nonempty_normalized_collections(field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        KnowledgeItemCreate.model_validate(_knowledge_create_data(**{field_name: ["", "  "]}))


def test_knowledge_update_requires_revision_and_allows_only_editable_fields() -> None:
    request = KnowledgeItemUpdate(revision=4, visibility="private", title="  Updated  ")

    assert request.revision == 4
    assert request.title == "Updated"
    with pytest.raises(ValidationError, match="revision"):
        KnowledgeItemUpdate(visibility="private")
    with pytest.raises(ValidationError):
        KnowledgeItemUpdate.model_validate({"revision": 4, "status": "ready"})


@pytest.mark.parametrize(
    "field_name",
    [
        "category",
        "title",
        "keywords",
        "content",
        "example",
        "steps",
        "difficulty",
        "visibility",
    ],
)
def test_knowledge_update_rejects_explicit_null_fields(field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        KnowledgeItemUpdate.model_validate({"revision": 4, field_name: None})


def test_knowledge_update_requires_at_least_one_editable_field() -> None:
    with pytest.raises(ValidationError, match="至少提供一个"):
        KnowledgeItemUpdate.model_validate({"revision": 4})


def test_read_models_are_frozen_and_do_not_expose_internal_fields() -> None:
    now = datetime.now(timezone.utc)
    item = KnowledgeItemRead(
        id=uuid4(),
        legacy_id=None,
        owner_id=uuid4(),
        category="algebra",
        title="Quadratic equations",
        keywords=["roots"],
        content="Factor the equation.",
        example="x² - 1 = 0",
        steps=["Factor"],
        difficulty="medium",
        visibility="public",
        status="ready",
        revision=1,
        created_at=now,
        updated_at=now,
    )

    with pytest.raises(ValidationError):
        item.title = "Other"


def test_ingestion_read_models_hide_storage_and_request_payload() -> None:
    now = datetime.now(timezone.utc)
    document = DocumentRead(
        id=uuid4(),
        owner_id=uuid4(),
        original_name="lesson.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
        sha256="a" * 64,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    job = IngestionJobRead(
        id=uuid4(),
        requested_by=uuid4(),
        document_id=document.id,
        job_type="pdf",
        status="pending",
        progress=0,
        attempt_count=0,
        error_code=None,
        error_message=None,
        started_at=None,
        finished_at=None,
        created_at=now,
        updated_at=now,
    )
    accepted = DocumentAccepted(document=document, job=job)

    assert "storage_path" not in document.model_dump()
    assert "request_payload" not in job.model_dump()
    assert accepted.document.id == document.id


def test_all_public_read_dtos_support_attributes_and_are_frozen() -> None:
    now = datetime.now(timezone.utc)
    document_id = uuid4()
    knowledge_source = SimpleNamespace(
        id=uuid4(),
        legacy_id=None,
        owner_id=uuid4(),
        category="algebra",
        title="Quadratic equations",
        keywords=["roots"],
        content="Factor the equation.",
        example="x² - 1 = 0",
        steps=["Factor"],
        difficulty="medium",
        visibility="public",
        status="ready",
        revision=1,
        created_at=now,
        updated_at=now,
    )
    document_source = SimpleNamespace(
        id=document_id,
        owner_id=uuid4(),
        original_name="lesson.pdf",
        mime_type="application/pdf",
        size_bytes=1024,
        sha256="a" * 64,
        status="pending",
        created_at=now,
        updated_at=now,
    )
    job_source = SimpleNamespace(
        id=uuid4(),
        requested_by=uuid4(),
        document_id=document_id,
        job_type="pdf",
        status="pending",
        progress=0,
        attempt_count=0,
        error_code=None,
        error_message=None,
        started_at=None,
        finished_at=None,
        created_at=now,
        updated_at=now,
    )

    knowledge = KnowledgeItemRead.model_validate(knowledge_source)
    knowledge_page = KnowledgeItemPage.model_validate(
        SimpleNamespace(items=[knowledge_source], page=1, page_size=20, total=1)
    )
    document = DocumentRead.model_validate(document_source)
    document_page = DocumentPage.model_validate(
        SimpleNamespace(items=[document_source], page=1, page_size=20, total=1)
    )
    job = IngestionJobRead.model_validate(job_source)
    accepted = DocumentAccepted.model_validate(
        SimpleNamespace(document=document_source, job=job_source)
    )

    assert knowledge.id == knowledge_source.id
    assert knowledge_page.items[0].id == knowledge_source.id
    assert document.id == document_id
    assert document_page.items[0].id == document_id
    assert job.document_id == document_id
    assert accepted.document.id == document_id
    for value in (knowledge, knowledge_page, document, document_page, job, accepted):
        with pytest.raises(ValidationError):
            value.page = 2
