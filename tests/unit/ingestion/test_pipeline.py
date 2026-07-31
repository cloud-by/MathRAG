"""导入 pipeline 的纯抽取、事件顺序和外部调用边界测试。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.modules.ingestion.errors import DocumentPdfInvalidError
from app.modules.ingestion.repository import (
    DocumentSnapshot,
    JobSnapshot,
    PipelineChunkSnapshot,
)
from app.modules.ingestion.service import IngestionService
from app.modules.knowledge.errors import EmbeddingUnavailableError
from app.services import knowledge_extractor
from app.services.knowledge_extractor import KnowledgeDraft


NOW = datetime(2026, 7, 31, 15, 0, tzinfo=UTC)
JOB_ID = UUID(int=801)
DOCUMENT_ID = UUID(int=802)
OWNER_ID = UUID(int=803)


def _raw_item() -> dict[str, object]:
    return {
        "category": " 代数 ",
        "title": " 一元一次方程 ",
        "keywords": ["方程", " 方程 ", "移项"],
        "content": " 通过移项求解。 ",
        "example": " x+1=2 ",
        "steps": ["移项", " 求解 "],
        "difficulty": "中等",
    }


def test_extract_knowledge_drafts_is_independent_from_jsonl_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        knowledge_extractor,
        "chat_json",
        lambda **_kwargs: SimpleNamespace(data={"items": [_raw_item()]}),
    )
    monkeypatch.setattr(
        knowledge_extractor,
        "generate_next_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("纯抽取接口不能访问 JSONL id")
        ),
    )

    drafts = knowledge_extractor.extract_knowledge_drafts(" 方程教材 ")

    assert drafts == [
        KnowledgeDraft(
            category="代数",
            title="一元一次方程",
            keywords=("方程", "移项"),
            content="通过移项求解。",
            example="x+1=2",
            steps=("移项", "求解"),
            difficulty="medium",
        )
    ]


class _Transaction:
    def __init__(self, sessions: "_Sessions") -> None:
        self.sessions = sessions

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.sessions.events.append(self.sessions.commit_names.pop(0))
        return False


class _Session:
    def __init__(self, sessions: "_Sessions") -> None:
        self.sessions = sessions

    async def __aenter__(self) -> "_Session":
        self.sessions.active += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.sessions.active -= 1
        return False

    def begin(self) -> _Transaction:
        return _Transaction(self.sessions)


class _Sessions:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.active = 0
        self.commit_names = [
            "claim.tx.commit",
            "knowledge.tx.commit",
            "finalize.tx.commit",
        ]

    def __call__(self) -> _Session:
        return _Session(self)


class _Repository:
    def __init__(self, sessions: _Sessions) -> None:
        self.sessions = sessions

    async def claim_pending(self, job_id: UUID, now: datetime) -> JobSnapshot:
        assert job_id == JOB_ID
        return JobSnapshot(
            job_id=JOB_ID,
            document_id=DOCUMENT_ID,
            requested_by=OWNER_ID,
            job_type="pdf",
            attempt_count=1,
            request_payload={"category": "代数"},
        )

    async def get_document_snapshot(self, document_id: UUID) -> DocumentSnapshot:
        assert document_id == DOCUMENT_ID
        return DocumentSnapshot(
            document_id=DOCUMENT_ID,
            storage_path="2026/07/source.pdf",
            status="processing",
        )

    async def list_pipeline_chunks(
        self,
        job_id: UUID,
    ) -> list[PipelineChunkSnapshot]:
        return []

    async def create_pipeline_items(
        self,
        snapshot: JobSnapshot,
        drafts: list[dict[str, object]],
    ) -> list[PipelineChunkSnapshot]:
        assert drafts[0]["title"] == "一元一次方程"
        return [
            PipelineChunkSnapshot(
                chunk_id=UUID(int=804),
                item_id=UUID(int=805),
                retrieval_text="检索文本",
            )
        ]

    async def finalize_pipeline(self, **kwargs: object) -> bool:
        return True


def test_pipeline_has_fixed_event_order_and_no_session_during_external_calls(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    sessions = _Sessions(events)
    repository = _Repository(sessions)
    source = tmp_path / "2026" / "07" / "source.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-test")

    def pdf_extractor(path: Path, *, max_pages: int):
        assert sessions.active == 0
        assert path == source
        events.append("source.read")
        return SimpleNamespace(text="教材正文", page_count=1, title=None)

    def extractor(text: str, category: str | None):
        assert sessions.active == 0
        events.append("llm.extract")
        return [
            KnowledgeDraft(
                category=category or "concept",
                title="一元一次方程",
                keywords=("方程",),
                content="通过移项求解。",
                example="",
                steps=("移项",),
                difficulty="medium",
            )
        ]

    class Provider:
        model = "test-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            assert sessions.active == 0
            assert texts == ["检索文本"]
            events.append("embedding.call")
            return [[1.0, *([0.0] * 1023)]]

    service = IngestionService(
        sessions,  # type: ignore[arg-type]
        SimpleNamespace(),
        repository_factory=lambda _session: repository,  # type: ignore[arg-type]
        embedding_provider=Provider(),  # type: ignore[arg-type]
        draft_extractor=extractor,
        pdf_extractor=pdf_extractor,
        upload_root=tmp_path,
        now=lambda: NOW,
    )

    asyncio.run(service.run_pending(JOB_ID))

    assert events == [
        "claim.tx.commit",
        "source.read",
        "llm.extract",
        "knowledge.tx.commit",
        "embedding.call",
        "finalize.tx.commit",
    ]


class _QuietTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False


class _QuietSession:
    async def __aenter__(self) -> "_QuietSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return False

    def begin(self) -> _QuietTransaction:
        return _QuietTransaction()


class _FailureRepository:
    def __init__(self, failure_stage: str) -> None:
        self.failure_stage = failure_stage
        self.failed: tuple[str, str] | None = None
        self.states = {
            "job": "running",
            "document": "processing",
            "item": "indexing",
            "chunk": "pending",
        }

    async def claim_pending(self, job_id: UUID, now: datetime) -> JobSnapshot:
        return JobSnapshot(
            job_id=JOB_ID,
            document_id=DOCUMENT_ID,
            requested_by=OWNER_ID,
            job_type="pdf",
            attempt_count=1,
            request_payload={},
        )

    async def get_document_snapshot(self, document_id: UUID) -> DocumentSnapshot:
        return DocumentSnapshot(DOCUMENT_ID, "source.pdf", "processing")

    async def list_pipeline_chunks(
        self,
        job_id: UUID,
    ) -> list[PipelineChunkSnapshot]:
        return []

    async def create_pipeline_items(self, snapshot, drafts):
        if self.failure_stage == "database":
            raise SQLAlchemyError("postgresql://secret@database")
        return [PipelineChunkSnapshot(UUID(int=901), UUID(int=902), "检索文本")]

    async def finalize_pipeline(self, **kwargs: object) -> bool:
        return True

    async def fail_pipeline(
        self,
        *,
        snapshot: JobSnapshot,
        code: str,
        message: str,
        now: datetime,
    ) -> bool:
        self.failed = (code, message)
        self.states = {key: "failed" for key in self.states}
        return True


@pytest.mark.parametrize(
    ("failure_stage", "expected_code", "expected_message"),
    [
        (
            "pdf",
            "INGESTION_PDF_INVALID",
            "PDF 文档无法读取或解析。",
        ),
        (
            "llm",
            "INGESTION_LLM_UNAVAILABLE",
            "知识抽取服务暂时不可用。",
        ),
        (
            "embedding",
            "INGESTION_EMBEDDING_UNAVAILABLE",
            "知识向量化服务暂时不可用。",
        ),
        (
            "database",
            "INGESTION_DATABASE_UNAVAILABLE",
            "数据库服务暂时不可用。",
        ),
    ],
)
def test_pipeline_failures_use_new_session_and_stable_sanitized_summary(
    tmp_path: Path,
    failure_stage: str,
    expected_code: str,
    expected_message: str,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-test")
    repository = _FailureRepository(failure_stage)

    def pdf_extractor(_path: Path, *, max_pages: int):
        if failure_stage == "pdf":
            raise DocumentPdfInvalidError()
        return SimpleNamespace(text="教材", page_count=1, title=None)

    def draft_extractor(text: str, category: str | None):
        if failure_stage == "llm":
            raise RuntimeError("api-key=secret-value")
        return [
            KnowledgeDraft(
                category="代数",
                title="方程",
                keywords=("方程",),
                content="方程知识。",
                example="",
                steps=("求解",),
                difficulty="easy",
            )
        ]

    class Provider:
        model = "failure-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            if failure_stage == "embedding":
                raise EmbeddingUnavailableError("Bearer secret-token")
            return [[1.0, *([0.0] * 1023)]]

    service = IngestionService(
        lambda: _QuietSession(),  # type: ignore[arg-type]
        SimpleNamespace(),
        repository_factory=lambda _session: repository,  # type: ignore[arg-type]
        embedding_provider=Provider(),  # type: ignore[arg-type]
        draft_extractor=draft_extractor,
        pdf_extractor=pdf_extractor,
        upload_root=tmp_path,
        now=lambda: NOW,
    )

    asyncio.run(service.run_pending(JOB_ID))

    assert repository.failed == (expected_code, expected_message)
    assert set(repository.states.values()) == {"failed"}
    assert "secret" not in repr(repository.failed).lower()


@pytest.mark.parametrize(
    "returned",
    [
        [],
        [[1.0, *([0.0] * 1022)]],
        [[float("nan"), *([0.0] * 1023)]],
        [[0.0] * 1024],
    ],
)
def test_pipeline_rejects_invalid_embedding_contract(
    tmp_path: Path,
    returned: list[list[float]],
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-test")
    repository = _FailureRepository("vector")

    class Provider:
        model = "vector-model"
        dimensions = 1024

        async def embed_texts(self, texts):
            return returned

    service = IngestionService(
        lambda: _QuietSession(),  # type: ignore[arg-type]
        SimpleNamespace(),
        repository_factory=lambda _session: repository,  # type: ignore[arg-type]
        embedding_provider=Provider(),  # type: ignore[arg-type]
        draft_extractor=lambda _text, _category: [
            KnowledgeDraft(
                category="代数",
                title="方程",
                keywords=("方程",),
                content="方程知识。",
                example="",
                steps=("求解",),
                difficulty="easy",
            )
        ],
        pdf_extractor=lambda _path, *, max_pages: SimpleNamespace(
            text="教材", page_count=1, title=None
        ),
        upload_root=tmp_path,
        now=lambda: NOW,
    )

    asyncio.run(service.run_pending(JOB_ID))

    assert repository.failed == (
        "INGESTION_EMBEDDING_UNAVAILABLE",
        "知识向量化服务暂时不可用。",
    )
