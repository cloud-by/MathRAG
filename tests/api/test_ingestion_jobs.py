"""导入任务查询、取消与重试 API 合同测试。"""

from __future__ import annotations

from uuid import UUID

from app.modules.ingestion.errors import IngestionJobStateConflictError
from app.modules.ingestion.repository import JobSnapshot

from tests.api.test_documents import (
    ADMIN_ID,
    JOB_ID,
    FakeIngestionService,
    _build_client,
    _job,
    _safe_headers,
)


class FakeJobService(FakeIngestionService):
    def __init__(self) -> None:
        super().__init__()
        self.current_job = _job()
        self.retry_snapshot = JobSnapshot(
            job_id=JOB_ID,
            document_id=self.current_job.document_id,
            requested_by=ADMIN_ID,
            job_type="pdf",
            attempt_count=2,
            request_payload={"category": "代数", "private": "不得公开"},
        )

    async def get_job(self, job_id: UUID):
        self.calls.append(("get_job", job_id))
        return self.current_job

    async def cancel(self, job_id: UUID):
        self.calls.append(("cancel", job_id))
        self.current_job = self.current_job.model_copy(
            update={"status": "cancelled"}
        )
        return self.current_job

    async def claim_retry(self, job_id: UUID):
        self.calls.append(("claim_retry", job_id))
        if self.current_job.status != "failed":
            raise IngestionJobStateConflictError()
        self.current_job = self.current_job.model_copy(
            update={"status": "running", "attempt_count": 2}
        )
        return self.retry_snapshot

    async def resume_retry(self, snapshot: JobSnapshot) -> None:
        self.calls.append(("resume_retry", snapshot))


def _job_client():
    client, _base = _build_client()
    service = FakeJobService()
    from app.modules.ingestion.router import get_ingestion_service

    client.app.dependency_overrides[get_ingestion_service] = lambda: service
    return client, service


def test_admin_gets_and_cancels_job_without_internal_payload() -> None:
    client, service = _job_client()

    fetched = client.get(
        f"/api/v1/ingestion-jobs/{JOB_ID}",
        headers={"X-Test-Role": "admin"},
    )
    cancelled = client.post(
        f"/api/v1/ingestion-jobs/{JOB_ID}/cancel",
        headers=_safe_headers(client, "admin"),
    )

    assert fetched.status_code == 200
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert "request_payload" not in fetched.text
    assert "private" not in fetched.text
    assert service.calls == [("get_job", JOB_ID), ("cancel", JOB_ID)]


def test_retry_claims_before_scheduling_and_returns_202_safe_job() -> None:
    client, service = _job_client()
    service.current_job = _job(status="failed", attempt_count=1)

    response = client.post(
        f"/api/v1/ingestion-jobs/{JOB_ID}/retry",
        headers=_safe_headers(client, "admin"),
    )

    assert response.status_code == 202
    assert response.json()["status"] == "running"
    assert response.json()["attempt_count"] == 2
    assert "request_payload" not in response.text
    assert [name for name, _value in service.calls] == [
        "claim_retry",
        "get_job",
        "resume_retry",
    ]
    assert service.calls[2] == ("resume_retry", service.retry_snapshot)


def test_job_routes_reject_anonymous_user_and_missing_csrf() -> None:
    client, service = _job_client()

    anonymous = client.get(f"/api/v1/ingestion-jobs/{JOB_ID}")
    user = client.get(
        f"/api/v1/ingestion-jobs/{JOB_ID}",
        headers={"X-Test-Role": "user"},
    )
    missing_csrf = client.post(
        f"/api/v1/ingestion-jobs/{JOB_ID}/cancel",
        headers={"X-Test-Role": "admin", "Origin": "http://localhost:5173"},
    )

    assert anonymous.status_code == 401
    assert user.status_code == 403
    assert missing_csrf.status_code == 403
    assert service.calls == []


def test_completed_job_retry_returns_stable_conflict_without_scheduling() -> None:
    client, service = _job_client()
    service.current_job = _job(status="completed", attempt_count=1)

    response = client.post(
        f"/api/v1/ingestion-jobs/{JOB_ID}/retry",
        headers=_safe_headers(client, "admin"),
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INGESTION_JOB_STATE_CONFLICT"
    assert service.calls == [("claim_retry", JOB_ID)]


def test_openapi_exposes_job_query_cancel_and_retry() -> None:
    client, _service = _job_client()

    schema = client.get("/openapi.json").json()["paths"]

    assert set(schema["/api/v1/ingestion-jobs/{job_id}"]) == {"get"}
    assert set(schema["/api/v1/ingestion-jobs/{job_id}/cancel"]) == {"post"}
    assert set(schema["/api/v1/ingestion-jobs/{job_id}/retry"]) == {"post"}
    assert "202" in schema["/api/v1/ingestion-jobs/{job_id}/retry"]["post"][
        "responses"
    ]
