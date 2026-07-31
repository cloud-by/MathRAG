"""导入任务查询、取消与重试 API 合同测试。"""

from __future__ import annotations

from uuid import UUID

from app.modules.ingestion.errors import IngestionJobStateConflictError
from app.modules.ingestion.repository import JobSnapshot
from app.modules.ingestion.schemas import IngestionJobPage

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
        self.jobs = [self.current_job]
        self.total = 1
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

    async def list_jobs(self, **filters):
        self.calls.append(("list_jobs", filters))
        return IngestionJobPage(
            items=self.jobs,
            total=self.total,
            offset=filters["offset"],
            limit=filters["limit"],
        )

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


def test_admin_lists_jobs_with_exact_filters_and_pagination() -> None:
    client, service = _job_client()

    response = client.get(
        "/api/v1/ingestion-jobs"
        f"?status=failed&job_type=pdf&document_id={service.current_job.document_id}"
        "&offset=5&limit=10",
        headers={"X-Test-Role": "admin"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [service.current_job.model_dump(mode="json")],
        "total": 1,
        "offset": 5,
        "limit": 10,
    }
    assert service.calls == [
        (
            "list_jobs",
            {
                "status": "failed",
                "job_type": "pdf",
                "document_id": service.current_job.document_id,
                "offset": 5,
                "limit": 10,
            },
        )
    ]


def test_admin_gets_empty_job_page_metadata() -> None:
    client, service = _job_client()
    service.jobs = []
    service.total = 0

    response = client.get(
        "/api/v1/ingestion-jobs?offset=25&limit=25",
        headers={"X-Test-Role": "admin"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 0,
        "offset": 25,
        "limit": 25,
    }


def test_job_collection_rejects_anonymous_and_ordinary_users() -> None:
    client, service = _job_client()

    anonymous = client.get("/api/v1/ingestion-jobs")
    ordinary = client.get(
        "/api/v1/ingestion-jobs",
        headers={"X-Test-Role": "user"},
    )

    assert anonymous.status_code == 401
    assert ordinary.status_code == 403
    assert service.calls == []


def test_job_collection_rejects_invalid_filters_before_service_call() -> None:
    invalid_queries = (
        "offset=-1",
        "limit=0",
        "limit=101",
        "status=unknown",
        "job_type=unknown",
        "document_id=not-a-uuid",
    )

    for query in invalid_queries:
        client, service = _job_client()
        response = client.get(
            f"/api/v1/ingestion-jobs?{query}",
            headers={"X-Test-Role": "admin"},
        )
        assert response.status_code == 422, query
        assert service.calls == [], query


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

    assert set(schema["/api/v1/ingestion-jobs"]) == {"get"}
    assert set(schema["/api/v1/ingestion-jobs/{job_id}"]) == {"get"}
    assert set(schema["/api/v1/ingestion-jobs/{job_id}/cancel"]) == {"post"}
    assert set(schema["/api/v1/ingestion-jobs/{job_id}/retry"]) == {"post"}
    assert "202" in schema["/api/v1/ingestion-jobs/{job_id}/retry"]["post"][
        "responses"
    ]
