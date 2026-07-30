from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.modules.system.router import get_readiness_service
from app.modules.system.service import ReadinessResult


class FakeReadinessService:
    def __init__(self, result: ReadinessResult) -> None:
        self.result = result

    async def check(self) -> ReadinessResult:
        return self.result


def build_client(result: ReadinessResult) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_readiness_service] = lambda: FakeReadinessService(result)
    return TestClient(app)


def test_live_does_not_depend_on_database_readiness() -> None:
    client = build_client(
        ReadinessResult(False, {"config": "ok", "database": "unavailable", "pgvector": "unknown"})
    )

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_returns_200_when_all_checks_pass() -> None:
    client = build_client(
        ReadinessResult(True, {"config": "ok", "database": "ok", "pgvector": "0.8.5"})
    )

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"config": "ok", "database": "ok", "pgvector": "0.8.5"},
    }


def test_ready_returns_503_without_leaking_database_error() -> None:
    client = build_client(
        ReadinessResult(False, {"config": "ok", "database": "unavailable", "pgvector": "unknown"})
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "password" not in response.text.lower()


def test_request_id_is_accepted_and_returned() -> None:
    client = build_client(
        ReadinessResult(True, {"config": "ok", "database": "ok", "pgvector": "0.8.5"})
    )

    response = client.get("/health/live", headers={"X-Request-ID": "m1-health-001"})

    assert response.headers["X-Request-ID"] == "m1-health-001"
