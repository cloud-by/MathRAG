from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_builds_frontend_then_pins_python_runtime() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith(
        "FROM node:24.11.1-bookworm-slim AS frontend-build\n"
    )
    assert "FROM python:3.11.9-slim AS runtime" in dockerfile
    assert "WORKDIR /frontend" in dockerfile
    assert "COPY frontend/package.json frontend/package-lock.json ./" in dockerfile
    assert "RUN npm ci" in dockerfile
    assert "COPY frontend/ ./" in dockerfile
    assert "RUN npm run build" in dockerfile
    assert "pip install --no-cache-dir -r requirements.lock.txt" in dockerfile
    assert (
        "COPY --from=frontend-build /frontend/dist /app/frontend/dist"
        in dockerfile
    )
    runtime = dockerfile.split("FROM python:3.11.9-slim AS runtime", 1)[1]
    assert "node_modules" not in runtime
    assert '"-w", "1"' in runtime


def test_compose_uses_local_app_and_pinned_pgvector() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    app = compose["services"]["mathrag"]
    postgres = compose["services"]["postgres"]

    assert app["build"] == {"context": ".", "dockerfile": "Dockerfile"}
    assert app["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "-w 1" in app["command"]
    assert postgres["image"] == "pgvector/pgvector:0.8.5-pg18-bookworm"
    assert "pg_isready" in postgres["healthcheck"]["test"][1]
    assert "postgres_data:/var/lib/postgresql" in postgres["volumes"]
    assert "postgres_data" in compose["volumes"]
    assert app["environment"]["UPLOAD_DIR"] == "/app/data/uploads"
    assert "upload_data:/app/data/uploads" in app["volumes"]
    assert "upload_data" in compose["volumes"]
    assert all(
        not str(volume).endswith(":/app/frontend/dist")
        for volume in app["volumes"]
    )


def test_compose_exposes_database_only_on_loopback() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    assert compose["services"]["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
