from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_pins_python_runtime() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.11.9-slim\n")
    assert "pip install --no-cache-dir -r requirements.lock.txt" in dockerfile


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


def test_compose_exposes_database_only_on_loopback() -> None:
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )

    assert compose["services"]["postgres"]["ports"] == ["127.0.0.1:5432:5432"]
