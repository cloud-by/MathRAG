from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_env_example(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        assert separator, f"无效的 .env.example 行：{raw_line}"
        values[name.strip()] = value.strip()
    return values


def test_run_main_uses_debug_setting_for_reload(monkeypatch) -> None:
    import run

    captured: dict = {}
    fake_settings = SimpleNamespace(
        APP_HOST="127.0.0.9",
        APP_PORT=8123,
        DEBUG=False,
    )

    def fake_uvicorn_run(app: str, **kwargs) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(run, "settings", fake_settings)
    monkeypatch.setattr(run.uvicorn, "run", fake_uvicorn_run)

    run.main()

    assert captured == {
        "app": "app.main:app",
        "host": "127.0.0.9",
        "port": 8123,
        "reload": False,
    }


def test_env_example_contains_runtime_settings_and_only_fake_secrets() -> None:
    values = _parse_env_example(PROJECT_ROOT / ".env.example")

    required_names = {
        "APP_NAME",
        "APP_HOST",
        "APP_PORT",
        "DEBUG",
        "TOP_K",
        "SESSION_SECRET",
        "SESSION_TTL_SECONDS",
        "ALLOWED_ORIGINS",
    }
    assert required_names <= values.keys()
    assert "APP_DEBUG" not in values
    assert values["SESSION_SECRET"] == ""
    assert values["SESSION_TTL_SECONDS"] == "604800"
    assert values["ALLOWED_ORIGINS"] == (
        "http://127.0.0.1:8000,http://localhost:8000"
    )

    sensitive_names = {
        name
        for name in values
        if name.endswith(("API_KEY", "_PASSWORD", "_SECRET", "_TOKEN"))
    }
    assert sensitive_names
    for name in sensitive_names:
        assert values[name].lower() in {"", "sk-xxxx", "xxxx", "changeme"}


def test_readme_documents_local_compose_database_workflow() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    docker_section = readme.split("## Docker 部署", maxsplit=1)[1].split("\n---", maxsplit=1)[0]

    assert "cloudby/mathrag:latest" not in docker_section
    assert "docker compose up -d postgres" in docker_section
    assert ".\\.venv\\Scripts\\alembic.exe upgrade head" in docker_section
    assert "docker compose up -d --build mathrag" in docker_section
    assert "mathrag:local" in docker_section
    assert "/health/live" in docker_section
    assert "/health/ready" in docker_section


def test_production_compose_requires_security_environment() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    mathrag_service = compose.split("  mathrag:", maxsplit=1)[1]

    assert "SESSION_SECRET: ${SESSION_SECRET:?" in mathrag_service
    assert "ALLOWED_ORIGINS: ${ALLOWED_ORIGINS:?" in mathrag_service
