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

    required_names = {"APP_NAME", "APP_HOST", "APP_PORT", "DEBUG", "TOP_K"}
    assert required_names <= values.keys()
    assert "APP_DEBUG" not in values

    sensitive_names = {
        name
        for name in values
        if name.endswith(("API_KEY", "_PASSWORD", "_SECRET", "_TOKEN"))
    }
    assert sensitive_names
    for name in sensitive_names:
        assert values[name].lower() in {"", "sk-xxxx", "xxxx", "changeme"}


def test_readme_distinguishes_remote_compose_and_local_image_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    docker_section = readme.split("## Docker 部署", maxsplit=1)[1].split("\n---", maxsplit=1)[0]

    assert "cloudby/mathrag:latest" in docker_section
    assert "docker compose up -d --build" not in docker_section
    assert "docker compose pull" in docker_section
    assert "docker build -t mathrag:local ." in docker_section
    assert "docker run" in docker_section
    assert "mathrag:local" in docker_section
