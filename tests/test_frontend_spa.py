from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings as app_settings


INDEX_MARKER = '<div id="app">vue-spa-test</div>'


@pytest.fixture
def frontend_dist(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        f"<!doctype html><html><body>{INDEX_MARKER}</body></html>",
        encoding="utf-8",
    )
    (assets / "app-a1b2c3.js").write_text(
        "console.log('mathrag')",
        encoding="utf-8",
    )
    (assets / "app-a1b2c3.css").write_text(
        ":root{color:#172033}",
        encoding="utf-8",
    )
    return dist


def create_client(monkeypatch: pytest.MonkeyPatch, dist: Path) -> TestClient:
    from app import main

    monkeypatch.setattr(
        main,
        "settings",
        replace(app_settings, FRONTEND_DIST_DIR=dist),
    )
    return TestClient(main.create_app())


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/chat",
        "/conversations/11111111-1111-4111-8111-111111111111",
        "/knowledge/new",
    ],
)
def test_get_frontend_routes_fall_back_to_vue_index(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
    path: str,
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    response = client.get(path, headers={"Accept": "text/html"})

    assert response.status_code == 200
    assert INDEX_MARKER in response.text
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-cache"


def test_head_frontend_route_returns_index_headers_without_body(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    response = client.head("/chat", headers={"Accept": "text/html"})

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-type"].startswith("text/html")


@pytest.mark.parametrize(
    ("path", "content_types"),
    [
        (
            "/assets/app-a1b2c3.js",
            ("text/javascript", "application/javascript"),
        ),
        ("/assets/app-a1b2c3.css", ("text/css",)),
    ],
)
def test_hashed_assets_have_mime_and_immutable_cache_headers(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
    path: str,
    content_types: tuple[str, ...],
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(content_types)
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


def test_missing_static_asset_and_non_html_navigation_do_not_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    missing_asset = client.get(
        "/assets/missing.js",
        headers={"Accept": "text/html"},
    )
    json_navigation = client.get("/chat", headers={"Accept": "application/json"})

    assert missing_asset.status_code == 404
    assert missing_asset.headers["content-type"].startswith("application/json")
    assert json_navigation.status_code == 404
    assert json_navigation.headers["content-type"].startswith("application/json")


def test_api_misses_and_removed_legacy_chat_return_json_404(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    v1_missing = client.get(
        "/api/v1/not-found",
        headers={"Accept": "text/html"},
    )
    legacy_chat = client.post(
        "/api/" + "chat",
        json={"question": "测试", "history": [], "top_k": 3},
        headers={"Accept": "text/html"},
    )

    assert v1_missing.status_code == 404
    assert v1_missing.headers["content-type"].startswith("application/json")
    assert v1_missing.json()["error"]["code"] == "HTTP_ERROR"
    assert legacy_chat.status_code == 404
    assert legacy_chat.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    "path",
    ["/health", "/docs", "/redoc", "/openapi.json"],
)
def test_reserved_application_routes_are_not_replaced_by_spa(
    monkeypatch: pytest.MonkeyPatch,
    frontend_dist: Path,
    path: str,
) -> None:
    client = create_client(monkeypatch, frontend_dist)

    response = client.get(path, headers={"Accept": "text/html"})

    assert response.status_code == 200
    assert INDEX_MARKER not in response.text


def test_missing_dist_does_not_block_startup_and_root_is_explicit_503(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = create_client(monkeypatch, tmp_path / "missing-dist")

    response = client.get("/", headers={"Accept": "text/html"})

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "前端构建产物不可用。"}
