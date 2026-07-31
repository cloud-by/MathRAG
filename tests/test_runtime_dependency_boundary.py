from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.utils import canonicalize_name, canonicalize_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELETED_RUNTIME_FILES = (
    "app/services/retriever.py",
    "app/services/vector_store.py",
    "app/services/embedding_service.py",
    "scripts/build_index.py",
    "scripts/rebuild_id_map_from_chunks.py",
)
LEGACY_FAISS_ALLOWLIST = frozenset(
    {
        "scripts/capture_retrieval_baseline.py",
        "scripts/evaluate_pgvector_retrieval.py",
        "scripts/legacy_faiss_retriever.py",
    }
)
ONLINE_SOURCES = (
    "app/api/chat.py",
    "app/frontend/index.html",
    "app/main.py",
    "app/modules/knowledge/search_service.py",
    "app/services/rag_pipeline.py",
)


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _locked_packages(relative_path: str) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw_line in _read(relative_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")) or "==" not in line:
            continue
        raw_name, raw_version = line.split("==", 1)
        name = canonicalize_name(raw_name.strip())
        version = canonicalize_version(raw_version.split(";", 1)[0].strip())
        assert name not in packages, f"lock 中存在重复包：{name}"
        packages[name] = version
    return packages


def test_online_sources_and_import_graph_do_not_load_faiss_artifacts() -> None:
    combined = "\n".join(_read(path) for path in ONLINE_SOURCES).lower()
    for forbidden in (
        "faiss",
        "id_map.json",
        "kb_chunks.jsonl",
        "app.services.retriever",
    ):
        assert forbidden not in combined

    command = (
        "import json, sys; "
        "import app.main; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    imported_modules = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not any(name == "faiss" or name.startswith("faiss.") for name in imported_modules)


def test_legacy_online_modules_and_index_builder_are_deleted() -> None:
    for relative_path in DELETED_RUNTIME_FILES:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path


def test_live_python_callers_do_not_reference_deleted_runtime_interfaces() -> None:
    sources = [
        path
        for root in (PROJECT_ROOT / "app", PROJECT_ROOT / "scripts")
        for path in root.rglob("*.py")
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for forbidden in (
        "app.services.retriever",
        "app.services.vector_store",
        "app.services.embedding_service",
        "scripts.build_index",
        "settings.FAISS_INDEX_PATH",
        "settings.ID_MAP_PATH",
        "settings.USE_INNER_PRODUCT",
    ):
        assert forbidden not in combined


def test_only_frozen_evaluation_allowlist_may_reference_legacy_faiss_artifacts() -> None:
    for root in (PROJECT_ROOT / "app", PROJECT_ROOT / "scripts"):
        for path in root.rglob("*.py"):
            relative_path = path.relative_to(PROJECT_ROOT).as_posix()
            if relative_path in LEGACY_FAISS_ALLOWLIST:
                continue

            source = path.read_text(encoding="utf-8")
            lowered = source.lower()
            for forbidden in ("faiss.index", "id_map.json", "faiss.write_index"):
                assert forbidden not in lowered, relative_path

            tree = ast.parse(source, filename=relative_path)
            imported_modules = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            imported_modules.update(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
            assert not any(
                name == "faiss" or name.startswith("faiss.")
                for name in imported_modules
            ), relative_path


def test_runtime_settings_drop_faiss_only_fields_but_keep_processed_input() -> None:
    from app.core.config import Settings

    for removed_name in ("FAISS_INDEX_PATH", "ID_MAP_PATH", "USE_INNER_PRODUCT"):
        assert not hasattr(Settings, removed_name)
        assert removed_name not in _read(".env.example")
    assert hasattr(Settings, "PROCESSED_KB_PATH")


def test_demo_query_uses_async_knowledge_search_without_external_resources(
    capsys,
) -> None:
    from scripts import demo_query

    calls: list[tuple[list[str], int]] = []

    class FakeHit:
        def to_reference(self, rank: int) -> dict[str, object]:
            return {
                "rank": rank,
                "title": "测试知识",
                "category": "concept",
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "score": 0.9,
                "keywords": [],
                "content": "内容",
                "example": "",
                "steps": [],
                "answer_context": "",
            }

    class FakeSearch:
        async def search(self, queries: list[str], *, top_k: int) -> list[FakeHit]:
            calls.append((list(queries), top_k))
            return [FakeHit()]

    assert inspect.iscoroutinefunction(demo_query.run_once)
    asyncio.run(
        demo_query.run_once(
            question="测试问题",
            top_k=2,
            show_context=False,
            search_service=FakeSearch(),
        )
    )

    assert calls == [(["测试问题"], 2)]
    assert "测试知识" in capsys.readouterr().out
    source = inspect.getsource(demo_query)
    assert "build_knowledge_search_service" in source
    assert "asyncio.run" in source
    assert "app.services.retriever" not in source


def test_demo_async_main_disposes_provider_then_engine_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import demo_query

    events: list[str] = []

    class FakeSearch:
        async def search(self, queries: list[str], *, top_k: int) -> list[object]:
            events.append("search")
            return []

    def build_search_service() -> FakeSearch:
        events.append("build")
        return FakeSearch()

    async def dispose_provider() -> None:
        events.append("dispose_provider")

    async def dispose_database() -> None:
        events.append("dispose_engine")

    monkeypatch.setattr(
        demo_query,
        "parse_args",
        lambda: SimpleNamespace(
            interactive=False,
            question="测试问题",
            show_context=False,
            top_k=1,
        ),
    )
    monkeypatch.setattr(demo_query, "build_knowledge_search_service", build_search_service)
    monkeypatch.setattr(
        demo_query,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(demo_query, "dispose_engine", dispose_database, raising=False)

    asyncio.run(demo_query.async_main())

    assert events == ["build", "search", "dispose_provider", "dispose_engine"]


def test_demo_async_main_preserves_business_error_when_both_cleanups_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import demo_query

    events: list[str] = []

    class BusinessError(RuntimeError):
        pass

    class FakeSearch:
        async def search(self, queries: list[str], *, top_k: int) -> list[object]:
            events.append("search")
            raise BusinessError("business")

    async def dispose_provider() -> None:
        events.append("dispose_provider")
        raise RuntimeError("provider cleanup")

    async def dispose_database() -> None:
        events.append("dispose_engine")
        raise RuntimeError("database cleanup")

    monkeypatch.setattr(
        demo_query,
        "parse_args",
        lambda: SimpleNamespace(
            interactive=False,
            question="测试问题",
            show_context=False,
            top_k=1,
        ),
    )
    monkeypatch.setattr(demo_query, "build_knowledge_search_service", FakeSearch)
    monkeypatch.setattr(
        demo_query,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(demo_query, "dispose_engine", dispose_database, raising=False)

    with pytest.raises(BusinessError, match="business"):
        asyncio.run(demo_query.async_main())

    assert events == ["search", "dispose_provider", "dispose_engine"]


def test_demo_async_main_raises_first_cleanup_error_after_trying_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import demo_query

    events: list[str] = []

    class ProviderCleanupError(RuntimeError):
        pass

    class FakeSearch:
        async def search(self, queries: list[str], *, top_k: int) -> list[object]:
            return []

    async def dispose_provider() -> None:
        events.append("dispose_provider")
        raise ProviderCleanupError("provider cleanup")

    async def dispose_database() -> None:
        events.append("dispose_engine")
        raise RuntimeError("database cleanup")

    monkeypatch.setattr(
        demo_query,
        "parse_args",
        lambda: SimpleNamespace(
            interactive=False,
            question="测试问题",
            show_context=False,
            top_k=1,
        ),
    )
    monkeypatch.setattr(demo_query, "build_knowledge_search_service", FakeSearch)
    monkeypatch.setattr(
        demo_query,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(demo_query, "dispose_engine", dispose_database, raising=False)

    with pytest.raises(ProviderCleanupError, match="provider cleanup"):
        asyncio.run(demo_query.async_main())

    assert events == ["dispose_provider", "dispose_engine"]


def test_demo_async_main_consecutive_runs_do_not_reuse_closed_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import demo_query

    resources: list[FakeSearch] = []

    class FakeSearch:
        def __init__(self) -> None:
            self.closed = False
            self.search_count = 0

        async def search(self, queries: list[str], *, top_k: int) -> list[object]:
            assert not self.closed
            self.search_count += 1
            return []

    def build_search_service() -> FakeSearch:
        resource = FakeSearch()
        resources.append(resource)
        return resource

    async def dispose_provider() -> None:
        resources[-1].closed = True

    async def dispose_database() -> None:
        return None

    monkeypatch.setattr(
        demo_query,
        "parse_args",
        lambda: SimpleNamespace(
            interactive=False,
            question="测试问题",
            show_context=False,
            top_k=1,
        ),
    )
    monkeypatch.setattr(demo_query, "build_knowledge_search_service", build_search_service)
    monkeypatch.setattr(
        demo_query,
        "dispose_embedding_provider",
        dispose_provider,
        raising=False,
    )
    monkeypatch.setattr(demo_query, "dispose_engine", dispose_database, raising=False)

    asyncio.run(demo_query.async_main())
    asyncio.run(demo_query.async_main())

    assert len(resources) == 2
    assert resources[0] is not resources[1]
    assert [resource.search_count for resource in resources] == [1, 1]
    assert all(resource.closed for resource in resources)


def test_runtime_and_evaluation_dependency_locks_are_separated() -> None:
    runtime_requirements = _read("requirements.txt").lower()
    runtime_lock = _read("requirements.lock.txt").lower()
    evaluation_requirements = _read("requirements-evaluation.txt").lower()
    evaluation_lock = _read("requirements-evaluation.lock.txt").lower()

    assert "faiss-cpu" not in runtime_requirements
    assert "faiss-cpu" not in runtime_lock
    assert "-r requirements.txt" in evaluation_requirements
    assert "faiss-cpu" in evaluation_requirements
    assert "faiss-cpu" in evaluation_lock
    runtime_packages = _locked_packages("requirements.lock.txt")
    evaluation_packages = _locked_packages("requirements-evaluation.lock.txt")
    assert set(evaluation_packages) - set(runtime_packages) == {"faiss-cpu"}
    assert {
        name: version
        for name, version in evaluation_packages.items()
        if name != "faiss-cpu"
    } == runtime_packages


def test_runtime_install_instructions_use_the_compiled_lock() -> None:
    sources = [
        path
        for root in (PROJECT_ROOT / "app", PROJECT_ROOT / "scripts")
        for path in root.rglob("*.py")
        if path.relative_to(PROJECT_ROOT).as_posix() not in LEGACY_FAISS_ALLOWLIST
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    assert "pip install -r requirements.txt" not in combined


def test_docker_installs_only_the_runtime_lock() -> None:
    dockerfile = _read("Dockerfile").lower()

    assert "requirements.lock.txt" in dockerfile
    assert "pip install --no-cache-dir -r requirements.lock.txt" in dockerfile
    assert "requirements-evaluation" not in dockerfile
    assert "faiss" not in dockerfile


def test_readme_documents_pgvector_workflow_and_no_longer_builds_faiss() -> None:
    readme = _read("README.md").lower()

    for required in (
        "alembic upgrade head",
        "scripts.import_legacy_knowledge",
        "scripts.reindex_knowledge",
        "scripts.demo_query",
        "scripts.evaluate_pgvector_retrieval",
        "requirements-evaluation",
        "回滚",
    ):
        assert required in readme
    assert "scripts.build_index" not in readme
    assert "pip install -r requirements.txt" not in readme
    assert "pip install -r requirements.lock.txt" in readme
    assert "pip install -r requirements-evaluation.lock.txt" in readme
    assert (
        "pytest -q --ignore=tests/evaluation "
        "--ignore=tests/test_retrieval_baseline.py"
    ) in readme
    assert "git switch --detach cd77635" in readme
    assert "docker compose stop mathrag" in readme
    assert 'docker image tag "${mathrag_rollback_image}" mathrag:local' in readme
    assert "docker compose up -d --no-build mathrag" in readme
    assert "chmod a-w data/index/faiss.index data/index/id_map.json" in readme
    assert '${mathrag_base_url}/health/live' in readme
    assert '${mathrag_base_url}/health/ready' in readme
    assert "禁止同一在线版本长期并行" in readme
