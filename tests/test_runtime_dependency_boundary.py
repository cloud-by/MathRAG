from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELETED_RUNTIME_FILES = (
    "app/services/retriever.py",
    "app/services/vector_store.py",
    "app/services/embedding_service.py",
    "scripts/build_index.py",
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


def _locked_packages(relative_path: str) -> set[str]:
    packages: set[str] = set()
    for raw_line in _read(relative_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "-")) or "==" not in line:
            continue
        packages.add(line.split("==", 1)[0].strip().lower().replace("_", "-"))
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
    assert _locked_packages("requirements.lock.txt") <= _locked_packages(
        "requirements-evaluation.lock.txt"
    )


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
