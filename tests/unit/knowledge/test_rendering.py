"""知识文本渲染规则测试。"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from app.modules.knowledge.rendering import build_answer_context, build_retrieval_text
from scripts import build_kb


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILD_KB_SCRIPT = PROJECT_ROOT / "scripts" / "build_kb.py"


def _legacy_difficulty_to_zh(difficulty: object) -> str:
    return {
        "easy": "简单",
        "medium": "中等",
        "hard": "困难",
    }.get(difficulty, str(difficulty))


def _legacy_retrieval_text(values: Mapping[str, object]) -> str:
    """冻结迁移前脚本的拼接规则，避免用新函数和自己比较。"""
    parts = [
        f"知识点类别：{values['category']}",
        f"知识点标题：{values['title']}",
    ]
    keywords = values["keywords"]
    if keywords:
        parts.append("关键词：" + "，".join(keywords))  # type: ignore[arg-type]
    parts.append("核心内容：" + str(values["content"]))
    if values["example"]:
        parts.append("例题示例：" + str(values["example"]))
    steps = values["steps"]
    if steps:
        parts.append("理解/解题步骤：" + "；".join(steps))  # type: ignore[arg-type]
    parts.append(f"难度：{_legacy_difficulty_to_zh(values['difficulty'])}")
    return "\n".join(parts)


def _legacy_answer_context(values: Mapping[str, object]) -> str:
    """冻结迁移前脚本的回答上下文拼接规则。"""
    lines = [
        f"【{values['title']}】",
        f"类别：{values['category']}",
        f"难度：{_legacy_difficulty_to_zh(values['difficulty'])}",
        str(values["content"]),
    ]
    if values["example"]:
        lines.append(f"示例：{values['example']}")
    steps = values["steps"]
    if steps:
        lines.append("参考步骤：")
        lines.extend(steps)  # type: ignore[arg-type]
    return "\n".join(lines)


def _payload(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "category": "代数",
        "title": "配方法",
        "keywords": ["二次式", "完全平方"],
        "content": "把二次式整理为完全平方。",
        "example": "x^2+2x+1=(x+1)^2",
        "steps": ["补项", "整理"],
        "difficulty": "easy",
    }
    values.update(changes)
    return values


def test_rendering_matches_frozen_legacy_builder_byte_for_byte() -> None:
    values = _payload()

    assert build_retrieval_text(values).encode("utf-8") == _legacy_retrieval_text(
        values
    ).encode("utf-8")
    assert build_answer_context(values).encode("utf-8") == _legacy_answer_context(
        values
    ).encode("utf-8")


def test_rendering_preserves_legacy_empty_optional_field_rules() -> None:
    values = _payload(keywords=[], example="", steps=[])

    assert build_retrieval_text(values) == (
        "知识点类别：代数\n"
        "知识点标题：配方法\n"
        "核心内容：把二次式整理为完全平方。\n"
        "难度：简单"
    )
    assert build_answer_context(values) == (
        "【配方法】\n类别：代数\n难度：简单\n把二次式整理为完全平方。"
    )


def test_build_script_uses_the_shared_rendering_functions() -> None:
    assert build_kb.build_retrieval_text is build_retrieval_text
    assert build_kb.build_answer_context is build_answer_context


def test_build_script_help_runs_in_direct_script_context() -> None:
    result = subprocess.run(
        [sys.executable, str(BUILD_KB_SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "将数学种子知识库预处理为可检索的 chunk 文件" in result.stdout
