"""知识条目的稳定检索文本与回答上下文渲染规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def _difficulty_to_zh(difficulty: object) -> object:
    return {
        "easy": "简单",
        "medium": "中等",
        "hard": "困难",
    }.get(difficulty, difficulty)


def build_retrieval_text(values: Mapping[str, object]) -> str:
    """按旧构建脚本的字节级规则生成向量检索文本。"""
    parts = [
        f"知识点类别：{values['category']}",
        f"知识点标题：{values['title']}",
    ]

    keywords = cast(Sequence[str], values["keywords"])
    if keywords:
        parts.append("关键词：" + "，".join(keywords))

    parts.append("核心内容：" + cast(str, values["content"]))

    example = cast(str, values["example"])
    if example:
        parts.append("例题示例：" + example)

    steps = cast(Sequence[str], values["steps"])
    if steps:
        parts.append("理解/解题步骤：" + "；".join(steps))

    parts.append(f"难度：{_difficulty_to_zh(values['difficulty'])}")
    return "\n".join(parts)


def build_answer_context(values: Mapping[str, object]) -> str:
    """按旧构建脚本的字节级规则生成回答上下文。"""
    lines = [
        f"【{values['title']}】",
        f"类别：{values['category']}",
        f"难度：{_difficulty_to_zh(values['difficulty'])}",
        cast(str, values["content"]),
    ]

    example = cast(str, values["example"])
    if example:
        lines.append(f"示例：{example}")

    steps = cast(Sequence[str], values["steps"])
    if steps:
        lines.append("参考步骤：")
        lines.extend(steps)

    return "\n".join(lines)
