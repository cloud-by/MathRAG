from __future__ import annotations

from typing import Any, Sequence

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


SYSTEM_PROMPT = """你是一个严谨、清晰、面向中学生/大学基础学习者的数学助教。
你必须优先依据提供的参考知识回答问题，不要编造不存在的公式、定义或例题。
如果参考知识不足以支撑确定答案，要明确说明“参考知识不足”，并给出最稳妥的说明。

你必须输出 json 对象，不要输出 markdown，不要输出代码块，不要输出额外说明文字。
json 输出格式必须为：
{
  "answer": "简明结论或核心回答",
  "steps": ["步骤1", "步骤2", "步骤3"],
  "used_knowledge": ["使用到的知识点1", "使用到的知识点2"],
  "related_questions": ["追问1", "追问2"]
}

约束：
1. answer 必须是字符串。
2. steps 必须是字符串数组，通常给 2 到 6 条。
3. used_knowledge 必须优先引用提供的参考知识标题。
4. related_questions 给 2 条即可，要求和当前问题紧密相关。
5. 不要在输出中包含“json如下”“```json```”之类多余内容。
"""


def _format_history(history: Sequence[dict[str, Any]] | None) -> str:
    if not history:
        return "无"

    lines: list[str] = []
    for item in history[-6:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        speaker = "助手" if role == "assistant" else "用户"
        lines.append(f"{speaker}：{content}")

    return "\n".join(lines) if lines else "无"


def _format_references(references: Sequence[dict[str, Any]]) -> str:
    if not references:
        return "无"

    blocks: list[str] = []
    for idx, ref in enumerate(references, start=1):
        title = str(ref.get("title", "")).strip()
        category = str(ref.get("category", "")).strip()
        keywords = ref.get("keywords", []) or []
        example = str(ref.get("example", "")).strip()
        steps = ref.get("steps", []) or []
        answer_context = str(ref.get("answer_context", "")).strip()
        score = ref.get("score", None)

        lines = [f"[参考知识 {idx}] 标题：{title}"]
        if category:
            lines.append(f"类别：{category}")
        if score is not None:
            lines.append(f"检索分数：{float(score):.6f}")
        if keywords:
            lines.append("关键词：" + "，".join(map(str, keywords)))
        if answer_context:
            lines.append("知识内容：\n" + answer_context)
        if example:
            lines.append(f"额外示例：{example}")
        if steps:
            lines.append("可参考步骤：" + "；".join(map(str, steps)))
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    references: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]] | None = None,
) -> str:
    history_text = _format_history(history)
    reference_text = _format_references(references)

    return f"""请根据下面信息回答用户问题，并严格按 json 对象格式输出。

【当前问题】
{question}

【最近对话历史】
{history_text}

【参考知识】
{reference_text}

要求：
1. 如果当前问题是解题题目，answer 先给出结果或方法判断，再在 steps 中按顺序解释。
2. 如果当前问题是概念题，answer 先下定义，再在 steps 中补充理解要点或使用方法。
3. used_knowledge 只写本次真正用到的知识点标题，不要乱写。
4. 如果参考知识能直接支持答案，就优先用参考知识中的表述。
5. 如果参考知识不够，请在 answer 中明确写出“参考知识不足以完全确定答案”。
6. 输出必须是单个 json 对象。
"""


def build_chat_messages(
    question: str,
    references: Sequence[dict[str, Any]],
    history: Sequence[dict[str, Any]] | None = None,
) -> list[ChatCompletionMessageParam]:
    system_message: ChatCompletionSystemMessageParam = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": build_user_prompt(question=question, references=references, history=history),
    }
    return [system_message, user_message]
