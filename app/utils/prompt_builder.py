from __future__ import annotations

from typing import Any, Sequence

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


SYSTEM_PROMPT = """你是一个严谨、清晰、尽量循序渐进的数学助教，服务对象覆盖小学、初中、高中到大学基础学习者。
你必须优先依据提供的参考知识回答问题，不要编造不存在的定义、公式、定理、例题或结论。
如果参考知识不足以支撑确定答案，要明确说明“参考知识不足”，并给出当前最稳妥、最保守的解释。

你会看到参考知识中的以下结构化信息：
- 学段（stage）
- 课程（course）
- 类别（category）
- 标题（title）
- 关键词（keywords）
- 内容（content / answer_context）
- 示例（example）
- 步骤（steps）
- 前置知识（prerequisites）
- 难度（difficulty）

使用原则：
1. 优先使用与当前问题最直接相关的参考知识。
2. 如果存在多个参考知识，优先综合标题、课程、学段、内容来判断相关性。
3. 如果参考知识来自不同学段，优先采用与当前问题深度最匹配的解释，不要把大学层面的抽象表述硬塞给小学或初中风格的问题，也不要把过于初级的解释强行用于明显的大学题目。
4. 如果问题是求解题，answer 先给结论、结果或解法判断，再用 steps 按顺序解释。
5. 如果问题是概念题，answer 先给定义或核心解释，再用 steps 补充理解要点、判断方法或应用方式。
6. used_knowledge 必须优先填写本次实际使用到的参考知识标题。
7. related_questions 需要与当前问题直接相关，适合作为下一步追问。

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
2. steps 必须是字符串数组，通常给 2 到 6 条；如果问题非常简单，也至少给 2 条。
3. used_knowledge 必须优先引用提供的参考知识标题，不要虚构知识点标题。
4. related_questions 给 2 条即可，要求和当前问题紧密相关。
5. 不要在输出中包含“json如下”“```json```”之类多余内容。
6. 输出必须是单个、可解析的 json 对象。
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

        if role == "assistant":
            speaker = "助手"
        elif role == "system":
            speaker = "系统"
        else:
            speaker = "用户"

        lines.append(f"{speaker}：{content}")

    return "\n".join(lines) if lines else "无"


def _stage_to_zh(stage: str) -> str:
    mapping = {
        "primary": "小学",
        "junior_secondary": "初中",
        "senior_secondary": "高中",
        "undergraduate": "大学",
    }
    return mapping.get(stage, stage or "未标注")


def _difficulty_to_zh(difficulty: str) -> str:
    mapping = {
        "easy": "简单",
        "medium": "中等",
        "hard": "困难",
    }
    return mapping.get(difficulty, difficulty or "未标注")


def _normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _format_references(references: Sequence[dict[str, Any]]) -> str:
    if not references:
        return "无"

    blocks: list[str] = []

    for idx, ref in enumerate(references, start=1):
        title = str(ref.get("title", "")).strip()
        category = str(ref.get("category", "")).strip()
        stage = _stage_to_zh(str(ref.get("stage", "")).strip())
        course = str(ref.get("course", "")).strip()
        difficulty = _difficulty_to_zh(str(ref.get("difficulty", "")).strip())
        keywords = _normalize_str_list(ref.get("keywords", []))
        example = str(ref.get("example", "")).strip()
        steps = _normalize_str_list(ref.get("steps", []))
        prerequisites = _normalize_str_list(ref.get("prerequisites", []))
        answer_context = str(ref.get("answer_context", "")).strip()
        content = str(ref.get("content", "")).strip()
        score = ref.get("score", None)
        source_id = str(ref.get("source_id", "")).strip()

        lines = [f"[参考知识 {idx}] 标题：{title or '未命名知识点'}"]

        if source_id:
            lines.append(f"知识点ID：{source_id}")
        if stage:
            lines.append(f"学段：{stage}")
        if course:
            lines.append(f"课程：{course}")
        if category:
            lines.append(f"类别：{category}")
        if difficulty:
            lines.append(f"难度：{difficulty}")
        if score is not None:
            try:
                lines.append(f"检索分数：{float(score):.6f}")
            except Exception:
                lines.append(f"检索分数：{score}")

        if keywords:
            lines.append("关键词：" + "，".join(keywords))

        if prerequisites:
            lines.append("前置知识：" + "，".join(prerequisites))

        if answer_context:
            lines.append("知识内容：\n" + answer_context)
        elif content:
            lines.append("知识内容：\n" + content)

        if example:
            lines.append(f"额外示例：{example}")

        if steps:
            lines.append("可参考步骤：" + "；".join(steps))

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

请特别注意：
1. 参考知识带有“学段、课程、难度、前置知识”等结构化信息，你需要利用这些信息判断解释深度。
2. 如果问题更像基础题，优先使用更基础、更直接的解释；如果问题明显属于高等数学、线性代数、概率统计等大学课程，则优先采用对应课程的知识。
3. 如果多个参考知识都相关，可以综合使用，但 used_knowledge 里只写真正用到的标题。
4. 如果参考知识只能支持部分回答，请在 answer 中明确写出“参考知识不足以完全确定答案”。

输出要求：
1. 如果当前问题是解题题目，answer 先给出结果、判断或解法方向，再在 steps 中按顺序解释。
2. 如果当前问题是概念题，answer 先给出定义或核心解释，再在 steps 中补充理解要点、判断方法或典型应用。
3. used_knowledge 只写本次真正用到的知识点标题，不要乱写。
4. related_questions 给 2 条，必须与当前问题直接相关，适合作为下一步追问。
5. 输出必须是单个 json 对象。
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