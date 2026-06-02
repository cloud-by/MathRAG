from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.config import settings
from app.schemas.knowledge import KnowledgeRecord
from app.services.llm_service import chat_json


DEFAULT_KNOWLEDGE_PATH = settings.RAW_KB_PATH
VALID_STAGES = {"primary", "junior_secondary", "senior_secondary", "undergraduate"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


STAGE_ALIASES = {
    "小学": "primary",
    "primary": "primary",
    "初中": "junior_secondary",
    "junior": "junior_secondary",
    "junior_secondary": "junior_secondary",
    "高中": "senior_secondary",
    "senior": "senior_secondary",
    "senior_secondary": "senior_secondary",
    "大学": "undergraduate",
    "本科": "undergraduate",
    "undergraduate": "undergraduate",
}


DIFFICULTY_ALIASES = {
    "简单": "easy",
    "基础": "easy",
    "easy": "easy",
    "中等": "medium",
    "一般": "medium",
    "medium": "medium",
    "困难": "hard",
    "较难": "hard",
    "hard": "hard",
}


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = _normalize_text(value)
        if not text:
            return []
        items = re.split(r"[;；、,\n]+", text)

    output: List[str] = []
    seen = set()
    for item in items:
        text = _normalize_text(item)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _normalize_category(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return "concept"
    if re.search(r"^[A-Za-z0-9_\-\s]+$", text):
        text = text.lower()
        text = re.sub(r"[\s\-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
    return text or "concept"


def _normalize_stage(value: Any, default: str | None = None) -> str:
    text = _normalize_text(value or default).lower().replace("-", "_").replace(" ", "_")
    text = STAGE_ALIASES.get(text, text)
    return text if text in VALID_STAGES else "junior_secondary"


def _normalize_difficulty(value: Any) -> str:
    text = _normalize_text(value).lower().replace("-", "_").replace(" ", "_")
    text = DIFFICULTY_ALIASES.get(text, text)
    return text if text in VALID_DIFFICULTIES else "medium"


def _load_existing_ids(path: Path) -> List[str]:
    if not path.exists():
        return []

    ids: List[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                item_id = _normalize_text(item.get("id"))
                if item_id:
                    ids.append(item_id)
    return ids


def generate_next_ids(count: int, path: Path = DEFAULT_KNOWLEDGE_PATH) -> List[str]:
    existing_ids = _load_existing_ids(path)
    max_number = 0
    for item_id in existing_ids:
        match = re.fullmatch(r"k(\d+)", item_id)
        if match:
            max_number = max(max_number, int(match.group(1)))

    width = max(4, len(str(max_number + count)))
    return [f"k{number:0{width}d}" for number in range(max_number + 1, max_number + count + 1)]


def _build_messages(
    text: str,
    stage: str | None = None,
    course: str | None = None,
    category: str | None = None,
) -> List[Dict[str, str]]:
    hints = {
        "stage": stage,
        "course": course,
        "category": category,
    }

    return [
        {
            "role": "system",
            "content": (
                "你是数学教材知识库编辑。请从教材片段中抽取可以独立检索和用于答疑的知识点，"
                "并只返回 JSON 对象。不要输出 markdown。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请把下面教材片段整理为知识库条目。\n"
                "输出格式必须是：\n"
                "{\n"
                '  "items": [\n'
                "    {\n"
                '      "category": "知识分类，优先用简短中文或 snake_case",\n'
                '      "stage": "primary|junior_secondary|senior_secondary|undergraduate",\n'
                '      "course": "课程名",\n'
                '      "title": "知识点标题",\n'
                '      "keywords": ["关键词1", "关键词2"],\n'
                '      "content": "用完整句子解释核心知识点",\n'
                '      "example": "来自片段或自造的简短例子，没有则为空字符串",\n'
                '      "steps": ["理解或解题步骤1", "理解或解题步骤2"],\n'
                '      "prerequisites": ["前置知识1"],\n'
                '      "difficulty": "easy|medium|hard"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "要求：\n"
                "1. 每个 item 只表达一个清晰知识点。\n"
                "2. content 不要照抄大段原文，要提炼成适合问答系统使用的解释。\n"
                "3. keywords 和 steps 不能为空。\n"
                "4. 如果提示信息中提供 stage/course/category，优先沿用。\n\n"
                f"提示信息：{json.dumps(hints, ensure_ascii=False)}\n\n"
                f"教材片段：\n{text}"
            ),
        },
    ]


def _normalize_raw_item(raw: Dict[str, Any], item_id: str, hints: Dict[str, str | None]) -> KnowledgeRecord:
    record = {
        "id": item_id,
        "category": _normalize_category(raw.get("category") or hints.get("category")),
        "stage": _normalize_stage(raw.get("stage"), hints.get("stage")),
        "course": _normalize_text(raw.get("course") or hints.get("course")),
        "title": _normalize_text(raw.get("title")),
        "keywords": _normalize_list(raw.get("keywords")),
        "content": _normalize_text(raw.get("content")),
        "example": _normalize_text(raw.get("example")),
        "steps": _normalize_list(raw.get("steps")),
        "prerequisites": _normalize_list(raw.get("prerequisites")),
        "difficulty": _normalize_difficulty(raw.get("difficulty")),
    }
    return KnowledgeRecord(**record)


def extract_knowledge_records(
    text: str,
    stage: str | None = None,
    course: str | None = None,
    category: str | None = None,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> List[KnowledgeRecord]:
    text = _normalize_text(text)
    if not text:
        raise ValueError("text cannot be empty")

    result = chat_json(
        messages=_build_messages(text=text, stage=stage, course=course, category=category),
        temperature=0.1,
    )
    items = result.data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("model response must contain a non-empty items list")

    next_ids = generate_next_ids(len(items), knowledge_path)
    hints = {"stage": stage, "course": course, "category": category}
    records: List[KnowledgeRecord] = []

    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        records.append(_normalize_raw_item(raw, next_ids[index], hints))

    if not records:
        raise ValueError("no valid knowledge records were extracted")

    return records


def append_records(records: Iterable[KnowledgeRecord], path: Path = DEFAULT_KNOWLEDGE_PATH) -> int:
    rows = list(records)
    if not rows:
        return 0

    existing_ids = set(_load_existing_ids(path))
    duplicate_ids = [record.id for record in rows if record.id in existing_ids]
    if duplicate_ids:
        raise ValueError(f"duplicate knowledge ids: {', '.join(duplicate_ids)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = path.exists() and path.stat().st_size > 0
    if needs_newline:
        with path.open("rb") as file:
            file.seek(-1, 2)
            needs_newline = file.read(1) != b"\n"

    with path.open("a", encoding="utf-8") as file:
        if needs_newline:
            file.write("\n")
        for record in rows:
            file.write(json.dumps(record.to_seed_dict(), ensure_ascii=False) + "\n")

    return len(rows)
