from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from app.core.config import settings
from app.schemas.knowledge import KnowledgeRecord
from app.services.llm_service import chat_json


DEFAULT_KNOWLEDGE_PATH = settings.RAW_KB_PATH
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


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


@dataclass(frozen=True)
class KnowledgeDraft:
    """不含 JSONL 遗留 id 的知识抽取纯数据。"""

    category: str
    title: str
    keywords: tuple[str, ...]
    content: str
    example: str
    steps: tuple[str, ...]
    difficulty: str

    def to_values(self) -> dict[str, object]:
        """返回可直接用于 PostgreSQL 知识条目的字段副本。"""
        return {
            "category": self.category,
            "title": self.title,
            "keywords": list(self.keywords),
            "content": self.content,
            "example": self.example,
            "steps": list(self.steps),
            "difficulty": self.difficulty,
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
    category: str | None = None,
) -> List[Dict[str, str]]:
    hints = {
        "category": category,
    }

    return [
        {
            "role": "system",
            "content": (
                "你是数学教材知识库编辑。请从教材片段中抽取可以独立检索和用于答疑的知识点，"
                "并只返回 JSON 对象。不要输出 markdown。"
                "涉及数学公式时，必须使用 KaTeX 可渲染的 LaTeX：行内公式用 \\( ... \\)，"
                "块级公式用 \\[ ... \\]，不要新增 $...$ 或 $$...$$。"
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
                '      "title": "知识点标题",\n'
                '      "keywords": ["关键词1", "关键词2"],\n'
                '      "content": "用完整句子解释核心知识点",\n'
                '      "example": "来自片段或自造的简短例子，没有则为空字符串",\n'
                '      "steps": ["理解或解题步骤1", "理解或解题步骤2"],\n'
                '      "difficulty": "easy|medium|hard"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "要求：\n"
                "1. 每个 item 只表达一个清晰知识点。\n"
                "2. content 不要照抄大段原文，要提炼成适合问答系统使用的解释。\n"
                "3. keywords 和 steps 不能为空。\n"
                "4. 如果提示信息中提供 category，优先沿用。\n"
                "5. content、example、steps 中如包含数学公式，必须统一为 KaTeX LaTeX 分隔符："
                "行内公式用 \\( ... \\)，块级公式用 \\[ ... \\]；不要新增 $...$ 或 $$...$$。\n"
                "6. 字符串字段内部不要包含原始换行，不要把公式逐字符、逐行拆开。\n"
                "7. 输出必须是可被 json.loads 解析的合法 JSON。\n\n"
                f"提示信息：{json.dumps(hints, ensure_ascii=False)}\n\n"
                f"教材片段：\n{text}"
            ),
        },
    ]


def _normalize_raw_item(raw: Dict[str, Any], item_id: str, hints: Dict[str, str | None]) -> KnowledgeRecord:
    record = {
        "id": item_id,
        "category": _normalize_category(raw.get("category") or hints.get("category")),
        "title": _normalize_text(raw.get("title")),
        "keywords": _normalize_list(raw.get("keywords")),
        "content": _normalize_text(raw.get("content")),
        "example": _normalize_text(raw.get("example")),
        "steps": _normalize_list(raw.get("steps")),
        "difficulty": _normalize_difficulty(raw.get("difficulty")),
    }
    return KnowledgeRecord(**record)


def _normalize_draft(raw: Mapping[str, Any], category: str | None) -> KnowledgeDraft:
    """复用旧 schema 校验规则，但不生成或读取任何 JSONL id。"""
    record = _normalize_raw_item(dict(raw), "k0000", {"category": category})
    return KnowledgeDraft(
        category=record.category,
        title=record.title,
        keywords=tuple(record.keywords),
        content=record.content,
        example=record.example,
        steps=tuple(record.steps),
        difficulty=record.difficulty,
    )


def normalize_drafts(
    data: Mapping[str, Any],
    category: str | None = None,
) -> List[KnowledgeDraft]:
    """把 LLM JSON 对象转换为与持久化介质无关的知识草稿。"""
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("model response must contain a non-empty items list")

    drafts = [
        _normalize_draft(raw, category)
        for raw in items
        if isinstance(raw, Mapping)
    ]
    if not drafts:
        raise ValueError("no valid knowledge records were extracted")
    return drafts


def extract_knowledge_drafts(
    text: str,
    category: str | None = None,
) -> List[KnowledgeDraft]:
    """从正文抽取知识草稿；在线导入不会访问 seed JSONL。"""
    normalized = _normalize_text(text)
    if not normalized:
        raise ValueError("text cannot be empty")
    result = chat_json(
        messages=_build_messages(normalized, category),
        temperature=0.1,
    )
    return normalize_drafts(result.data, category)


def extract_knowledge_records(
    text: str,
    category: str | None = None,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
) -> List[KnowledgeRecord]:
    drafts = extract_knowledge_drafts(text, category)
    next_ids = generate_next_ids(len(drafts), knowledge_path)
    return [
        KnowledgeRecord(id=item_id, **draft.to_values())
        for item_id, draft in zip(next_ids, drafts, strict=True)
    ]


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
