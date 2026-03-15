from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "math_knowledge_seed.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "kb_chunks.jsonl"


def normalize_text(text: Any) -> str:
    """基础文本清洗：去空白、统一换行、压缩多余空格。"""
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


VALID_CATEGORIES = {
    "concept",
    "number",
    "algebra",
    "expression",
    "equation",
    "linear_equation",
    "system_of_equations",
    "quadratic_equation",
    "inequality",
    "function",
    "geometry",
    "statistics",
    "probability",
    "trigonometry",
    "calculus",
}


CATEGORY_ALIASES = {
    "linear": "linear_equation",
    "quadratic": "quadratic_equation",
    "system": "system_of_equations",
}


REQUIRED_FIELDS = ["id", "category", "title", "content"]
OPTIONAL_LIST_FIELDS = ["keywords", "steps"]
OPTIONAL_TEXT_FIELDS = ["example"]


def normalize_category(category: Any) -> str:
    value = normalize_text(category).lower()
    if not value:
        return "concept"
    value = CATEGORY_ALIASES.get(value, value)
    return value if value in VALID_CATEGORIES else value



def normalize_list(value: Any) -> List[str]:
    """把关键词/步骤统一成字符串列表。"""
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        text = normalize_text(value)
        if not text:
            return []
        items = re.split(r"[;,；、\n]+", text)

    cleaned: List[str] = []
    seen = set()
    for item in items:
        item_text = normalize_text(item)
        if not item_text:
            continue
        if item_text not in seen:
            cleaned.append(item_text)
            seen.add(item_text)
    return cleaned



def ensure_step_prefix(step: str, index: int) -> str:
    if re.match(r"^(步骤\d+|Step\s*\d+|\d+[.、])", step, flags=re.IGNORECASE):
        return step
    return f"步骤{index}：{step}"



def validate_record(record: Dict[str, Any], line_no: int) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        if not normalize_text(record.get(field)):
            errors.append(f"缺少必填字段 {field}")
    return len(errors) == 0, errors



def build_retrieval_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"知识点标题：{item['title']}")
    parts.append(f"知识点类别：{item['category']}")

    if item["keywords"]:
        parts.append("关键词：" + "，".join(item["keywords"]))

    parts.append("核心内容：" + item["content"])

    if item["example"]:
        parts.append("例题示例：" + item["example"])

    if item["steps"]:
        parts.append("解题步骤：" + "；".join(item["steps"]))

    return "\n".join(parts)



def build_answer_context(item: Dict[str, Any]) -> str:
    lines: List[str] = [f"【{item['title']}】", item["content"]]

    if item["example"]:
        lines.append(f"示例：{item['example']}")

    if item["steps"]:
        lines.append("参考步骤：")
        lines.extend(item["steps"])

    return "\n".join(lines)



def normalize_record(raw: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    item["id"] = normalize_text(raw.get("id"))
    item["category"] = normalize_category(raw.get("category"))
    item["title"] = normalize_text(raw.get("title"))
    item["content"] = normalize_text(raw.get("content"))
    item["example"] = normalize_text(raw.get("example"))
    item["keywords"] = normalize_list(raw.get("keywords"))

    steps = normalize_list(raw.get("steps"))
    item["steps"] = [ensure_step_prefix(step, idx + 1) for idx, step in enumerate(steps)]

    valid, errors = validate_record(item, line_no)
    if not valid:
        raise ValueError(f"第 {line_no} 行数据不合法：{'；'.join(errors)}")

    item["source_line"] = line_no
    item["retrieval_text"] = build_retrieval_text(item)
    item["answer_context"] = build_answer_context(item)
    return item



def load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            yield line_no, data



def convert_to_chunk(record: Dict[str, Any]) -> Dict[str, Any]:
    """将标准化后的知识条目转成后续检索与索引脚本使用的 chunk。"""
    return {
        "chunk_id": f"{record['id']}_chunk_0",
        "source_id": record["id"],
        "category": record["category"],
        "title": record["title"],
        "keywords": record["keywords"],
        "content": record["content"],
        "example": record["example"],
        "steps": record["steps"],
        "source_line": record["source_line"],
        "retrieval_text": record["retrieval_text"],
        "answer_context": record["answer_context"],
        "metadata": {
            "source_file": "math_knowledge_seed.jsonl",
            "chunk_index": 0,
            "has_example": bool(record["example"]),
            "has_steps": bool(record["steps"]),
        },
    }



def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def build_kb(input_path: Path, output_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    rows: List[Dict[str, Any]] = []
    seen_ids = set()

    for line_no, raw in load_jsonl(input_path):
        record = normalize_record(raw, line_no)

        if record["id"] in seen_ids:
            raise ValueError(f"检测到重复 id：{record['id']}（第 {line_no} 行）")
        seen_ids.add(record["id"])

        chunk = convert_to_chunk(record)
        rows.append(chunk)

    write_jsonl(output_path, rows)
    return rows



def print_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    category_count: Dict[str, int] = {}
    with_example = 0
    with_steps = 0

    for row in rows:
        category = row["category"]
        category_count[category] = category_count.get(category, 0) + 1
        with_example += int(bool(row["example"]))
        with_steps += int(bool(row["steps"]))

    print(f"知识预处理完成，共生成 {len(rows)} 条 chunk。")
    print(f"输出文件：{output_path}")
    print(f"包含 example 的条目数：{with_example}")
    print(f"包含 steps 的条目数：{with_steps}")
    print("分类统计：")
    for category, count in sorted(category_count.items(), key=lambda x: x[0]):
        print(f"  - {category}: {count}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将数学种子知识库预处理为可检索的 chunk 文件")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 JSONL 文件路径，默认 data/raw/math_knowledge_seed.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出 JSONL 文件路径，默认 data/processed/kb_chunks.jsonl",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    rows = build_kb(args.input, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
