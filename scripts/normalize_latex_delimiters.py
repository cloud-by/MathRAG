from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "math_knowledge_seed.jsonl"

SEED_FIELD_ORDER = [
    "id",
    "category",
    "title",
    "keywords",
    "content",
    "example",
    "steps",
    "difficulty",
]

TEXT_FIELDS = {
    "title",
    "content",
    "answer_context",
    "example",
}

LIST_TEXT_FIELDS = {
    "keywords",
    "steps",
}


BLOCK_DOLLAR_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", flags=re.DOTALL)
INLINE_DOLLAR_RE = re.compile(
    r"(?<!\\)(?<!\$)\$(?!\$)(.+?)(?<!\\)(?<!\$)\$(?!\$)",
    flags=re.DOTALL,
)


@dataclass
class NormalizeStats:
    records_total: int = 0
    records_changed: int = 0
    fields_changed: int = 0
    block_formulas_changed: int = 0
    inline_formulas_changed: int = 0
    examples: list[str] = field(default_factory=list)


def normalize_latex_delimiters(text: str) -> tuple[str, int, int]:
    """Convert $...$ / $$...$$ delimiters to \\(...\\) / \\[...\\]."""

    block_count = 0
    inline_count = 0

    def replace_block(match: re.Match[str]) -> str:
        nonlocal block_count
        block_count += 1
        body = match.group(1).strip()
        return f"\\[{body}\\]"

    def replace_inline(match: re.Match[str]) -> str:
        nonlocal inline_count
        inline_count += 1
        body = match.group(1).strip()
        return f"\\({body}\\)"

    normalized = BLOCK_DOLLAR_RE.sub(replace_block, text)
    normalized = INLINE_DOLLAR_RE.sub(replace_inline, normalized)
    return normalized, block_count, inline_count


def normalize_value(value: Any) -> tuple[Any, int, int, bool]:
    if isinstance(value, str):
        normalized, block_count, inline_count = normalize_latex_delimiters(value)
        return normalized, block_count, inline_count, normalized != value

    if isinstance(value, list):
        changed = False
        block_total = 0
        inline_total = 0
        normalized_items: list[Any] = []

        for item in value:
            if isinstance(item, str):
                normalized, block_count, inline_count = normalize_latex_delimiters(item)
                changed = changed or normalized != item
                block_total += block_count
                inline_total += inline_count
                normalized_items.append(normalized)
            else:
                normalized_items.append(item)

        return normalized_items, block_total, inline_total, changed

    return value, 0, 0, False


def ordered_seed_dict(data: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for field_name in SEED_FIELD_ORDER:
        if field_name in data:
            ordered[field_name] = data[field_name]

    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def normalize_record(data: dict[str, Any], line_no: int, stats: NormalizeStats) -> dict[str, Any]:
    normalized = dict(data)
    record_changed = False

    for field_name in sorted(TEXT_FIELDS | LIST_TEXT_FIELDS):
        if field_name not in normalized:
            continue

        old_value = normalized[field_name]
        new_value, block_count, inline_count, changed = normalize_value(old_value)

        if changed:
            normalized[field_name] = new_value
            record_changed = True
            stats.fields_changed += 1
            stats.block_formulas_changed += block_count
            stats.inline_formulas_changed += inline_count

            if len(stats.examples) < 8:
                before = str(old_value)
                after = str(new_value)
                stats.examples.append(
                    f"line {line_no}, field {field_name}: "
                    f"{before[:120].replace(chr(10), ' ')} -> "
                    f"{after[:120].replace(chr(10), ' ')}"
                )

    if record_changed:
        stats.records_changed += 1

    return ordered_seed_dict(normalized)


def load_and_normalize(input_path: Path) -> tuple[list[dict[str, Any]], NormalizeStats]:
    stats = NormalizeStats()
    rows: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
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

            stats.records_total += 1
            rows.append(normalize_record(data, line_no, stats))

    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_latex_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize MathRAG knowledge-base LaTeX delimiters: $...$ -> \\(...\\), $$...$$ -> \\[...\\]."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL path. Defaults to data/raw/math_knowledge_seed.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to input path when --write is used.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write normalized JSONL. Without this flag, only prints a dry-run summary.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup when overwriting the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    output_path = args.output.resolve() if args.output else input_path
    rows, stats = load_and_normalize(input_path)

    print("LaTeX delimiter normalization summary")
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"records_total: {stats.records_total}")
    print(f"records_changed: {stats.records_changed}")
    print(f"fields_changed: {stats.fields_changed}")
    print(f"block_formulas_changed: {stats.block_formulas_changed}")
    print(f"inline_formulas_changed: {stats.inline_formulas_changed}")

    if stats.examples:
        print("examples:")
        for example in stats.examples:
            print(f"  - {example}")

    if not args.write:
        print("dry-run only; add --write to update files.")
        return

    backup_path: Path | None = None
    if output_path == input_path and not args.no_backup:
        backup_path = create_backup(input_path)
        print(f"backup: {backup_path}")

    write_jsonl(output_path, rows)
    print("write complete.")


if __name__ == "__main__":
    main()
