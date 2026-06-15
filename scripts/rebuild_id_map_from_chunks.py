from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "kb_chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "index" / "id_map.json"


ID_MAP_FIELDS = [
    "chunk_id",
    "source_id",
    "category",
    "title",
    "keywords",
    "content",
    "example",
    "steps",
    "difficulty",
    "answer_context",
    "source_line",
    "metadata",
]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc}") from exc

            if not isinstance(item, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")

            chunk_id = str(item.get("chunk_id", "")).strip()
            if not chunk_id:
                raise ValueError(f"第 {line_no} 行缺少 chunk_id")
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"检测到重复 chunk_id：{chunk_id}（第 {line_no} 行）")
            seen_chunk_ids.add(chunk_id)
            rows.append(item)

    return rows


def build_id_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    id_map: dict[str, dict[str, Any]] = {}

    for idx, chunk in enumerate(chunks):
        item: dict[str, Any] = {}
        for field_name in ID_MAP_FIELDS:
            if field_name == "keywords":
                item[field_name] = chunk.get(field_name, [])
            elif field_name == "steps":
                item[field_name] = chunk.get(field_name, [])
            elif field_name == "metadata":
                metadata = chunk.get(field_name, {})
                item[field_name] = metadata if isinstance(metadata, dict) else {}
            elif field_name == "source_line":
                item[field_name] = chunk.get(field_name)
            else:
                item[field_name] = str(chunk.get(field_name, "")).strip()
        id_map[str(idx)] = item

    return id_map


def create_backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak_latex_{timestamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild data/index/id_map.json from data/processed/kb_chunks.jsonl without recalculating embeddings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input kb_chunks JSONL path. Defaults to data/processed/kb_chunks.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output id_map JSON path. Defaults to data/index/id_map.json.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before overwriting id_map.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    chunks = load_chunks(input_path)
    id_map = build_id_map(chunks)

    backup_path = None if args.no_backup else create_backup(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print(f"chunks: {len(chunks)}")
    print(f"id_map_entries: {len(id_map)}")
    print(f"output: {output_path}")
    if backup_path is not None:
        print(f"backup: {backup_path}")


if __name__ == "__main__":
    main()
