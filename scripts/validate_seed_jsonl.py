from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from app.core.config import settings
from app.schemas.knowledge import KnowledgeRecord, SEED_FIELD_ORDER


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_error(**kwargs: Any) -> Dict[str, Any]:
    return {"timestamp": utc_timestamp(), **kwargs}


def chinese_ratio(text: str) -> float:
    meaningful = re.findall(r"[\u4e00-\u9fffA-Za-z]", text)
    if not meaningful:
        return 0.0
    chinese_count = sum(1 for char in meaningful if "\u4e00" <= char <= "\u9fff")
    return chinese_count / len(meaningful)


def record_chinese_ratio(record: KnowledgeRecord) -> float:
    text = " ".join(
        [
            record.category,
            record.course,
            record.title,
            " ".join(record.keywords),
            record.content,
            record.example,
            " ".join(record.steps),
            " ".join(record.prerequisites),
        ]
    )
    return chinese_ratio(text)


def validate_seed_file(path: Path) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(build_error(line=line_no, error=f"invalid JSON: {exc}"))
                continue

            if not isinstance(item, dict):
                errors.append(build_error(line=line_no, error="line is not a JSON object"))
                continue

            keys = list(item.keys())
            if keys != SEED_FIELD_ORDER:
                errors.append(
                    build_error(
                        line=line_no,
                        error="field set or field order does not match MathRAG seed JSONL",
                        keys=keys,
                    )
                )
                continue

            try:
                record = KnowledgeRecord(**item)
            except ValidationError as exc:
                errors.append(build_error(line=line_no, error=exc.errors()))
                continue

            if record.id in seen_ids:
                errors.append(build_error(line=line_no, error=f"duplicate id: {record.id}"))
            seen_ids.add(record.id)

            ratio = record_chinese_ratio(record)
            if ratio < 0.35:
                errors.append(
                    build_error(
                        line=line_no,
                        id=record.id,
                        title=record.title,
                        error=f"record content should be Chinese; chinese_ratio={ratio:.2f}",
                    )
                )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MathRAG raw seed JSONL format.")
    parser.add_argument("--input", type=Path, default=settings.RAW_KB_PATH)
    parser.add_argument("--error-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = validate_seed_file(args.input)

    if args.error_output and errors:
        args.error_output.parent.mkdir(parents=True, exist_ok=True)
        with args.error_output.open("w", encoding="utf-8") as file:
            for error in errors:
                file.write(json.dumps(error, ensure_ascii=False) + "\n")

    if errors:
        print(f"Invalid seed JSONL: {len(errors)} error(s)")
        for error in errors[:20]:
            print(json.dumps(error, ensure_ascii=False))
        raise SystemExit(1)

    print("Seed JSONL is valid.")


if __name__ == "__main__":
    main()
