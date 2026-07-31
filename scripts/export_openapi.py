from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "frontend" / "openapi.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def serialize_openapi_document(document: Mapping[str, Any]) -> str:
    return json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def export_openapi_document(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    # 延迟导入应用，便于单独测试序列化逻辑。
    from app.main import create_app

    document = create_app().openapi()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        serialize_openapi_document(document),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    export_openapi_document()
