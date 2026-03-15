from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rag_pipeline import chat_with_rag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简单验证 MathRAG 的 RAG 问答链路")
    parser.add_argument("--question", type=str, required=True, help="要测试的问题")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数")
    parser.add_argument("--show-references", action="store_true", help="是否打印参考知识")
    parser.add_argument("--show-full-json", action="store_true", help="是否直接打印完整 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = chat_with_rag(
        question=args.question,
        history=[],
        top_k=args.top_k,
    )

    if args.show_full_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"问题：{result['question']}")
    print()
    print("回答：")
    print(result["answer"])
    print()

    steps = result.get("steps", [])
    if steps:
        print("步骤：")
        for i, step in enumerate(steps, start=1):
            print(f"  {i}. {step}")
        print()

    used_knowledge = result.get("used_knowledge", [])
    if used_knowledge:
        print("使用到的知识点：")
        for item in used_knowledge:
            print(f"  - {item}")
        print()

    related_questions = result.get("related_questions", [])
    if related_questions:
        print("推荐追问：")
        for item in related_questions:
            print(f"  - {item}")
        print()

    if args.show_references:
        refs = result.get("references", [])
        print(f"参考知识（{len(refs)} 条）：")
        for i, ref in enumerate(refs, start=1):
            print(f"[{i}] {ref.get('title', '')} | score={ref.get('score', 0):.6f}")
            print(f"    category: {ref.get('category', '')}")
            print(f"    content : {ref.get('content', '')}")
            example = ref.get('example', '')
            if example:
                print(f"    example : {example}")
            print()


if __name__ == "__main__":
    main()
