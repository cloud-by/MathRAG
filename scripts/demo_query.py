from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.services.retriever import retrieve


def truncate_text(text: str, max_len: int = 160) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_result(item: Dict[str, Any], show_context: bool = False) -> None:
    print(f"\n[{item['rank']}] {item['title']}")
    print(f"  category : {item['category']}")
    print(f"  chunk_id  : {item['chunk_id']}")
    print(f"  source_id : {item['source_id']}")
    print(f"  score     : {item['score']:.6f}")

    keywords = item.get("keywords", []) or []
    if keywords:
        print(f"  keywords  : {', '.join(keywords)}")

    content = item.get("content", "")
    if content:
        print(f"  content   : {truncate_text(content, 180)}")

    example = item.get("example", "")
    if example:
        print(f"  example   : {truncate_text(example, 160)}")

    steps = item.get("steps", []) or []
    if steps:
        print("  steps     :")
        for step in steps[:4]:
            print(f"    - {step}")
        if len(steps) > 4:
            print(f"    - ... 共 {len(steps)} 步")

    if show_context:
        answer_context = item.get("answer_context", "")
        if answer_context:
            print("  answer_context:")
            for line in answer_context.splitlines():
                print(f"    {line}")


def run_once(question: str, top_k: int, show_context: bool) -> None:
    print(f"\n检索问题：{question}")
    print(f"top_k={top_k}\n")

    results = retrieve(question=question, top_k=top_k)
    if not results:
        print("没有检索到结果。")
        return

    print(f"共返回 {len(results)} 条结果：")
    for item in results:
        print_result(item, show_context=show_context)


def interactive_loop(top_k: int, show_context: bool) -> None:
    print("进入交互检索模式。直接输入数学问题，输入 exit / quit 退出。")
    while True:
        question = input("\n请输入问题> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("已退出检索演示。")
            break
        if not question:
            print("问题不能为空。")
            continue
        try:
            run_once(question=question, top_k=top_k, show_context=show_context)
        except Exception as exc:  # noqa: BLE001
            print(f"检索失败：{exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="命令行检索验证脚本")
    parser.add_argument("--question", type=str, default="", help="单次检索的问题文本")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K, help="返回前 k 条结果")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="是否打印 answer_context，便于后续接大模型前做人工检查",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互模式，多次输入问题进行检索验证",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interactive:
        interactive_loop(top_k=args.top_k, show_context=args.show_context)
        return

    question = args.question.strip()
    if not question:
        question = "x^2+4x+3=0 怎么解？"

    run_once(question=question, top_k=args.top_k, show_context=args.show_context)


if __name__ == "__main__":
    main()
