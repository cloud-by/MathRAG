from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.core.config import settings
from app.services.embedding_service import embed_texts
from app.services.vector_store import (
    build_faiss_index,
    build_id_map_from_chunks,
    save_id_map,
    save_index,
)


DEFAULT_INPUT = settings.PROCESSED_KB_PATH
DEFAULT_INDEX = settings.FAISS_INDEX_PATH
DEFAULT_ID_MAP = settings.ID_MAP_PATH


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败：{exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"第 {line_no} 行不是 JSON 对象")
            yield item


def prepare_texts(chunks: List[Dict[str, Any]]) -> List[str]:
    texts: List[str] = []
    for idx, chunk in enumerate(chunks):
        retrieval_text = str(chunk.get("retrieval_text", "")).strip()
        if not retrieval_text:
            raise ValueError(f"第 {idx} 条 chunk 缺少 retrieval_text")
        texts.append(retrieval_text)
    return texts


def build_index(
    input_path: Path,
    index_path: Path,
    id_map_path: Path,
    use_inner_product: bool = True,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到处理后的知识文件：{input_path}")

    chunks = list(load_jsonl(input_path))
    if not chunks:
        raise ValueError("处理后的知识文件为空，无法构建索引")

    texts = prepare_texts(chunks)
    print(f"已加载 {len(chunks)} 条 chunk，开始调用 embedding 接口……")
    vectors = embed_texts(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"embedding 返回数量与 chunk 数量不一致：vectors={len(vectors)}, chunks={len(chunks)}"
        )

    print("embedding 完成，开始构建 FAISS 索引……")
    index = build_faiss_index(vectors, use_inner_product=use_inner_product)
    id_map = build_id_map_from_chunks(chunks)

    save_index(index, index_path)
    save_id_map(id_map, id_map_path)

    print("索引构建完成。")
    print(f"FAISS 索引文件：{index_path}")
    print(f"ID 映射文件：{id_map_path}")
    print(f"向量条目数：{index.ntotal}")
    print(f"向量维度：{index.d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据处理后的知识块文件构建 FAISS 索引")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 JSONL 文件路径，默认 data/processed/kb_chunks.jsonl",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=DEFAULT_INDEX,
        help="输出 FAISS 索引路径，默认 data/index/faiss.index",
    )
    parser.add_argument(
        "--id-map-output",
        type=Path,
        default=DEFAULT_ID_MAP,
        help="输出 id_map 路径，默认 data/index/id_map.json",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ip",
        choices=["ip", "l2"],
        help="索引度量方式：ip 表示内积，l2 表示欧氏距离",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(
        input_path=args.input,
        index_path=args.index_output,
        id_map_path=args.id_map_output,
        use_inner_product=(args.metric == "ip"),
    )


if __name__ == "__main__":
    main()
