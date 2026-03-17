from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.services.embedding_service import embed_query
from app.services.vector_store import load_id_map, load_index, search_index


@dataclass
class RetrieverConfig:
    index_path: Path = settings.FAISS_INDEX_PATH
    id_map_path: Path = settings.ID_MAP_PATH
    processed_kb_path: Path = settings.PROCESSED_KB_PATH
    default_top_k: int = settings.TOP_K


class Retriever:
    """基于 FAISS + embedding 的最小检索器（扩展 schema 版本）。"""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self.index = load_index(self.config.index_path)
        self.id_map = load_id_map(self.config.id_map_path)
        self.chunk_map = self._load_chunk_map(self.config.processed_kb_path)

        if self.index.ntotal != len(self.id_map):
            raise ValueError(
                f"FAISS 索引条目数与 id_map 数量不一致："
                f"index={self.index.ntotal}, id_map={len(self.id_map)}"
            )

    @staticmethod
    def _load_chunk_map(path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"找不到处理后的知识文件：{path}")

        chunk_map: Dict[str, Dict[str, Any]] = {}
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
                if chunk_id in chunk_map:
                    raise ValueError(f"检测到重复 chunk_id：{chunk_id}（第 {line_no} 行）")

                chunk_map[chunk_id] = item

        return chunk_map

    def _get_id_map_item(self, index: int) -> Dict[str, Any]:
        item = self.id_map.get(str(index))

        if item is None:
            raise KeyError(f"id_map 中不存在索引 {index} 对应的条目")

        if not isinstance(item, dict):
            raise ValueError(f"id_map 中索引 {index} 对应的值不是对象")

        return item

    @staticmethod
    def _to_row(values: Any) -> List[Any]:
        if hasattr(values, "tolist"):
            values = values.tolist()

        if isinstance(values, list):
            if values and isinstance(values[0], list):
                return values[0]
            return values

        return list(values)

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        k = top_k or self.config.default_top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")

        k = min(k, self.index.ntotal)

        query_vector = embed_query(question)
        distances, indices = search_index(self.index, query_vector, top_k=k)

        distance_row = self._to_row(distances)
        index_row = self._to_row(indices)

        detailed_results: List[Dict[str, Any]] = []

        for rank, (score, faiss_index) in enumerate(zip(distance_row, index_row), start=1):
            faiss_index = int(faiss_index)
            if faiss_index < 0:
                continue

            id_item = self._get_id_map_item(faiss_index)

            chunk_id = str(id_item.get("chunk_id", "")).strip()
            chunk = self.chunk_map.get(chunk_id, {})

            result: Dict[str, Any] = {
                "rank": rank,
                "score": float(score),
                "index": faiss_index,
                "chunk_id": chunk_id,
                "source_id": id_item.get("source_id") or chunk.get("source_id", ""),
                "category": id_item.get("category") or chunk.get("category", ""),
                "stage": id_item.get("stage") or chunk.get("stage", ""),
                "course": id_item.get("course") or chunk.get("course", ""),
                "title": id_item.get("title") or chunk.get("title", ""),
                "keywords": id_item.get("keywords") or chunk.get("keywords", []),
                "content": id_item.get("content") or chunk.get("content", ""),
                "example": id_item.get("example") or chunk.get("example", ""),
                "steps": id_item.get("steps") or chunk.get("steps", []),
                "prerequisites": id_item.get("prerequisites") or chunk.get("prerequisites", []),
                "difficulty": id_item.get("difficulty") or chunk.get("difficulty", ""),
                "answer_context": id_item.get("answer_context") or chunk.get("answer_context", ""),
                "retrieval_text": chunk.get("retrieval_text", ""),
                "source_line": id_item.get("source_line", chunk.get("source_line")),
                "metadata": id_item.get("metadata") or chunk.get("metadata", {}),
            }
            detailed_results.append(result)

        return detailed_results

    def retrieve_by_query(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        return self.retrieve(question=question, top_k=top_k)


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    return get_retriever().retrieve(question=question, top_k=top_k)