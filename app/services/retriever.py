from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.core.config import settings
from app.services.embedding_service import embed_query
from app.services.vector_store import load_id_map, load_index, resolve_search_results, search_index


@dataclass
class RetrieverConfig:
    index_path: Path = settings.FAISS_INDEX_PATH
    id_map_path: Path = settings.ID_MAP_PATH
    processed_kb_path: Path = settings.PROCESSED_KB_PATH
    default_top_k: int = settings.TOP_K


class Retriever:
    """基于 FAISS + embedding 的最小检索器。"""

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self.config = config or RetrieverConfig()
        self.index = load_index(self.config.index_path)
        self.id_map = load_id_map(self.config.id_map_path)
        self.chunk_map = self._load_chunk_map(self.config.processed_kb_path)

        if self.index.ntotal != len(self.id_map):
            raise ValueError(
                f"FAISS 索引条目数与 id_map 数量不一致：index={self.index.ntotal}, id_map={len(self.id_map)}"
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
                chunk_map[chunk_id] = item
        return chunk_map

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        question = str(question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        k = top_k or self.config.default_top_k
        if k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_vector = embed_query(question)
        distances, indices = search_index(self.index, query_vector, top_k=k)
        brief_results = resolve_search_results(distances, indices, self.id_map)

        detailed_results: List[Dict[str, Any]] = []
        for rank, item in enumerate(brief_results, start=1):
            chunk_id = item.get("chunk_id", "")
            chunk = self.chunk_map.get(chunk_id, {})

            result: Dict[str, Any] = {
                "rank": rank,
                "score": float(item.get("score", 0.0)),
                "index": item.get("index"),
                "chunk_id": chunk_id,
                "source_id": item.get("source_id", ""),
                "title": item.get("title") or chunk.get("title", ""),
                "category": item.get("category") or chunk.get("category", ""),
                "keywords": chunk.get("keywords", []),
                "content": chunk.get("content", ""),
                "example": chunk.get("example", ""),
                "steps": chunk.get("steps", []),
                "answer_context": chunk.get("answer_context", ""),
                "retrieval_text": chunk.get("retrieval_text", ""),
                "metadata": chunk.get("metadata", {}),
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
