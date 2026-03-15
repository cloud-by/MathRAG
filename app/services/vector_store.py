from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np


def to_float32_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    if not vectors:
        raise ValueError("向量列表为空，无法构建或搜索索引")

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError(f"向量矩阵维度不正确，期望二维，实际为 {matrix.ndim} 维")
    if matrix.shape[1] == 0:
        raise ValueError("向量维度不能为 0")
    return matrix


def build_faiss_index(
    vectors: Sequence[Sequence[float]],
    use_inner_product: bool = True,
) -> faiss.Index:
    matrix = to_float32_matrix(vectors)
    dim = matrix.shape[1]

    if use_inner_product:
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(matrix)  # type: ignore[call-arg]
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> faiss.Index:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 FAISS 索引文件：{path}")
    return faiss.read_index(str(path))


def search_index(
    index: faiss.Index,
    query_vector: Sequence[float],
    top_k: int = 3,
) -> Tuple[List[float], List[int]]:
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    if query.ndim != 2:
        raise ValueError("query_vector 无法转换为二维查询向量")

    distances, indices = index.search(query, top_k)  # type: ignore[call-arg]
    return distances[0].tolist(), indices[0].tolist()


def save_id_map(id_map: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def load_id_map(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 id_map 文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("id_map.json 格式错误，期望为 list")
    return data


def build_id_map_from_chunks(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    id_map: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        id_map.append(
            {
                "index": idx,
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "source_id": chunk.get("source_id", ""),
                "title": chunk.get("title", ""),
                "category": chunk.get("category", ""),
                "metadata": chunk.get("metadata", {}),
            }
        )
    return id_map


def resolve_search_results(
    distances: Sequence[float],
    indices: Sequence[int],
    id_map: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for score, idx in zip(distances, indices):
        if idx < 0 or idx >= len(id_map):
            continue
        item = dict(id_map[idx])
        item["score"] = float(score)
        results.append(item)
    return results
