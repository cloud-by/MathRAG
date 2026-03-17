from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


def to_float32_query(query_vector: Sequence[float]) -> np.ndarray:
    query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
    if query.ndim != 2:
        raise ValueError("query_vector 无法转换为二维查询向量")
    if query.shape[1] == 0:
        raise ValueError("query_vector 维度不能为 0")
    return query


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

    if index.ntotal <= 0:
        raise ValueError("FAISS 索引为空，无法执行搜索")

    real_top_k = min(top_k, index.ntotal)
    query = to_float32_query(query_vector)

    distances, indices = index.search(query, real_top_k)  # type: ignore[call-arg]
    return distances[0].tolist(), indices[0].tolist()


def save_id_map(id_map: Mapping[str, Dict[str, Any]] | List[Dict[str, Any]], path: str | Path) -> None:
    """
    保存 id_map。
    推荐格式：dict[str, dict]，例如：
    {
        "0": {...},
        "1": {...}
    }

    为兼容旧逻辑，也允许传入 list[dict]。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def _convert_legacy_list_id_map(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    converted: Dict[str, Dict[str, Any]] = {}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"id_map.json 中第 {idx} 个元素不是对象")

        raw_index = item.get("index", idx)
        try:
            key = str(int(raw_index))
        except Exception as exc:
            raise ValueError(f"id_map.json 中第 {idx} 个元素的 index 非法：{raw_index}") from exc

        new_item = dict(item)
        new_item.pop("index", None)
        converted[key] = new_item

    return converted


def load_id_map(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    读取 id_map，并统一返回 dict[str, dict]。

    支持两种输入格式：
    1. 新版：{"0": {...}, "1": {...}}
    2. 旧版：[{"index": 0, ...}, {"index": 1, ...}]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 id_map 文件：{path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if not isinstance(value, dict):
                raise ValueError(f"id_map.json 中键 {key} 对应的值不是对象")
            normalized[str(key)] = value
        return normalized

    if isinstance(data, list):
        return _convert_legacy_list_id_map(data)

    raise ValueError("id_map.json 格式错误，期望为 dict 或 legacy list")


def build_id_map_from_chunks(chunks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    根据 chunk 构建较完整的 id_map。
    虽然你当前的 build_index.py 已经自己构建 rich id_map，
    这里仍保留一个通用版本，方便其它脚本复用。
    """
    id_map: Dict[str, Dict[str, Any]] = {}

    for idx, chunk in enumerate(chunks):
        id_map[str(idx)] = {
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
            "source_id": chunk.get("source_id", ""),
            "category": chunk.get("category", ""),
            "stage": chunk.get("stage", ""),
            "course": chunk.get("course", ""),
            "title": chunk.get("title", ""),
            "keywords": chunk.get("keywords", []),
            "content": chunk.get("content", ""),
            "example": chunk.get("example", ""),
            "steps": chunk.get("steps", []),
            "prerequisites": chunk.get("prerequisites", []),
            "difficulty": chunk.get("difficulty", ""),
            "answer_context": chunk.get("answer_context", ""),
            "source_line": chunk.get("source_line"),
            "metadata": chunk.get("metadata", {}),
        }

    return id_map


def resolve_search_results(
    distances: Sequence[float],
    indices: Sequence[int],
    id_map: Mapping[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    兼容函数：
    - 支持新版 dict[str, dict] id_map
    - 也支持旧版 list[dict] id_map
    """
    if isinstance(id_map, Mapping):
        normalized_id_map: Dict[str, Dict[str, Any]] = {
            str(k): v for k, v in id_map.items()
        }
    else:
        normalized_id_map = _convert_legacy_list_id_map(list(id_map))

    results: List[Dict[str, Any]] = []
    for score, idx in zip(distances, indices):
        idx = int(idx)
        if idx < 0:
            continue

        item = normalized_id_map.get(str(idx))
        if item is None:
            continue

        result = dict(item)
        result["index"] = idx
        result["score"] = float(score)
        results.append(result)

    return results