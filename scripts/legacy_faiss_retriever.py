"""仅供离线对账使用的只读 legacy FAISS 适配器。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import faiss
import numpy as np


EMBEDDING_DIMENSIONS = 1024


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """解析 JSON 对象时拒绝重复键，避免后值静默覆盖前值。"""
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("id_map JSON 包含重复键")
        output[key] = value
    return output


class LegacyFaissRetriever:
    """从冻结的 FAISS/index map 读取 legacy source ID，不提供写接口。"""

    def __init__(self, *, index_path: Path, id_map_path: Path) -> None:
        index_file = Path(index_path)
        id_map_file = Path(id_map_path)
        if not index_file.is_file():
            raise FileNotFoundError("找不到 FAISS 索引文件")
        if not id_map_file.is_file():
            raise FileNotFoundError("找不到 id_map 文件")

        try:
            payload = json.loads(
                id_map_file.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("id_map JSON 解析失败") from exc
        if not isinstance(payload, dict):
            raise ValueError("id_map 根节点必须是对象")

        index = faiss.read_index(str(index_file))
        ntotal = getattr(index, "ntotal", None)
        if type(ntotal) is not int or ntotal < 0:
            raise ValueError("FAISS 索引条目数无效")
        dimensions = getattr(index, "d", EMBEDDING_DIMENSIONS)
        if type(dimensions) is not int or dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError("FAISS 索引维度必须为 1024")
        if ntotal != len(payload):
            raise ValueError("FAISS 索引数量与 id_map 不一致")

        expected_keys = {str(index_value) for index_value in range(ntotal)}
        if set(payload) != expected_keys:
            raise ValueError("id_map 索引键存在缺项或越界项")

        source_ids: set[str] = set()
        validated_map: dict[str, dict[str, Any]] = {}
        for key in sorted(payload, key=int):
            row = payload[key]
            if not isinstance(row, dict):
                raise ValueError("id_map 条目必须是对象")
            source_id_value = row.get("source_id")
            if type(source_id_value) is not str or not source_id_value.strip():
                raise ValueError("id_map 条目缺少 source_id")
            source_id = source_id_value.strip()
            if source_id in source_ids:
                raise ValueError("id_map 包含重复 source_id")
            source_ids.add(source_id)
            validated_map[key] = {"source_id": source_id}

        self._index = index
        self._id_map = validated_map

    def search_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
    ) -> list[str]:
        """用现成 query vector 搜索，只返回 legacy source ID。"""
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k 必须是 1 到 10 的整数")
        if isinstance(vector, (str, bytes)):
            raise ValueError("query vector 必须是 1024 维有限向量")
        try:
            raw_values = list(vector)
        except Exception:
            raise ValueError("query vector 必须是 1024 维有限向量") from None
        if any(type(value) is bool for value in raw_values):
            raise ValueError("query vector 必须是 1024 维有限向量")
        try:
            values = [float(value) for value in raw_values]
        except Exception:
            raise ValueError("query vector 必须是 1024 维有限向量") from None
        if len(values) != EMBEDDING_DIMENSIONS or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("query vector 必须是 1024 维有限向量")
        if not any(value != 0.0 for value in values):
            raise ValueError("query vector 不能是零向量")

        requested_k = min(top_k, self._index.ntotal)
        if requested_k == 0:
            return []
        query = np.asarray([values], dtype="float32")
        _distances, indices = self._index.search(query, requested_k)
        try:
            index_row = indices.tolist()[0]
        except Exception:
            raise ValueError("FAISS 搜索返回的索引结构无效") from None

        output: list[str] = []
        seen: set[str] = set()
        for raw_index in index_row:
            try:
                index_value = int(raw_index)
            except Exception:
                raise ValueError("FAISS 搜索返回了无效索引") from None
            if index_value == -1:
                continue
            if index_value < 0:
                raise ValueError("FAISS 搜索返回了越界索引")
            row = self._id_map.get(str(index_value))
            if row is None:
                raise ValueError("FAISS 搜索结果在 id_map 中缺项")
            source_id = row["source_id"]
            if source_id in seen:
                raise ValueError("FAISS 搜索返回了重复 source_id")
            seen.add(source_id)
            output.append(source_id)
        return output
