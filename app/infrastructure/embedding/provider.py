"""异步 Embedding Provider 与固定向量契约。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.config import settings
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)


class EmbeddingProvider(Protocol):
    """知识检索所依赖的异步向量化接口。"""

    @property
    def model(self) -> str:
        """返回当前 Embedding 模型标识。"""
        raise NotImplementedError

    @property
    def dimensions(self) -> int:
        """返回固定输出维度。"""
        raise NotImplementedError

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """按输入顺序返回归一化向量。"""
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放 Provider 持有的异步连接池。"""
        raise NotImplementedError


def validate_and_normalize_vector(
    values: Sequence[float], dimensions: int
) -> list[float]:
    """把 SDK 向量转换为有限、非零且 L2 归一化的固定维度向量。"""
    try:
        vector = [float(value) for value in values]
    except Exception:
        raise EmbeddingResponseError("Embedding 向量无法转换为浮点数") from None

    if len(vector) != dimensions:
        raise EmbeddingResponseError("Embedding 维度与配置不一致")
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingResponseError("Embedding 包含非有限数值")

    scale = max(abs(value) for value in vector)
    if scale <= 0.0:
        raise EmbeddingResponseError("Embedding 不能是零向量")
    scaled = [value / scale for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in scaled))
    return [value / norm for value in scaled]


class OpenAIEmbeddingProvider:
    """基于 AsyncOpenAI 连接池的无状态批量向量化实现。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str = settings.EMBEDDING_MODEL,
        dimensions: int = settings.EMBEDDING_DIMENSIONS,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ) -> None:
        if not model.strip():
            raise EmbeddingInputError("EMBEDDING_MODEL 不能为空")
        if dimensions != 1024:
            raise EmbeddingInputError("EMBEDDING_DIMENSIONS 必须为 1024")
        if batch_size <= 0:
            raise EmbeddingInputError("EMBEDDING_BATCH_SIZE 必须大于 0")

        self._model = model.strip()
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = client or AsyncOpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.EMBEDDING_TIMEOUT,
        )

    @property
    def model(self) -> str:
        """返回去除首尾空白后的模型标识。"""
        return self._model

    @property
    def dimensions(self) -> int:
        """返回固定的 1024 维契约。"""
        return self._dimensions

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """清洗并分批向量化文本，严格恢复每批输入顺序。"""
        cleaned = [" ".join(str(text).split()).strip() for text in texts]
        if not cleaned or any(not text for text in cleaned):
            raise EmbeddingInputError("Embedding 输入必须是非空文本数组")

        output: list[list[float]] = []
        for offset in range(0, len(cleaned), self._batch_size):
            batch = cleaned[offset : offset + self._batch_size]
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimensions,
                    encoding_format="float",
                )
            except (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                APIError,
            ) as exc:
                raise EmbeddingUnavailableError(type(exc).__name__) from None

            try:
                data = list(response.data)
            except Exception:
                raise EmbeddingResponseError("Embedding 返回结构无效") from None
            if len(data) != len(batch):
                raise EmbeddingResponseError("Embedding 返回数量与输入不一致")

            try:
                ordered = sorted(data, key=lambda item: item.index)
                indexes = [item.index for item in ordered]
            except (AttributeError, TypeError):
                raise EmbeddingResponseError("Embedding 返回索引无效") from None
            if any(type(index) is not int for index in indexes):
                raise EmbeddingResponseError("Embedding 返回索引无效")
            if indexes != list(range(len(batch))):
                raise EmbeddingResponseError("Embedding 返回索引不连续")

            output.extend(
                validate_and_normalize_vector(item.embedding, self._dimensions)
                for item in ordered
            )

        if len(output) != len(cleaned):
            raise EmbeddingResponseError("Embedding 返回数量与输入不一致")
        return output

    async def aclose(self) -> None:
        """关闭 AsyncOpenAI 持有的连接池。"""
        await self._client.close()


_embedding_provider: OpenAIEmbeddingProvider | None = None


def get_embedding_provider() -> OpenAIEmbeddingProvider:
    """惰性创建并复用全局 SDK 连接池。"""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = OpenAIEmbeddingProvider()
    return _embedding_provider


async def dispose_embedding_provider() -> None:
    """释放全局 SDK 连接池并清除引用。"""
    global _embedding_provider
    provider = _embedding_provider
    _embedding_provider = None
    if provider is not None:
        await provider.aclose()
