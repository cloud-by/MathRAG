from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import math

from openai import OpenAI

from app.core.config import settings


@dataclass
class EmbeddingResult:
    text: str
    vector: List[float]


class EmbeddingService:
    """OpenAI-compatible embedding service.

    Works with:
    - DashScope OpenAI-compatible embedding endpoint
    - OpenAI embedding endpoint
    - Any provider exposing an OpenAI-compatible /embeddings API
    """

    def __init__(self) -> None:
        if not settings.EMBEDDING_API_KEY:
            raise ValueError("EMBEDDING_API_KEY 未配置")
        if not settings.EMBEDDING_BASE_URL:
            raise ValueError("EMBEDDING_BASE_URL 未配置")
        if not settings.EMBEDDING_MODEL:
            raise ValueError("EMBEDDING_MODEL 未配置")

        self.model = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.normalize = settings.EMBEDDING_NORMALIZE

        self.client = OpenAI(
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_BASE_URL,
            timeout=settings.EMBEDDING_TIMEOUT,
        )

    def embed_text(self, text: str) -> List[float]:
        vectors = self.embed_texts([text])
        if not vectors:
            raise RuntimeError("embed_text 返回空结果")
        return vectors[0]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        cleaned = [self._clean_text(t) for t in texts if self._clean_text(t)]
        if not cleaned:
            return []

        all_vectors: List[List[float]] = []
        for batch in self._chunked(cleaned, self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )

            batch_vectors = [item.embedding for item in response.data]
            if self.normalize:
                batch_vectors = [self._l2_normalize(vec) for vec in batch_vectors]
            all_vectors.extend(batch_vectors)

        return all_vectors

    def embed_documents(self, documents: Sequence[str]) -> List[EmbeddingResult]:
        vectors = self.embed_texts(documents)
        cleaned = [self._clean_text(t) for t in documents if self._clean_text(t)]
        return [EmbeddingResult(text=text, vector=vec) for text, vec in zip(cleaned, vectors)]

    @staticmethod
    def _clean_text(text: str) -> str:
        if text is None:
            return ""
        return " ".join(str(text).split()).strip()

    @staticmethod
    def _chunked(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
        for i in range(0, len(items), batch_size):
            yield list(items[i : i + batch_size])

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_text(text: str) -> List[float]:
    return get_embedding_service().embed_text(text)


def embed_query(query: str) -> List[float]:
    return get_embedding_service().embed_query(query)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    return get_embedding_service().embed_texts(texts)
