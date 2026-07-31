"""知识向量检索的请求级编排服务。"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.database.session import get_session_factory
from app.infrastructure.embedding.provider import (
    EmbeddingProvider,
    get_embedding_provider,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    KnowledgeSearchError,
)
from app.modules.knowledge.repository import KnowledgeRepository
from app.modules.knowledge.search import KnowledgeSearchHit, merge_search_hits


class KnowledgeSearchService:
    """先批量向量化，再用单一会话顺序检索并合并结果。"""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        provider: EmbeddingProvider,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider

    @property
    def embedding_model(self) -> str:
        """返回本服务检索向量使用的只读模型标识。"""
        return self._provider.model

    async def search(
        self,
        queries: Sequence[str],
        *,
        top_k: int,
    ) -> list[KnowledgeSearchHit]:
        """清洗查询，并在关闭数据库会话后返回跨查询 Top-K。"""
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise KnowledgeSearchError("top_k 必须是 1 到 10 的整数")

        if isinstance(queries, (str, bytes, bytearray)):
            raise EmbeddingInputError("queries 必须是字符串序列")

        normalized: list[str] = []
        seen: set[str] = set()
        for query in queries:
            if not isinstance(query, str):
                raise EmbeddingInputError("查询元素必须是字符串")
            cleaned = " ".join(query.split()).strip()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
        if not normalized:
            raise KnowledgeSearchError("查询文本不能为空")
        if len(normalized) > 4:
            raise KnowledgeSearchError("单次检索最多支持 4 条不同查询")

        vectors = await self._provider.embed_texts(normalized)
        try:
            vector_count = len(vectors)
        except Exception:
            raise EmbeddingResponseError("Embedding 返回数量与输入不一致") from None
        if vector_count != len(normalized):
            raise EmbeddingResponseError("Embedding 返回数量与输入不一致")

        groups: list[list[KnowledgeSearchHit]] = []
        async with self._session_factory() as session:
            repository = KnowledgeRepository(session)
            for vector in vectors:
                groups.append(
                    await repository.search_ready_chunks(
                        query_vector=vector,
                        embedding_model=self._provider.model,
                        limit=top_k,
                    )
                )

        return merge_search_hits(groups, top_k)


def build_knowledge_search_service() -> KnowledgeSearchService:
    """组合进程级 Provider 与按请求创建会话的工厂。"""
    return KnowledgeSearchService(get_session_factory(), get_embedding_provider())
