"""异步 Embedding Provider 的严格契约测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from app.infrastructure.embedding.provider import (
    OpenAIEmbeddingProvider,
    dispose_embedding_provider,
    get_embedding_provider,
)
from app.modules.knowledge.errors import (
    EmbeddingInputError,
    EmbeddingResponseError,
    EmbeddingUnavailableError,
)


@dataclass(frozen=True)
class FakeEmbedding:
    """模拟 SDK 返回的一条向量记录。"""

    index: int
    embedding: Any


class FakeEmbeddingsAPI:
    """记录调用并按顺序返回预设响应。"""

    def __init__(self, responses: list[list[FakeEmbedding] | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("测试未配置本次 Embedding 响应")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return SimpleNamespace(data=response)


class FakeAsyncOpenAI:
    """只实现 Provider 使用到的异步 SDK 接口。"""

    def __init__(self, *responses: list[FakeEmbedding] | BaseException) -> None:
        self.embeddings = FakeEmbeddingsAPI(list(responses))
        self.close_count = 0

    async def close(self) -> None:
        self.close_count += 1


def _axis_vector(axis: int, magnitude: float = 1.0) -> list[float]:
    """构造一个仅指定坐标非零的 1024 维向量。"""
    values = [0.0] * 1024
    values[axis] = magnitude
    return values


def test_provider_sorts_by_index_normalizes_and_sends_cleaned_texts() -> None:
    """乱序响应按 index 复原，输入先清洗，输出执行 L2 归一化。"""
    client = FakeAsyncOpenAI(
        [
            FakeEmbedding(index=1, embedding=_axis_vector(1, 4.0)),
            FakeEmbedding(index=0, embedding=_axis_vector(0, 3.0)),
        ]
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model="  embedding-test  ",
        dimensions=1024,
        batch_size=10,
    )

    result = asyncio.run(provider.embed_texts(["  第一个\n问题 ", "第二个\t问题  "]))

    assert provider.model == "embedding-test"
    assert provider.dimensions == 1024
    assert result[0][0] == pytest.approx(1.0)
    assert result[1][1] == pytest.approx(1.0)
    assert sum(value * value for value in result[0]) == pytest.approx(1.0)
    assert sum(value * value for value in result[1]) == pytest.approx(1.0)
    assert client.embeddings.calls == [
        {
            "model": "embedding-test",
            "input": ["第一个 问题", "第二个 问题"],
            "dimensions": 1024,
            "encoding_format": "float",
        }
    ]


@pytest.mark.parametrize("texts", [[], [" \n\t "], ["有效文本", "  "]])
def test_provider_rejects_empty_or_blank_texts(texts: list[str]) -> None:
    """空序列或任一空白文本都不得触发 SDK 请求。"""
    client = FakeAsyncOpenAI([])
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")

    with pytest.raises(EmbeddingInputError):
        asyncio.run(provider.embed_texts(texts))

    assert client.embeddings.calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": " \t "},
        {"dimensions": 1023},
        {"dimensions": 1536},
        {"batch_size": 0},
        {"batch_size": -1},
    ],
)
def test_provider_rejects_invalid_configuration(overrides: dict[str, Any]) -> None:
    """模型、固定维度与批大小在构造阶段即执行严格校验。"""
    arguments: dict[str, Any] = {
        "client": FakeAsyncOpenAI([]),
        "model": "embedding-test",
        "dimensions": 1024,
        "batch_size": 10,
    }
    arguments.update(overrides)

    with pytest.raises(EmbeddingInputError):
        OpenAIEmbeddingProvider(**arguments)


@pytest.mark.parametrize(
    "bad_vector",
    [
        [0.0] * 1023,
        [0.0] * 1024,
        [float("nan")] + [0.0] * 1023,
        [float("inf")] + [0.0] * 1023,
    ],
    ids=["wrong-dimensions", "zero", "nan", "inf"],
)
def test_provider_rejects_invalid_vectors_without_echoing_input(
    bad_vector: list[float],
) -> None:
    """无效向量转换为稳定响应错误，且不回显输入正文或连接信息。"""
    sensitive_values = (
        "不要出现在错误中的知识正文",
        "https://private.example/v1",
        "sk-private-key",
        "raw-response-body",
    )
    client = FakeAsyncOpenAI([FakeEmbedding(index=0, embedding=bad_vector)])
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")

    with pytest.raises(EmbeddingResponseError) as raised:
        asyncio.run(provider.embed_texts([" ".join(sensitive_values)]))

    assert all(value not in str(raised.value) for value in sensitive_values)


def test_vector_conversion_failure_is_safely_mapped() -> None:
    """float 转换失败也不得泄露原始响应内容。"""

    class UnsafeFloat:
        def __float__(self) -> float:
            raise ValueError(
                "raw-response-body https://private.example/v1 sk-private-key"
            )

    client = FakeAsyncOpenAI(
        [FakeEmbedding(index=0, embedding=[UnsafeFloat()] + [0.0] * 1023)]
    )
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")

    with pytest.raises(EmbeddingResponseError) as raised:
        asyncio.run(provider.embed_texts(["不要泄露的输入正文"]))

    detail = str(raised.value)
    assert "不要泄露的输入正文" not in detail
    assert "raw-response-body" not in detail
    assert "https://private.example/v1" not in detail
    assert "sk-private-key" not in detail


def test_index_property_failure_is_safely_mapped() -> None:
    """读取恶意 index 属性失败时不得泄露原始响应诊断。"""
    sensitive = "raw-response-body sk-private-key https://private.example/v1"

    class UnsafeIndexEmbedding:
        @property
        def index(self) -> int:
            raise ValueError(sensitive)

        @property
        def embedding(self) -> list[float]:
            return _axis_vector(0)

    provider = OpenAIEmbeddingProvider(
        client=FakeAsyncOpenAI([UnsafeIndexEmbedding()]),
        model="embedding-test",
    )

    with pytest.raises(EmbeddingResponseError) as raised:
        asyncio.run(provider.embed_texts(["不要泄露的输入正文"]))

    assert sensitive not in str(raised.value)


def test_embedding_property_failure_is_safely_mapped() -> None:
    """读取恶意 embedding 属性失败时不得泄露原始响应诊断。"""
    sensitive = "raw-response-body sk-private-key https://private.example/v1"

    class UnsafeEmbedding:
        index = 0

        @property
        def embedding(self) -> list[float]:
            raise ValueError(sensitive)

    provider = OpenAIEmbeddingProvider(
        client=FakeAsyncOpenAI([UnsafeEmbedding()]),
        model="embedding-test",
    )

    with pytest.raises(EmbeddingResponseError) as raised:
        asyncio.run(provider.embed_texts(["不要泄露的输入正文"]))

    assert sensitive not in str(raised.value)


def test_provider_normalizes_large_finite_vectors_without_overflow() -> None:
    """有限且非零的超大分量仍应得到有效的单位向量。"""
    client = FakeAsyncOpenAI(
        [FakeEmbedding(index=0, embedding=[1e308] * 1024)]
    )
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")

    result = asyncio.run(provider.embed_texts(["超大有限向量"]))[0]

    assert result[0] == pytest.approx(1.0 / 32.0)
    assert sum(value * value for value in result) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("texts", "data"),
    [
        (["一", "二"], [FakeEmbedding(0, _axis_vector(0)), FakeEmbedding(2, _axis_vector(1))]),
        (["一", "二"], [FakeEmbedding(0, _axis_vector(0)), FakeEmbedding(0, _axis_vector(1))]),
        (["一", "二"], [FakeEmbedding(0, _axis_vector(0))]),
        (["一"], [FakeEmbedding(0, _axis_vector(0)), FakeEmbedding(1, _axis_vector(1))]),
    ],
    ids=["non-contiguous", "duplicate", "too-few", "too-many"],
)
def test_provider_rejects_invalid_response_indexes_or_counts(
    texts: list[str], data: list[FakeEmbedding]
) -> None:
    """每批响应的索引集合必须精确等于 0..n-1。"""
    provider = OpenAIEmbeddingProvider(
        client=FakeAsyncOpenAI(data),
        model="embedding-test",
    )

    with pytest.raises(EmbeddingResponseError):
        asyncio.run(provider.embed_texts(texts))


def test_provider_batches_requests_and_preserves_total_order() -> None:
    """分批响应各自排序后仍保持整批输入的全局顺序。"""
    client = FakeAsyncOpenAI(
        [FakeEmbedding(1, _axis_vector(1)), FakeEmbedding(0, _axis_vector(0))],
        [FakeEmbedding(1, _axis_vector(3)), FakeEmbedding(0, _axis_vector(2))],
        [FakeEmbedding(0, _axis_vector(4))],
    )
    provider = OpenAIEmbeddingProvider(
        client=client,
        model="embedding-test",
        batch_size=2,
    )

    result = asyncio.run(
        provider.embed_texts(["  零 ", "一", "二\n号", " 三", "四 号 "])
    )

    assert [vector[index] for index, vector in enumerate(result)] == pytest.approx(
        [1.0] * 5
    )
    assert [call["input"] for call in client.embeddings.calls] == [
        ["零", "一"],
        ["二 号", "三"],
        ["四 号"],
    ]


def _sdk_errors() -> list[BaseException]:
    """构造包含敏感诊断的真实 OpenAI SDK 异常。"""
    request = httpx.Request(
        "POST", "https://private.example/v1?api_key=sk-private-key"
    )
    response = httpx.Response(429, request=request)
    return [
        APIConnectionError(
            message="raw-response-body sk-private-key", request=request
        ),
        APITimeoutError(request=request),
        RateLimitError(
            "raw-response-body sk-private-key",
            response=response,
            body={"url": "https://private.example/v1"},
        ),
        APIError(
            "raw-response-body sk-private-key",
            request=request,
            body={"url": "https://private.example/v1"},
        ),
    ]


@pytest.mark.parametrize("sdk_error", _sdk_errors(), ids=lambda error: type(error).__name__)
def test_sdk_failures_map_to_safe_unavailable_error(sdk_error: BaseException) -> None:
    """四类 SDK 故障只以异常类型名暴露稳定错误。"""
    provider = OpenAIEmbeddingProvider(
        client=FakeAsyncOpenAI(sdk_error),
        model="embedding-test",
    )

    with pytest.raises(EmbeddingUnavailableError) as raised:
        asyncio.run(
            provider.embed_texts(
                ["正文 https://private.example/v1 sk-private-key raw-response-body"]
            )
        )

    assert str(raised.value) == type(sdk_error).__name__


def test_default_constructor_builds_client_from_embedding_settings(monkeypatch) -> None:
    """默认实例把现有配置字段完整传给 AsyncOpenAI。"""
    from app.infrastructure.embedding import provider as provider_module

    client = FakeAsyncOpenAI([])
    created: list[dict[str, Any]] = []

    def fake_async_openai(**kwargs: Any) -> FakeAsyncOpenAI:
        created.append(kwargs)
        return client

    monkeypatch.setattr(provider_module, "AsyncOpenAI", fake_async_openai)

    provider = provider_module.OpenAIEmbeddingProvider()

    assert provider.model == provider_module.settings.EMBEDDING_MODEL.strip()
    assert provider.dimensions == provider_module.settings.EMBEDDING_DIMENSIONS
    assert created == [
        {
            "api_key": provider_module.settings.EMBEDDING_API_KEY,
            "base_url": provider_module.settings.EMBEDDING_BASE_URL,
            "timeout": provider_module.settings.EMBEDDING_TIMEOUT,
        }
    ]


def test_aclose_closes_injected_sdk_client() -> None:
    """Provider 释放其持有的 SDK 连接池。"""
    client = FakeAsyncOpenAI([])
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")

    asyncio.run(provider.aclose())

    assert client.close_count == 1


def test_global_provider_is_reused_disposed_and_recreated(monkeypatch) -> None:
    """全局 Provider 仅复用连接池，释放后下一次访问创建新实例。"""
    from app.infrastructure.embedding import provider as provider_module

    clients: list[FakeAsyncOpenAI] = []

    def factory() -> OpenAIEmbeddingProvider:
        client = FakeAsyncOpenAI([])
        clients.append(client)
        return OpenAIEmbeddingProvider(client=client, model="embedding-test")

    monkeypatch.setattr(provider_module, "_embedding_provider", None)
    monkeypatch.setattr(provider_module, "OpenAIEmbeddingProvider", factory)

    first = get_embedding_provider()
    assert get_embedding_provider() is first

    asyncio.run(dispose_embedding_provider())
    second = get_embedding_provider()

    assert second is not first
    assert [client.close_count for client in clients] == [1, 0]
    asyncio.run(dispose_embedding_provider())
    assert [client.close_count for client in clients] == [1, 1]


def test_provider_does_not_retain_request_state() -> None:
    """请求文本、索引和结果只存在于调用栈，不写入全局 Provider。"""
    client = FakeAsyncOpenAI([FakeEmbedding(0, _axis_vector(0))])
    provider = OpenAIEmbeddingProvider(client=client, model="embedding-test")
    state_before = set(vars(provider))

    asyncio.run(provider.embed_texts(["一次性查询正文"]))

    assert set(vars(provider)) == state_before
    assert all(
        fragment not in attribute
        for attribute in vars(provider)
        for fragment in ("session", "index", "result", "text", "output")
    )


def test_app_lifespan_disposes_provider_and_database_on_failure(monkeypatch) -> None:
    """应用异常退出也通过 finally 释放 Provider 和数据库连接。"""
    from app import main

    events: list[str] = []

    async def dispose_provider() -> None:
        events.append("embedding")

    async def dispose_database() -> None:
        events.append("database")

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(validate_runtime=lambda: events.append("validate")),
    )
    monkeypatch.setattr(main, "dispose_embedding_provider", dispose_provider)
    monkeypatch.setattr(main, "dispose_engine", dispose_database)

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with main.lifespan(main.app):
                events.append("body")
                raise RuntimeError("boom")

    asyncio.run(exercise())

    assert events == ["validate", "body", "embedding", "database"]
