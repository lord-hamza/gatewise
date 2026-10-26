import json
import math
from unittest.mock import Mock

import httpx
import pytest

from gatewise.cache import HashingEmbedder, OpenAIEmbedder, SemanticCache, cache_key
from gatewise.errors import ProviderError
from gatewise.types import CompletionResponse, Message

QUESTION = "What is the capital of France?"


def reply(text: str = "Paris") -> CompletionResponse:
    return CompletionResponse(text=text, model="fake-1", provider="fake")


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return await super().embed(text)


async def test_exact_hit_returns_cached_copy() -> None:
    cache = SemanticCache()
    original = reply()
    assert await cache.get(QUESTION) is None
    await cache.set(QUESTION, original)

    hit = await cache.get(QUESTION)
    assert hit is not None
    assert hit is not original
    assert hit.cached and not original.cached
    assert hit.text == "Paris"
    assert (cache.hits, cache.misses) == (1, 1)


@pytest.mark.parametrize(
    ("stored", "asked", "hit"),
    [
        (QUESTION, "what is the capital of france", True),
        (
            "Can you explain how a hash map works in Python?",
            "can you explain how a hash map works in python, please?",
            True,
        ),
        (QUESTION, "What is the capital of Spain?", False),
        (
            "What is the boiling point of water at sea level in Celsius?",
            "What is the boiling point of water at sea level in Fahrenheit?",
            False,
        ),
    ],
)
async def test_similar_prompts(stored: str, asked: str, hit: bool) -> None:
    cache = SemanticCache()
    await cache.set(stored, reply())
    assert (await cache.get(asked) is not None) is hit


async def test_namespaces_are_isolated() -> None:
    cache = SemanticCache()
    await cache.set(QUESTION, reply(), namespace="acme")
    assert await cache.get(QUESTION, namespace="globex") is None
    assert await cache.get(QUESTION, namespace="acme") is not None


async def test_entries_expire_after_ttl() -> None:
    clock = Mock(return_value=0.0)
    cache = SemanticCache(ttl=60, clock=clock)
    await cache.set(QUESTION, reply())
    clock.return_value = 59.0
    assert await cache.get(QUESTION) is not None
    clock.return_value = 60.0
    assert await cache.get(QUESTION) is None
    assert len(cache) == 0


async def test_evicts_least_recently_used() -> None:
    cache = SemanticCache(max_entries=2)
    await cache.set("first question", reply("1"))
    await cache.set("second question", reply("2"))
    assert await cache.get("first question") is not None
    await cache.set("third question", reply("3"))

    assert len(cache) == 2
    assert await cache.get("second question") is None
    assert await cache.get("first question") is not None


async def test_miss_then_set_embeds_once() -> None:
    embedder = CountingEmbedder()
    cache = SemanticCache(embedder)
    assert await cache.get(QUESTION) is None
    await cache.set(QUESTION, reply())
    assert await cache.get(QUESTION) is not None
    assert embedder.calls == 1


async def test_hashing_embedder() -> None:
    embedder = HashingEmbedder(dim=64)
    vector = await embedder.embed("The quick brown fox")
    assert len(vector) == 64
    assert math.isclose(sum(value * value for value in vector), 1.0)
    assert vector == await HashingEmbedder(dim=64).embed("the QUICK, brown fox!")
    assert await embedder.embed("") == [0.0] * 64


def test_cache_key_splits_context_from_prompt() -> None:
    system = Message("system", "Answer in one word.")
    question = Message("user", QUESTION)
    context, prompt = cache_key([system, question])

    assert prompt == QUESTION
    assert cache_key([system, Message("user", "Why?")])[0] == context
    assert cache_key([Message("system", "Answer at length."), question])[0] != context
    assert cache_key([question])[0] != context


async def test_openai_embedder_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [0.6, 0.8]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = OpenAIEmbedder(api_key="sk-test", client=client)

    assert await embedder.embed("hello") == [0.6, 0.8]
    assert str(requests[0].url) == "https://api.openai.com/v1/embeddings"
    assert requests[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(requests[0].content) == {"model": "text-embedding-3-small", "input": "hello"}


async def test_openai_embedder_env_key_only_for_default_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await OpenAIEmbedder(client=client).embed("hi")
    await OpenAIEmbedder(base_url="http://localhost:11434/v1", client=client).embed("hi")

    assert requests[0].headers["authorization"] == "Bearer sk-env"
    assert str(requests[1].url) == "http://localhost:11434/v1/embeddings"
    assert "authorization" not in requests[1].headers


@pytest.mark.parametrize(
    ("response", "retryable"),
    [
        (httpx.Response(503), True),
        (httpx.Response(401, json={"error": {"message": "bad key"}}), False),
        (httpx.Response(200, json={"data": []}), False),
    ],
)
async def test_openai_embedder_errors(response: httpx.Response, retryable: bool) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    with pytest.raises(ProviderError) as info:
        await OpenAIEmbedder(api_key="sk-test", client=client).embed("hi")
    assert info.value.retryable is retryable
