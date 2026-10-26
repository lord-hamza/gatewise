import hashlib
import json
import math
import os
import re
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Protocol

import httpx

from gatewise.errors import ProviderError
from gatewise.types import CompletionResponse, Message

_WORD = re.compile(r"\w+")
_RECENT_VECTORS = 64


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Offline embedder that hashes word unigrams and bigrams into a fixed-size vector."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        vector = [0.0] * self.dim
        for feature in [*words, *map(" ".join, pairwise(words))]:
            digest = zlib.crc32(feature.encode())
            vector[digest % self.dim] += 1.0 if digest >> 31 else -1.0
        return _normalise(vector)


class OpenAIEmbedder:
    """Embeddings from OpenAI or any OpenAI-compatible server."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if base_url is None:
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                headers=self._headers,
            )
        except httpx.HTTPError as error:
            raise ProviderError(
                f"embedding request failed: {error!r}", provider="openai", retryable=True
            ) from error
        if response.is_error:
            status = response.status_code
            raise ProviderError(
                f"embedding request failed with HTTP {status}",
                provider="openai",
                status_code=status,
                retryable=status in (408, 409, 429) or status >= 500,
            )
        try:
            return [float(value) for value in response.json()["data"][0]["embedding"]]
        except (ValueError, LookupError, TypeError) as error:
            raise ProviderError("malformed embedding response", provider="openai") from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(slots=True)
class _Entry:
    vector: list[float]
    response: CompletionResponse
    expires_at: float


class SemanticCache:
    """In-memory LRU cache that also matches prompts by embedding similarity."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.95,
        max_entries: int = 1024,
        ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._recent: OrderedDict[str, list[float]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    async def get(self, prompt: str, namespace: str = "") -> CompletionResponse | None:
        self._purge()
        key: tuple[str, str] | None = (namespace, prompt)
        if key not in self._entries:
            key = self._closest(await self._embed(prompt), namespace)
        if key is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return replace(self._entries[key].response, cached=True)

    async def set(self, prompt: str, response: CompletionResponse, namespace: str = "") -> None:
        vector = await self._embed(prompt)
        self._purge()
        key = (namespace, prompt)
        expires_at = math.inf if self.ttl is None else self._clock() + self.ttl
        self._entries[key] = _Entry(vector, response, expires_at)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self._recent.clear()
        self.hits = self.misses = 0

    async def _embed(self, prompt: str) -> list[float]:
        vector = self._recent.get(prompt)
        if vector is None:
            vector = _normalise(await self.embedder.embed(prompt))
            self._recent[prompt] = vector
            if len(self._recent) > _RECENT_VECTORS:
                self._recent.popitem(last=False)
        return vector

    def _closest(self, vector: list[float], namespace: str) -> tuple[str, str] | None:
        best: tuple[str, str] | None = None
        best_score = self.threshold
        for key, entry in self._entries.items():
            if key[0] != namespace:
                continue
            score = sum(x * y for x, y in zip(vector, entry.vector, strict=True))
            if score >= best_score:
                best, best_score = key, score
        return best

    def _purge(self) -> None:
        now = self._clock()
        for key in [key for key, entry in self._entries.items() if entry.expires_at <= now]:
            del self._entries[key]


def cache_key(messages: Sequence[Message]) -> tuple[str, str]:
    *context, last = messages
    payload = json.dumps([[message.role, message.content] for message in context])
    return hashlib.sha256(payload.encode()).hexdigest()[:16], last.content


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
