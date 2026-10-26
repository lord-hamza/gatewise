import json
import math
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import httpx

from gatewise.errors import ProviderError
from gatewise.types import Chunk, CompletionRequest, CompletionResponse

RETRYABLE_STATUS = (408, 409, 429, 500, 502, 503, 504, 529)
RETRYABLE_ERROR_TYPES = ("overloaded_error", "api_error", "rate_limit_error", "server_error")


class Provider(ABC):
    """A single LLM backend the gateway can route requests to."""

    name: str
    model: str

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse: ...

    @abstractmethod
    def stream(self, request: CompletionRequest) -> AsyncGenerator[Chunk, None]: ...

    async def aclose(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"


class HTTPProvider(Provider):
    """Shared httpx plumbing for providers that speak JSON over HTTP."""

    def __init__(
        self,
        model: str,
        *,
        name: str,
        api_key: str | None,
        base_url: str,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient()

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(
                self.base_url + path, json=payload, headers=self.headers(), timeout=self.timeout
            )
        except httpx.HTTPError as exc:
            raise self.transport_error(exc) from exc
        if not response.is_success:
            raise self.status_error(response)
        return self.decode(response.text)

    async def events(
        self, path: str, payload: Mapping[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            async with self.client.stream(
                "POST",
                self.base_url + path,
                json=payload,
                headers=self.headers(),
                timeout=self.timeout,
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise self.status_error(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        return
                    yield self.decode(data)
        except httpx.HTTPError as exc:
            raise self.transport_error(exc) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def decode(self, text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            raise ProviderError(f"malformed response: {text[:100]!r}", provider=self.name)
        error = data.get("error")
        if isinstance(error, dict):
            raise ProviderError(
                str(error.get("message") or error.get("type") or "unknown error"),
                provider=self.name,
                retryable=error.get("type") in RETRYABLE_ERROR_TYPES,
            )
        return data

    @contextmanager
    def parsing(self) -> Iterator[None]:
        try:
            yield
        except (LookupError, TypeError, ValueError, AttributeError) as exc:
            raise ProviderError(f"malformed response: {exc!r}", provider=self.name) from exc

    def status_error(self, response: httpx.Response) -> ProviderError:
        try:
            message = str(response.json()["error"]["message"])
        except (ValueError, LookupError, TypeError):
            message = f"HTTP {response.status_code} {response.reason_phrase}"
        return ProviderError(
            message,
            provider=self.name,
            status_code=response.status_code,
            retryable=response.status_code in RETRYABLE_STATUS,
            retry_after=parse_retry_after(response.headers.get("retry-after")),
        )

    def transport_error(self, exc: httpx.HTTPError) -> ProviderError:
        message = "request timed out" if isinstance(exc, httpx.TimeoutException) else str(exc)
        return ProviderError(message or type(exc).__name__, provider=self.name, retryable=True)


def parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None
