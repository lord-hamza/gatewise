import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Mapping, Sequence
from contextlib import aclosing
from typing import Self, TypeVar

from pydantic import BaseModel

from llm_tollgate.budget import TokenBudget
from llm_tollgate.cache import SemanticCache, cache_key
from llm_tollgate.errors import ProviderError, ProvidersExhaustedError
from llm_tollgate.guardrails import Guardrails
from llm_tollgate.providers import Provider
from llm_tollgate.retry import RetryPolicy
from llm_tollgate.structured import generate_structured
from llm_tollgate.types import (
    Chunk,
    CompletionRequest,
    CompletionResponse,
    Message,
    Usage,
    estimate_tokens,
)

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")

Prompt = str | Sequence[Message | Mapping[str, str]]

logger = logging.getLogger("llm_tollgate")


class Gateway:
    """One async API over several LLM providers, with failover, caching, budgets and guardrails."""

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        retry: RetryPolicy | None = None,
        timeout: float | None = 30.0,
        cache: SemanticCache | None = None,
        budget: TokenBudget | None = None,
        guardrails: Guardrails | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Gateway needs at least one provider")
        self.providers = list(providers)
        self.retry = retry or RetryPolicy()
        self.timeout = timeout
        self.cache = cache
        self.budget = budget
        self.guardrails = guardrails

    async def complete(
        self,
        prompt: Prompt,
        *,
        tenant: str = "default",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> CompletionResponse:
        messages = self._prepare(prompt, system)
        cached = await self._lookup(tenant, messages) if use_cache else None
        if cached is not None:
            return cached
        self._check_budget(tenant, messages)
        response = await self._complete(CompletionRequest(messages, temperature, max_tokens))
        self._charge(tenant, response.usage)
        if use_cache:
            await self._store(tenant, messages, response)
        return response

    async def stream(
        self,
        prompt: Prompt,
        *,
        tenant: str = "default",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> AsyncGenerator[str, None]:
        messages = self._prepare(prompt, system)
        cached = await self._lookup(tenant, messages) if use_cache else None
        if cached is not None:
            yield cached.text
            return
        self._check_budget(tenant, messages)
        request = CompletionRequest(messages, temperature, max_tokens)
        provider: Provider | None = None
        received: list[Chunk] = []
        response: CompletionResponse | None = None
        try:
            async with aclosing(self._stream(request)) as chunks:
                async for source, chunk in chunks:
                    provider = source
                    received.append(chunk)
                    if chunk.text:
                        yield chunk.text
        finally:
            if provider is not None:
                response = assemble(provider, request, received)
                self._charge(tenant, response.usage)
        if use_cache and response is not None:
            await self._store(tenant, messages, response)

    async def structured(
        self,
        prompt: Prompt,
        schema: type[T],
        *,
        tenant: str = "default",
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_repairs: int = 2,
    ) -> T:
        calls = 0

        async def complete(messages: list[Message]) -> str:
            nonlocal calls
            calls += 1
            response = await self.complete(
                messages,
                tenant=tenant,
                temperature=temperature,
                max_tokens=max_tokens,
                use_cache=calls == 1,
            )
            return response.text

        messages = to_messages(prompt, system)
        return await generate_structured(complete, messages, schema, max_repairs=max_repairs)

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _prepare(self, prompt: Prompt, system: str | None) -> list[Message]:
        messages = to_messages(prompt, system)
        return messages if self.guardrails is None else self.guardrails.apply(messages)

    async def _complete(self, request: CompletionRequest) -> CompletionResponse:
        errors: list[ProviderError] = []
        for provider in self.providers:
            if errors:
                logger.warning("%s; failing over to %s", errors[-1], provider.name)
            for attempt in range(1, self.retry.max_attempts + 1):
                try:
                    return await self._timed(provider, provider.complete(request))
                except ProviderError as error:
                    errors.append(error)
                    if not await self._backoff(attempt, error):
                        break
        raise ProvidersExhaustedError(errors)

    async def _stream(
        self, request: CompletionRequest
    ) -> AsyncGenerator[tuple[Provider, Chunk], None]:
        errors: list[ProviderError] = []
        for provider in self.providers:
            if errors:
                logger.warning("%s; failing over to %s", errors[-1], provider.name)
            for attempt in range(1, self.retry.max_attempts + 1):
                started = False
                try:
                    async with aclosing(provider.stream(request)) as chunks:
                        while (chunk := await self._next(provider, chunks)) is not None:
                            started = started or bool(chunk.text)
                            yield provider, chunk
                    return
                except ProviderError as error:
                    if started:
                        raise
                    errors.append(error)
                    if not await self._backoff(attempt, error):
                        break
        raise ProvidersExhaustedError(errors)

    async def _next(self, provider: Provider, chunks: AsyncIterator[Chunk]) -> Chunk | None:
        return await self._timed(provider, anext(chunks, None))

    async def _timed(self, provider: Provider, step: Awaitable[R]) -> R:
        try:
            async with asyncio.timeout(self.timeout):
                return await step
        except TimeoutError:
            raise ProviderError(
                f"no response within {self.timeout}s", provider=provider.name, retryable=True
            ) from None

    async def _backoff(self, attempt: int, error: ProviderError) -> bool:
        if not error.retryable or attempt >= self.retry.max_attempts:
            return False
        delay = self.retry.delay(attempt, error.retry_after)
        logger.warning("%s; retrying in %.2fs", error, delay)
        await asyncio.sleep(delay)
        return True

    async def _lookup(self, tenant: str, messages: list[Message]) -> CompletionResponse | None:
        if self.cache is None:
            return None
        context, prompt = cache_key(messages)
        try:
            return await self.cache.get(prompt, f"{tenant}:{context}")
        except ProviderError as error:
            logger.warning("cache lookup failed: %s", error)
            return None

    async def _store(
        self, tenant: str, messages: list[Message], response: CompletionResponse
    ) -> None:
        if self.cache is None:
            return
        context, prompt = cache_key(messages)
        try:
            await self.cache.set(prompt, response, f"{tenant}:{context}")
        except ProviderError as error:
            logger.warning("cache update failed: %s", error)

    def _check_budget(self, tenant: str, messages: list[Message]) -> None:
        if self.budget is not None:
            self.budget.check(tenant, prompt_tokens(messages))

    def _charge(self, tenant: str, usage: Usage) -> None:
        if self.budget is not None:
            self.budget.charge(tenant, usage.total_tokens)


def to_messages(prompt: Prompt, system: str | None = None) -> list[Message]:
    if isinstance(prompt, str):
        messages = [Message("user", prompt)]
    else:
        messages = [to_message(item) for item in prompt]
    if not messages:
        raise ValueError("prompt has no messages")
    return messages if system is None else [Message("system", system), *messages]


def to_message(item: Message | Mapping[str, str]) -> Message:
    match item:
        case Message():
            return item
        case {"role": "system" | "user" | "assistant" as role, "content": str(content)}:
            return Message(role, content)
        case _:
            raise ValueError(f"not a chat message: {item!r}")


def prompt_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(message.content) for message in messages)


def assemble(
    provider: Provider, request: CompletionRequest, chunks: Sequence[Chunk]
) -> CompletionResponse:
    text = "".join(chunk.text for chunk in chunks)
    usage = Usage(prompt_tokens(request.messages), estimate_tokens(text))
    finish_reason: str | None = None
    for chunk in chunks:
        usage = chunk.usage or usage
        finish_reason = chunk.finish_reason or finish_reason
    return CompletionResponse(text, provider.model, provider.name, usage, finish_reason)
