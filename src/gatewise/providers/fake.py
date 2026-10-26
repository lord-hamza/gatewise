import asyncio
import re
from collections.abc import AsyncGenerator, Sequence

from gatewise.providers.base import Provider
from gatewise.types import Chunk, CompletionRequest, CompletionResponse, Usage, estimate_tokens


class FakeProvider(Provider):
    """Scripted provider for tests and offline demos."""

    def __init__(
        self,
        responses: str | Sequence[str | Exception] = "ok",
        *,
        name: str = "fake",
        model: str = "fake-1",
        delay: float = 0.0,
    ) -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        if not self.responses:
            raise ValueError("FakeProvider needs at least one response")
        self.name = name
        self.model = model
        self.delay = delay
        self.calls: list[CompletionRequest] = []

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        text = await self._respond(request)
        return CompletionResponse(
            text=text,
            model=self.model,
            provider=self.name,
            usage=estimate_usage(request, text),
            finish_reason="stop",
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[Chunk, None]:
        text = await self._respond(request)
        for word in re.findall(r"\s*\S+\s*|\s+", text):
            yield Chunk(word)
        yield Chunk(usage=estimate_usage(request, text), finish_reason="stop")

    async def _respond(self, request: CompletionRequest) -> str:
        self.calls.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if isinstance(item, Exception):
            raise item
        return item


def estimate_usage(request: CompletionRequest, text: str) -> Usage:
    prompt = sum(estimate_tokens(message.content) for message in request.messages)
    return Usage(prompt, estimate_tokens(text))
