import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from llm_tollgate.providers.base import HTTPProvider
from llm_tollgate.types import Chunk, CompletionRequest, CompletionResponse, Usage

OPENAI_URL = "https://api.openai.com/v1"


class OpenAIProvider(HTTPProvider):
    """OpenAI Chat Completions, or any OpenAI-compatible server via base_url."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str = "openai",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            # The env key is only for OpenAI itself, never for a third-party host.
            base_url = OPENAI_URL
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(
            model, name=name, api_key=api_key, base_url=base_url, timeout=timeout, client=client
        )

    def payload(self, request: CompletionRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        data = await self.post("/chat/completions", self.payload(request))
        with self.parsing():
            choice = data["choices"][0]
            return CompletionResponse(
                text=choice["message"]["content"] or "",
                model=self.model,
                provider=self.name,
                usage=parse_usage(data.get("usage") or {}),
                finish_reason=choice.get("finish_reason"),
            )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[Chunk, None]:
        payload = {
            **self.payload(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with aclosing(self.events("/chat/completions", payload)) as events:
            async for event in events:
                with self.parsing():
                    choice = event["choices"][0] if event.get("choices") else {}
                    chunk = Chunk(
                        text=(choice.get("delta") or {}).get("content") or "",
                        usage=parse_usage(event["usage"]) if event.get("usage") else None,
                        finish_reason=choice.get("finish_reason"),
                    )
                if chunk.text or chunk.usage or chunk.finish_reason:
                    yield chunk


def parse_usage(usage: dict[str, Any]) -> Usage:
    return Usage(int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
