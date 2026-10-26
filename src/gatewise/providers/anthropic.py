import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from llm_tollgate.providers.base import HTTPProvider
from llm_tollgate.types import Chunk, CompletionRequest, CompletionResponse, Usage

ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(HTTPProvider):
    """Anthropic Messages API."""

    api_key: str

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        name: str = "anthropic",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key missing: pass api_key or set ANTHROPIC_API_KEY")
        super().__init__(
            model,
            name=name,
            api_key=api_key,
            base_url=base_url or ANTHROPIC_URL,
            timeout=timeout,
            client=client,
        )
        self.max_tokens = max_tokens

    def headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION}

    def payload(self, request: CompletionRequest) -> dict[str, Any]:
        system = [message.content for message in request.messages if message.role == "system"]
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
                if message.role != "system"
            ],
        }
        if system:
            payload["system"] = "\n\n".join(system)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        data = await self.post("/messages", self.payload(request))
        with self.parsing():
            usage = data["usage"]
            return CompletionResponse(
                text="".join(block["text"] for block in data["content"] if block["type"] == "text"),
                model=self.model,
                provider=self.name,
                usage=Usage(int(usage["input_tokens"]), int(usage["output_tokens"])),
                finish_reason=data.get("stop_reason"),
            )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[Chunk, None]:
        usage = Usage()
        finish_reason: str | None = None
        payload = {**self.payload(request), "stream": True}
        async with aclosing(self.events("/messages", payload)) as events:
            async for event in events:
                with self.parsing():
                    match event["type"]:
                        case "message_start":
                            usage.input_tokens = int(event["message"]["usage"]["input_tokens"])
                        case "content_block_delta" if event["delta"]["type"] == "text_delta":
                            yield Chunk(event["delta"]["text"])
                        case "message_delta":
                            usage.output_tokens = int(event["usage"]["output_tokens"])
                            finish_reason = event["delta"].get("stop_reason")
        yield Chunk(usage=usage, finish_reason=finish_reason)
