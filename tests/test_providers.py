import json
from typing import Any

import httpx
import pytest

from gatewise.errors import ProviderError
from gatewise.providers import AnthropicProvider, FakeProvider, OpenAIProvider, Provider
from gatewise.types import (
    Chunk,
    CompletionRequest,
    CompletionResponse,
    Message,
    Usage,
    estimate_tokens,
)

REQUEST = CompletionRequest([Message("system", "Be brief."), Message("user", "Hi there")])
OPENAI_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
}


def serve(status: int = 200, **kwargs: Any) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, **kwargs)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), sent


def sse(*events: dict[str, Any] | str) -> bytes:
    lines = (event if isinstance(event, str) else json.dumps(event) for event in events)
    return "".join(f"data: {line}\n\n" for line in lines).encode()


async def collect(provider: Provider, request: CompletionRequest = REQUEST) -> list[Chunk]:
    return [chunk async for chunk in provider.stream(request)]


async def test_openai_complete() -> None:
    client, sent = serve(json=OPENAI_REPLY)
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    response = await provider.complete(REQUEST)

    assert response == CompletionResponse("Hello!", "gpt-5-mini", "openai", Usage(12, 3), "stop")
    assert sent[0].url == "https://api.openai.com/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer sk-test"
    assert json.loads(sent[0].content) == {
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi there"},
        ],
    }


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, {}),
        ({"temperature": 0.0}, {"temperature": 0.0}),
        ({"temperature": 0.7, "max_tokens": 64}, {"temperature": 0.7, "max_completion_tokens": 64}),
    ],
)
async def test_openai_sends_sampling_options_only_when_set(
    options: dict[str, Any], expected: dict[str, Any]
) -> None:
    client, sent = serve(json=OPENAI_REPLY)
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    await provider.complete(CompletionRequest(REQUEST.messages, **options))

    body = json.loads(sent[0].content)
    keys = ("temperature", "max_tokens", "max_completion_tokens")
    assert {key: body[key] for key in keys if key in body} == expected


async def test_compatible_servers_get_max_tokens() -> None:
    client, sent = serve(json=OPENAI_REPLY)
    provider = OpenAIProvider("llama3.2", base_url="http://localhost:11434/v1", client=client)

    await provider.complete(CompletionRequest(REQUEST.messages, max_tokens=64))

    assert json.loads(sent[0].content)["max_tokens"] == 64


async def test_openai_stream() -> None:
    client, sent = serve(
        content=sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": None}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            "[DONE]",
            {"choices": [{"delta": {"content": "after done"}}]},
        )
    )
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    chunks = await collect(provider)

    assert chunks == [
        Chunk("Hello"),
        Chunk(" world"),
        Chunk(finish_reason="stop"),
        Chunk(usage=Usage(9, 2)),
    ]
    body = json.loads(sent[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


async def test_openai_env_key_only_goes_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    client, sent = serve(json=OPENAI_REPLY)

    await OpenAIProvider("gpt-5-mini", client=client).complete(REQUEST)
    await OpenAIProvider("llama3.2", base_url="http://localhost:11434/v1/", client=client).complete(
        REQUEST
    )

    assert sent[0].headers["authorization"] == "Bearer sk-env"
    assert sent[1].url == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in sent[1].headers


async def test_anthropic_complete() -> None:
    client, sent = serve(
        json={
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Bonjour"},
                {"type": "text", "text": " !"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 20, "output_tokens": 4},
        }
    )
    provider = AnthropicProvider("claude-haiku-5-5", api_key="sk-ant-test", client=client)
    system = [Message("system", "Be brief."), Message("system", "Answer in French.")]

    response = await provider.complete(CompletionRequest([*system, Message("user", "Hi")]))

    assert response == CompletionResponse(
        "Bonjour !", "claude-haiku-5-5", "anthropic", Usage(20, 4), "end_turn"
    )
    assert sent[0].url == "https://api.anthropic.com/v1/messages"
    assert sent[0].headers["x-api-key"] == "sk-ant-test"
    assert sent[0].headers["anthropic-version"] == "2023-06-01"
    assert json.loads(sent[0].content) == {
        "model": "claude-haiku-5-5",
        "max_tokens": 1024,
        "system": "Be brief.\n\nAnswer in French.",
        "messages": [{"role": "user", "content": "Hi"}],
    }


async def test_anthropic_stream() -> None:
    client, sent = serve(
        content=sse(
            {"type": "message_start", "message": {"usage": {"input_tokens": 25}}},
            {"type": "ping"},
            {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
            {"type": "message_stop"},
        )
    )
    provider = AnthropicProvider("claude-haiku-5-5", api_key="sk-ant-test", client=client)

    chunks = await collect(provider)

    assert chunks == [
        Chunk("Hello"),
        Chunk(" world"),
        Chunk(usage=Usage(25, 12), finish_reason="end_turn"),
    ]
    assert json.loads(sent[0].content)["stream"] is True


@pytest.mark.parametrize(
    ("kind", "retryable"), [("overloaded_error", True), ("invalid_request_error", False)]
)
async def test_anthropic_error_event_mid_stream(kind: str, retryable: bool) -> None:
    client, _ = serve(
        content=sse(
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hel"}},
            {"type": "error", "error": {"type": kind, "message": "stream interrupted"}},
        )
    )
    provider = AnthropicProvider("claude-haiku-5-5", api_key="sk-ant-test", client=client)
    chunks: list[Chunk] = []

    with pytest.raises(ProviderError, match="stream interrupted") as info:
        async for chunk in provider.stream(REQUEST):
            chunks.append(chunk)

    assert chunks == [Chunk("Hel")]
    assert info.value.retryable is retryable


def test_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("claude-haiku-5-5")


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("status", "headers", "retryable", "retry_after"),
    [
        (429, {"Retry-After": "2"}, True, 2.0),
        (529, {"Retry-After": "Mon, 26 Oct 2026 09:00:00 GMT"}, True, None),
        (500, {}, True, None),
        (401, {}, False, None),
    ],
)
async def test_http_errors_are_mapped(
    status: int, headers: dict[str, str], retryable: bool, retry_after: float | None, stream: bool
) -> None:
    client, _ = serve(status, headers=headers, json={"error": {"type": "error", "message": "nope"}})
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    with pytest.raises(ProviderError) as info:
        await (collect(provider) if stream else provider.complete(REQUEST))

    error = info.value
    assert str(error) == "openai: nope"
    assert error.status_code == status
    assert (error.retryable, error.retry_after) == (retryable, retry_after)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("error", [httpx.ConnectError("connection refused"), httpx.ReadTimeout("")])
async def test_transport_errors_are_retryable(error: httpx.HTTPError, stream: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    with pytest.raises(ProviderError) as info:
        await (collect(provider) if stream else provider.complete(REQUEST))

    assert info.value.retryable
    assert info.value.status_code is None


@pytest.mark.parametrize(
    "body",
    [
        {"content": b"<html>Bad gateway</html>"},
        {"json": ["not", "an", "object"]},
        {"json": {"choices": []}},
        {"json": {"choices": [{"message": None}]}},
    ],
)
async def test_malformed_responses_raise_provider_error(body: dict[str, Any]) -> None:
    client, _ = serve(**body)
    provider = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=client)

    with pytest.raises(ProviderError, match="malformed response"):
        await provider.complete(REQUEST)


async def test_fake_provider_follows_its_script() -> None:
    boom = ProviderError("boom", provider="fake", retryable=True)
    provider = FakeProvider(["first", boom, "last"])

    assert (await provider.complete(REQUEST)).text == "first"
    with pytest.raises(ProviderError) as info:
        await provider.complete(REQUEST)
    assert info.value is boom
    assert [(await provider.complete(REQUEST)).text for _ in range(3)] == ["last"] * 3
    assert len(provider.calls) == 5


async def test_fake_provider_stream_rebuilds_the_text() -> None:
    text = "  Hello,  world!\nSecond line "

    chunks = await collect(FakeProvider(text))

    assert [chunk.text for chunk in chunks[:-1]] == ["  Hello,  ", "world!\n", "Second ", "line "]
    assert "".join(chunk.text for chunk in chunks) == text
    prompt = sum(estimate_tokens(message.content) for message in REQUEST.messages)
    assert chunks[-1] == Chunk(usage=Usage(prompt, estimate_tokens(text)), finish_reason="stop")


async def test_aclose_only_closes_clients_the_provider_created() -> None:
    injected = httpx.AsyncClient()
    shared = OpenAIProvider("gpt-5-mini", api_key="sk-test", client=injected)
    owned = AnthropicProvider("claude-haiku-5-5", api_key="sk-ant-test")

    await shared.aclose()
    await owned.aclose()

    assert not injected.is_closed
    assert owned.client.is_closed
    await injected.aclose()
