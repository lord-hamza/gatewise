import asyncio

from pydantic import BaseModel

from gatewise import (
    AnthropicProvider,
    Gateway,
    Guardrails,
    OpenAIProvider,
    SemanticCache,
    TokenBudget,
)


class Ticket(BaseModel):
    category: str
    priority: int
    summary: str


async def main() -> None:
    gateway = Gateway(
        [OpenAIProvider("gpt-5-mini"), AnthropicProvider("claude-haiku-5-5")],
        timeout=20,
        cache=SemanticCache(),
        budget=TokenBudget(default=50_000),
        guardrails=Guardrails(),
    )
    async with gateway:
        response = await gateway.complete(
            "Explain exponential backoff in two sentences.", tenant="acme"
        )
        print(f"{response.provider}/{response.model} ({response.usage.total_tokens} tokens)")
        print(response.text, end="\n\n")

        async for text in gateway.stream("Write a haiku about load balancers.", tenant="acme"):
            print(text, end="", flush=True)
        print("\n")

        ticket = await gateway.structured(
            "The invoice page has been timing out since this morning and I can't pay.",
            Ticket,
            tenant="acme",
        )
        print(ticket.model_dump())


if __name__ == "__main__":
    asyncio.run(main())
