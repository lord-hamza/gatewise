from dataclasses import dataclass, field
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(slots=True)
class CompletionRequest:
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class CompletionResponse:
    text: str
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    cached: bool = False


@dataclass(slots=True)
class Chunk:
    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4
