import contextlib
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from gatewise.errors import StructuredOutputError
from gatewise.types import Message

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```[\w-]*\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"'})


def with_schema(messages: Sequence[Message], schema: type[BaseModel]) -> list[Message]:
    spec = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    instructions = (
        "Respond with JSON only: no markdown fences, no commentary. "
        f"It must match this JSON schema: {spec}"
    )
    if messages and messages[0].role == "system":
        first, *rest = messages
        return [Message("system", f"{first.content}\n\n{instructions}"), *rest]
    return [Message("system", instructions), *messages]


def extract_json(text: str) -> str:
    if fenced := _FENCE.search(text):
        text = fenced.group(1)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return text.strip()
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start : end + 1] if end > start else text[start:].strip()


def parse_output(text: str, schema: type[T]) -> T:
    payload = extract_json(text)
    try:
        return schema.model_validate_json(payload)
    except ValidationError:
        fixed = _TRAILING_COMMA.sub(r"\1", payload.translate(_SMART_QUOTES))
        if fixed != payload:
            with contextlib.suppress(ValidationError):
                return schema.model_validate_json(fixed)
        raise


async def generate_structured(
    complete: Callable[[list[Message]], Awaitable[str]],
    messages: Sequence[Message],
    schema: type[T],
    *,
    max_repairs: int = 2,
) -> T:
    conversation = with_schema(messages, schema)
    repairs = 0
    while True:
        text = await complete(conversation)
        try:
            return parse_output(text, schema)
        except ValidationError as error:
            if repairs >= max_repairs:
                raise StructuredOutputError(
                    f"reply did not match {schema.__name__} after {repairs} repairs", raw=text
                ) from error
            repairs += 1
            feedback = (
                f"Your reply did not match the schema:\n{_describe(error)}\n"
                "Reply again with the corrected JSON only."
            )
            conversation = [*conversation, Message("assistant", text), Message("user", feedback)]


def _describe(error: ValidationError) -> str:
    lines = []
    for issue in error.errors(include_url=False):
        path = ".".join(str(part) for part in issue["loc"])
        lines.append(f"- {path}: {issue['msg']}" if path else f"- {issue['msg']}")
    return "\n".join(lines)
