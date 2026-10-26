import pytest
from pydantic import BaseModel, ValidationError

from llm_tollgate.errors import StructuredOutputError
from llm_tollgate.structured import extract_json, generate_structured, parse_output, with_schema
from llm_tollgate.types import Message


class City(BaseModel):
    name: str
    population: int


LISBON = City(name="Lisbon", population=545000)


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "Lisbon", "population": 545000}',
        '```json\n{"name": "Lisbon", "population": 545000}\n```',
        'Sure, here it is:\n{"name": "Lisbon", "population": 545000}\nAnything else?',
        '{"name": "Lisbon", "population": 545000,}',
        "{\u201cname\u201d: \u201cLisbon\u201d, \u201cpopulation\u201d: 545000}",
    ],
)
def test_parse_output_handles_messy_replies(text: str) -> None:
    assert parse_output(text, City) == LISBON


def test_parse_output_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        parse_output('{"name": "Lisbon", "population": "lots"}', City)


def test_extract_json_returns_outermost_span() -> None:
    text = 'Result: [{"tags": [1, 2]}, {"meta": {}}] as requested.'
    assert extract_json(text) == '[{"tags": [1, 2]}, {"meta": {}}]'


def test_with_schema_merges_into_existing_system_message() -> None:
    messages = [Message("system", "Be terse."), Message("user", "Lisbon?")]
    prompted = with_schema(messages, City)
    assert [message.role for message in prompted] == ["system", "user"]
    assert prompted[0].content.startswith("Be terse.\n\n")
    assert '"population"' in prompted[0].content
    assert messages[0].content == "Be terse."


def test_with_schema_prepends_system_message() -> None:
    prompted = with_schema([Message("user", "Lisbon?")], City)
    assert [message.role for message in prompted] == ["system", "user"]
    assert "JSON" in prompted[0].content


async def test_generate_structured_repairs_bad_reply() -> None:
    replies = iter(['{"name": "Lisbon"}', '{"name": "Lisbon", "population": 545000}'])
    calls: list[list[Message]] = []

    async def complete(messages: list[Message]) -> str:
        calls.append(messages)
        return next(replies)

    assert await generate_structured(complete, [Message("user", "Lisbon?")], City) == LISBON
    first, retry = calls
    assert retry[: len(first)] == first
    assert retry[-2] == Message("assistant", '{"name": "Lisbon"}')
    assert retry[-1].role == "user"
    assert "population: Field required" in retry[-1].content


async def test_generate_structured_gives_up_after_max_repairs() -> None:
    replies = iter(["no idea", "still no idea"])
    calls: list[list[Message]] = []

    async def complete(messages: list[Message]) -> str:
        calls.append(messages)
        return next(replies)

    with pytest.raises(StructuredOutputError) as info:
        await generate_structured(complete, [Message("user", "Lisbon?")], City, max_repairs=1)
    assert info.value.raw == "still no idea"
    assert len(calls) == 2
