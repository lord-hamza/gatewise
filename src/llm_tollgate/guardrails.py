import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import replace

from llm_tollgate.errors import PromptInjectionError
from llm_tollgate.types import Message

_OCTET = r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"

# Earlier entries win when two patterns match at the same position.
DEFAULT_PATTERNS: dict[str, str] = {
    "EMAIL": r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b",
    "API_KEY": r"\b(?:(?:sk|pk|rk)[-_](?=[\w-]*\d)[\w-]{20,}|AKIA[0-9A-Z]{16})\b",
    "SSN": r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
    "CREDIT_CARD": (
        r"\b(?:[2-6]\d{3}(?:[ -]?\d{4}){3}|3[47]\d{2}[ -]?\d{6}[ -]?\d{5}|[2-6]\d{12,18})\b"
    ),
    "PHONE": (
        r"(?<![\w+])(?:(?:\+\d{1,3}|1)[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}\b"
        r"|\+\d{1,3}(?:[ .-]?\d{2,5}){2,4}\b"
    ),
    "IP_ADDRESS": rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?!\.?\w)",
}

_SIGNALS: dict[str, tuple[float, re.Pattern[str]]] = {
    "ignore_instructions": (
        0.8,
        re.compile(
            r"\b(?:ignore|disregard|forget|override)(?: \w+){0,3}? "
            r"(?:all|any|previous|prior|above|earlier|preceding|initial|original|your)"
            r"(?: \w+){0,2}? (?:instructions?|prompts?|directives|guidelines|rules)\b"
        ),
    ),
    "reveal_system_prompt": (
        0.6,
        re.compile(
            r"\b(?:reveal|show|print|repeat|output|display|leak|dump|tell|what is|what's|what are)"
            r"(?: me| us)? (?:your|the) "
            r"(?:system prompt|(?:initial|original|hidden|secret) (?:instructions|prompt))"
        ),
    ),
    "unrestricted_roleplay": (
        0.5,
        re.compile(
            r"\b(?:pretend|act as|act like|role[- ]?play|you are now|imagine you are)\b.{0,60}?"
            r"\b(?:unrestricted|unfiltered|uncensored|(?:no|without|free of) (?:any )?"
            r"(?:restrictions|limitations|filters|censorship|guidelines|rules|boundaries))\b"
        ),
    ),
    "jailbreak": (
        0.8,
        re.compile(
            r"\b(?:(?:dan|jailbreak|jailbroken|unrestricted|unfiltered) mode"
            r"|do anything now|you are (?:now )?dan)\b"
        ),
    ),
    "safety_override": (
        0.6,
        re.compile(
            r"\b(?:bypass|disable|override|turn off|ignore|circumvent)(?: \w+){0,2}? "
            r"(?:safety|moderation|ethical) "
            r"(?:filters?|guidelines|policies|rules|restrictions|guardrails|protocols)\b"
        ),
    ),
    "role_marker": (0.7, re.compile(r"<\|[a-z_]+\|>|\[/?inst\]|<</?sys>>")),
    "new_instructions": (
        0.35,
        re.compile(r"\b(?:from now on|new instructions|your new (?:task|role|goal) is)\b"),
    ),
}

_ZERO_WIDTH = re.compile(r"[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")


def _passes_luhn(number: str) -> bool:
    digits = [int(char) for char in reversed(number) if char.isdigit()]
    doubled = (digit * 2 - 9 if digit > 4 else digit * 2 for digit in digits[1::2])
    return (sum(digits[::2]) + sum(doubled)) % 10 == 0


def _normalize(text: str) -> str:
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    return " ".join(text.lower().split())


class PIIRedactor:
    """Replaces emails, card numbers, phone numbers and similar PII with [ENTITY] tags."""

    def __init__(self, patterns: Mapping[str, str] | None = None) -> None:
        patterns = DEFAULT_PATTERNS if patterns is None else patterns
        self.entities = list(patterns)
        self._pattern = re.compile(
            "|".join(f"(?P<{name}>{regex})" for name, regex in patterns.items())
        )

    def redact(self, text: str) -> str:
        return self._pattern.sub(self._replace, text) if self.entities else text

    def _replace(self, match: re.Match[str]) -> str:
        entity = match.lastgroup
        if entity == "CREDIT_CARD" and not _passes_luhn(match.group()):
            return match.group()
        return f"[{entity}]"


class InjectionDetector:
    """Scores text for prompt-injection phrasing using weighted regex signals."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, text: str) -> tuple[float, list[str]]:
        normalized = _normalize(text)
        fired = [label for label, (_, pattern) in _SIGNALS.items() if pattern.search(normalized)]
        # Noisy-OR: each signal is independent evidence, so weak ones add up.
        miss = math.prod(1 - _SIGNALS[label][0] for label in fired)
        return round(1.0 - miss, 3), fired

    def is_injection(self, text: str) -> bool:
        return self.score(text)[0] >= self.threshold


class Guardrails:
    """Blocks likely prompt injections and redacts PII before messages reach a provider."""

    def __init__(
        self,
        *,
        redact_pii: bool = True,
        detect_injection: bool = True,
        injection_threshold: float = 0.5,
        redactor: PIIRedactor | None = None,
        detector: InjectionDetector | None = None,
    ) -> None:
        self.redactor = (redactor or PIIRedactor()) if redact_pii else None
        self.detector = (
            (detector or InjectionDetector(injection_threshold)) if detect_injection else None
        )

    def apply(self, messages: Sequence[Message]) -> list[Message]:
        if self.detector is not None:
            for message in messages:
                if message.role != "user":
                    continue
                score, signals = self.detector.score(message.content)
                if score >= self.detector.threshold:
                    raise PromptInjectionError(score, signals)
        return [self._redact(message) for message in messages]

    def _redact(self, message: Message) -> Message:
        if self.redactor is None or message.role == "system":
            return replace(message)
        return replace(message, content=self.redactor.redact(message.content))
