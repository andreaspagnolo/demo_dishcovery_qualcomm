from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Literal


Action = Literal["run"]
TaskName = Literal["task1", "task2", "both", "calories"]


class CommandParseError(ValueError):
    """Raised when a spoken command cannot be mapped to a demo action."""


@dataclass(frozen=True)
class DemoCommand:
    action: Action
    task: TaskName | None = None
    mode: Literal["fast"] | None = None
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
_TASK1_PATTERNS = (
    "find ingredients",
    "find ingredient",
    "find the ingredients",
    "find the ingredient",
    "fine ingredients",
    "fine ingredient",
    "task1",
    "task one",
    "task 1",
    "task won",
    "that s one",
    "thats one",
    "tuscany",
    "tax one",
    "tack one",
    "test one",
    "test won",
    "one",
    "ingredient",
    "ingredients",
    "ingredient recognition",
    "recognition",
)
_TASK2_PATTERNS = (
    "describe the dish",
    "describe dish",
    "describe the dishes",
    "describe this dish",
    "dish description",
    "task2",
    "task two",
    "task 2",
    "task too",
    "task to",
    "task true",
    "task through",
    "task chu",
    "task chew",
    "task control",
    "task truthful",
    "task chufoel",
    "tusk chu",
    "taskoford",
    "daskutru",
    "dasku tru",
    "das kutru",
    "das kuchu",
    "das ge trufuul",
    "that s good true",
    "thats good true",
    "tax two",
    "tax too",
    "tack two",
    "test two",
    "test too",
    "test to",
    "two",
    "too",
    "caption",
    "captions",
    "alignment",
    "align",
    "description",
    "descriptions",
    "describe",
    "captioning",
)
_BOTH_PATTERNS = (
    "execute both",
    "run both",
    "do both",
    "both tasks",
    "execute both tasks",
    "run both tasks",
    "execute all",
    "run all",
    "both",
)
_CALORIE_PATTERNS = (
    "estimate calories",
    "estimate calorie",
    "estimate the calories",
    "estimate dish calories",
    "estimate dish calorie",
    "calorie estimate",
    "calories estimate",
    "calculate calories",
    "calculate calorie",
    "count calories",
    "count calorie",
    "find calories",
    "find calorie",
    "dish calories",
    "dish calorie",
    "calories",
    "calorie",
    "nutrition",
    "nutritional estimate",
)
_FILLER_PATTERNS = (
    "please",
    "okay",
    "ok",
    "hey",
    "dishcovery",
    "discovery",
    "um",
    "uh",
)

def normalize_command_text(text: str) -> str:
    text = re.sub(r"<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\((music|silence|noise|inaudible|applause)[^)]*\)", " ", text, flags=re.IGNORECASE)
    normalized = text.strip().lower().translate(_PUNCT_TRANSLATION)
    normalized = re.sub(r"\btask\s*([12])\b", r"task \1", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    words = [word for word in normalized.split() if word not in _FILLER_PATTERNS]
    normalized = " ".join(words)
    return normalized.strip()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {pattern} " in padded for pattern in patterns)


def _parse_task(text: str) -> TaskName | None:
    has_calories = _contains_any(text, _CALORIE_PATTERNS)
    has_both = _contains_any(text, _BOTH_PATTERNS)
    has_task1 = _contains_any(text, _TASK1_PATTERNS)
    has_task2 = _contains_any(text, _TASK2_PATTERNS)
    if has_calories:
        return "calories"
    if has_both or (has_task1 and has_task2):
        return "both"
    if has_task1 and not has_task2:
        return "task1"
    if has_task2 and not has_task1:
        return "task2"
    return None


def parse_command(text: str) -> DemoCommand:
    normalized = normalize_command_text(text)
    if not normalized:
        raise CommandParseError("Empty command")

    task = _parse_task(normalized)
    if task is None:
        raise CommandParseError(
            "Command must be one of: find ingredients, describe the dish, execute both, estimate calories"
        )
    return DemoCommand(action="run", task=task, mode="fast", text=normalized)
