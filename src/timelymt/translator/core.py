"""Provider-neutral source-only translation contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class TranslationError(RuntimeError):
    """Base error raised by a translator implementation."""


class InputTooLongError(TranslationError):
    """Raised instead of silently truncating an overlength model input."""


@dataclass(frozen=True)
class TranslationResult:
    """One hypothesis and source-only inference metadata."""

    translated_text: str
    source_text: str
    source_token_count: int | None = None
    target_token_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Translator(ABC):
    """A model-neutral translator that accepts source text only."""

    @abstractmethod
    def translate(self, text: str) -> TranslationResult:
        """Translate exactly one non-empty source string."""

    def translate_batch(self, texts: Sequence[str]) -> list[TranslationResult]:
        """Translate source strings in input order."""

        return [self.translate(text) for text in texts]


def validate_source_text(text: str) -> None:
    """Reject invalid input without normalizing or mutating valid text."""

    if not isinstance(text, str):
        raise TypeError("source text must be a string")
    if not text.strip():
        raise ValueError("source text must not be empty or whitespace-only")
