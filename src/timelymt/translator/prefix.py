"""Provider-neutral inference over caller-provided source prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core import TranslationResult, Translator


@dataclass(frozen=True)
class PrefixTranslation:
    """Translation paired with the position and exact source prefix provided."""

    prefix_index: int
    source_text: str
    translation: TranslationResult


def translate_prefixes(
    translator: Translator, prefixes: Sequence[str], *, batch_size: int = 8
) -> list[PrefixTranslation]:
    """Translate existing prefixes in order without deriving any new source text."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    sources = list(prefixes)
    results: list[PrefixTranslation] = []
    for start in range(0, len(sources), batch_size):
        batch = sources[start : start + batch_size]
        translations = translator.translate_batch(batch)
        if len(translations) != len(batch):
            raise RuntimeError("translator returned a different number of prefix translations")
        results.extend(
            PrefixTranslation(start + offset, source, translation)
            for offset, (source, translation) in enumerate(zip(batch, translations, strict=True))
        )
    return results
