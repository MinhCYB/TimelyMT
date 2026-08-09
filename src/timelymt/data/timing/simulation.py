"""Synthetic duration and integer token-boundary allocation."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Sequence

from .tokenization import alphanumeric_weight


ALLOCATION_MODES = {"uniform", "character_weighted"}


def simulated_duration_ms(token_count: int, words_per_second: float) -> int:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    _validate_words_per_second(words_per_second)
    duration = Decimal(token_count) * Decimal(1000) / Decimal(str(words_per_second))
    return int(duration.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def allocate_emit_times(
    tokens: Sequence[str],
    start_ms: int,
    end_ms: int,
    allocation: str,
) -> list[int]:
    """Allocate token completion boundaries exactly across an integer interval."""
    if allocation not in ALLOCATION_MODES:
        raise ValueError(f"Unsupported allocation mode: {allocation!r}")
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError("Invalid segment interval")
    if not tokens:
        return []
    weights = [1] * len(tokens) if allocation == "uniform" else [alphanumeric_weight(token) for token in tokens]
    total_weight = sum(weights)
    duration = end_ms - start_ms
    cumulative = 0
    emit_times: list[int] = []
    for weight in weights:
        cumulative += weight
        emit_times.append(start_ms + duration * cumulative // total_weight)
    emit_times[-1] = end_ms
    return emit_times


def _validate_words_per_second(words_per_second: float) -> None:
    if (
        isinstance(words_per_second, bool)
        or not isinstance(words_per_second, (int, float))
        or not math.isfinite(words_per_second)
        or words_per_second <= 0
    ):
        raise ValueError("words_per_second must be finite and greater than zero")
