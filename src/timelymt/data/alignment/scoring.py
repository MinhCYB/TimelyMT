"""Deterministic structural scoring for candidate bilingual segment groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata

from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript


ANNOTATION = re.compile(r"^\s*[\[(][^\])]+[\])]\s*$")


@dataclass(frozen=True)
class ScoredGroup:
    cost: float
    features: dict[str, float | int | None]


class GroupScorer:
    """Score groups with lower costs representing better structural compatibility."""

    def __init__(
        self,
        source: ParsedTranscript,
        target: ParsedTranscript,
        *,
        group_penalty: float = 0.65,
    ) -> None:
        self.source = source
        self.target = target
        self.source_lengths = [_normalized_length(segment.text) for segment in source.segments]
        self.target_lengths = [_normalized_length(segment.text) for segment in target.segments]
        self.source_prefix = _prefix_sums(self.source_lengths)
        self.target_prefix = _prefix_sums(self.target_lengths)
        self.source_total = max(1, self.source_prefix[-1])
        self.target_total = max(1, self.target_prefix[-1])
        self.expected_ratio = self.target_total / self.source_total
        self.source_time_range = _time_range(source)
        self.target_time_range = _time_range(target)
        self.group_penalty = group_penalty

    def score(self, source_start: int, source_size: int, target_start: int, target_size: int) -> ScoredGroup:
        source_group = self.source.segments[source_start : source_start + source_size]
        target_group = self.target.segments[target_start : target_start + target_size]
        source_length = self.source_prefix[source_start + source_size] - self.source_prefix[source_start]
        target_length = self.target_prefix[target_start + target_size] - self.target_prefix[target_start]
        observed_ratio = (target_length + 1) / (source_length + 1)
        length_cost = abs(math.log(observed_ratio / self.expected_ratio))

        source_midpoint = (
            self.source_prefix[source_start] + self.source_prefix[source_start + source_size]
        ) / (2 * self.source_total)
        target_midpoint = (
            self.target_prefix[target_start] + self.target_prefix[target_start + target_size]
        ) / (2 * self.target_total)
        position_cost = 1.5 * abs(source_midpoint - target_midpoint)
        timing_cost = self._timing_cost(source_group, target_group)
        group_penalty = self.group_penalty * (source_size + target_size - 2)
        if source_size > 1 and target_size > 1:
            group_penalty += 0.2
        annotation_penalty = _annotation_penalty(source_group, target_group)
        total = length_cost + position_cost + group_penalty + annotation_penalty
        if timing_cost is not None:
            total += timing_cost
        features: dict[str, float | int | None] = {
            "length_cost": _rounded(length_cost),
            "position_cost": _rounded(position_cost),
            "timing_cost": _rounded(timing_cost) if timing_cost is not None else None,
            "group_penalty": _rounded(group_penalty),
            "annotation_penalty": _rounded(annotation_penalty),
            "source_normalized_length": source_length,
            "target_normalized_length": target_length,
            "expected_target_source_ratio": _rounded(self.expected_ratio),
        }
        return ScoredGroup(_rounded(total), features)

    def _timing_cost(
        self,
        source_group: tuple[ParsedSegment, ...],
        target_group: tuple[ParsedSegment, ...],
    ) -> float | None:
        source_times = [segment.start_ms for segment in source_group if segment.start_ms is not None]
        target_times = [segment.start_ms for segment in target_group if segment.start_ms is not None]
        if not source_times or not target_times or self.source_time_range is None or self.target_time_range is None:
            return None
        source_start, source_span = self.source_time_range
        target_start, target_span = self.target_time_range
        source_position = (sum(source_times) / len(source_times) - source_start) / source_span
        target_position = (sum(target_times) / len(target_times) - target_start) / target_span
        return 0.5 * abs(source_position - target_position)


def _normalized_length(text: str) -> int:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if ANNOTATION.fullmatch(normalized):
        return 1
    normalized = "".join(character for character in normalized if character.isalnum() or character.isspace())
    return len("".join(normalized.split()))


def _annotation_penalty(
    source_group: tuple[ParsedSegment, ...], target_group: tuple[ParsedSegment, ...]
) -> float:
    source_annotations = all(ANNOTATION.fullmatch(segment.text) for segment in source_group)
    target_annotations = all(ANNOTATION.fullmatch(segment.text) for segment in target_group)
    if source_annotations and target_annotations:
        source_text = " ".join(segment.text.casefold() for segment in source_group)
        target_text = " ".join(segment.text.casefold() for segment in target_group)
        return 0.0 if source_text == target_text else 0.25
    return 1.25 if source_annotations != target_annotations else 0.0


def _prefix_sums(values: list[int]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _time_range(transcript: ParsedTranscript) -> tuple[int, int] | None:
    values = [segment.start_ms for segment in transcript.segments if segment.start_ms is not None]
    if len(values) < 2 or values[-1] <= values[0]:
        return None
    return values[0], values[-1] - values[0]


def _rounded(value: float) -> float:
    return round(value, 6)
