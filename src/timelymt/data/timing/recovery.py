"""Source caption boundary recovery and timing-mode detection."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript

from .simulation import simulated_duration_ms


def detect_timing_mode(source: ParsedTranscript) -> str:
    starts = [segment.start_ms for segment in source.segments]
    if all(start is None for start in starts):
        return "simulated"
    if all(start is not None for start in starts):
        return "recovered_from_caption_starts"
    raise ValueError("Source has partially populated start_ms values")


def recover_intervals(
    segments: Sequence[ParsedSegment],
    token_counts: Sequence[int],
    words_per_second: float,
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[int, int]], str]:
    if len(segments) != len(token_counts):
        raise ValueError("Segment and token counts differ")
    starts: list[int] = []
    for segment in segments:
        if segment.start_ms is None:
            raise ValueError("Recovered timing requires start_ms for every segment")
        starts.append(segment.start_ms)
    for current, following in zip(starts, starts[1:]):
        if following < current:
            raise ValueError("Source caption start_ms values move backward")

    final_start = starts[-1]
    metadata_duration = _metadata_duration_ms(source_metadata)
    if metadata_duration is not None and metadata_duration >= final_start:
        final_end = metadata_duration
        fallback = "source_metadata_duration"
    else:
        final_end = final_start + simulated_duration_ms(token_counts[-1], words_per_second)
        fallback = "speech_rate_estimate"
    ends = starts[1:] + [final_end]
    return list(zip(starts, ends)), fallback


def simulated_intervals(
    token_counts: Sequence[int],
    words_per_second: float,
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    current = 0
    for token_count in token_counts:
        following = current + simulated_duration_ms(token_count, words_per_second)
        intervals.append((current, following))
        current = following
    return intervals


def _metadata_duration_ms(source_metadata: Mapping[str, Any] | None) -> int | None:
    if not source_metadata:
        return None
    for key in ("duration_ms", "talk_duration_ms"):
        value = source_metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None
