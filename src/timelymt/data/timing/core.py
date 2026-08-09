"""Timed-source records, construction, semantic validation, and serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from timelymt.data.alignment.core import load_parsed_transcript
from timelymt.data.parsing.core import ParsedTranscript, validate_parsed_transcript

from .recovery import detect_timing_mode, recover_intervals, simulated_intervals
from .simulation import ALLOCATION_MODES, allocate_emit_times, simulated_duration_ms
from .tokenization import lexical_tokens


SCHEMA_VERSION = "1.0.0"
TIMING_VERSION = "1.0.0"


@dataclass(frozen=True)
class TimedToken:
    token_id: str
    global_index: int
    segment_index: int
    source_segment_id: str
    text: str
    emit_ms: int


@dataclass(frozen=True)
class TimedSegment:
    segment_id: str
    index: int
    text: str
    start_ms: int
    end_ms: int
    tokens: tuple[TimedToken, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tokens"] = [asdict(token) for token in self.tokens]
        return value


@dataclass(frozen=True)
class TimedSource:
    talk_id: str
    timing: Mapping[str, Any]
    segments: tuple[TimedSegment, ...]
    statistics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    language: str = "en"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "talk_id": self.talk_id,
            "language": self.language,
            "timing": dict(self.timing),
            "segments": [segment.to_dict() for segment in self.segments],
            "statistics": dict(self.statistics),
            "provenance": dict(self.provenance),
        }


def build_timed_source(
    source: ParsedTranscript,
    *,
    source_path: Path,
    words_per_second: float = 2.5,
    allocation: str = "character_weighted",
) -> TimedSource:
    validate_parsed_transcript(source)
    if source.language != "en":
        raise ValueError("Timed source language must be en")
    if allocation not in ALLOCATION_MODES:
        raise ValueError(f"Unsupported allocation mode: {allocation!r}")
    simulated_duration_ms(0, words_per_second)

    segment_tokens = [lexical_tokens(segment.text) for segment in source.segments]
    token_counts = [len(tokens) for tokens in segment_tokens]
    mode = detect_timing_mode(source)
    if mode == "simulated":
        intervals = simulated_intervals(token_counts, words_per_second)
        original_timing_source = None
        final_fallback = "not_applicable"
    else:
        timing_sources = {segment.timing_source for segment in source.segments}
        if len(timing_sources) != 1 or "none" in timing_sources:
            raise ValueError("Recovered timing requires one authoritative source timing type")
        metadata = source.provenance.get("source_metadata")
        intervals, final_fallback = recover_intervals(
            source.segments,
            token_counts,
            words_per_second,
            metadata if isinstance(metadata, Mapping) else None,
        )
        original_timing_source = next(iter(timing_sources))

    timed_segments: list[TimedSegment] = []
    global_index = 0
    for parsed, tokens, (start_ms, end_ms) in zip(source.segments, segment_tokens, intervals):
        emit_times = allocate_emit_times(tokens, start_ms, end_ms, allocation)
        timed_tokens = tuple(
            TimedToken(
                token_id=f"tok-{global_index + offset + 1:06d}",
                global_index=global_index + offset,
                segment_index=offset,
                source_segment_id=parsed.segment_id,
                text=token,
                emit_ms=emit_times[offset],
            )
            for offset, token in enumerate(tokens)
        )
        global_index += len(timed_tokens)
        timed_segments.append(
            TimedSegment(parsed.segment_id, parsed.index, parsed.text, start_ms, end_ms, timed_tokens)
        )

    timed = TimedSource(
        talk_id=source.talk_id,
        timing={
            "mode": mode,
            "parameters": {
                "words_per_second": words_per_second,
                "allocation": allocation,
                "original_timing_source": original_timing_source,
                "final_segment_fallback": final_fallback,
            },
        },
        segments=tuple(timed_segments),
        statistics=timing_statistics(timed_segments),
        provenance={
            "source_artifact": source_path.as_posix(),
            "source_checksum_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "timing_tool": "timelymt.data.timing",
            "timing_version": TIMING_VERSION,
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        },
    )
    validate_timed_source(timed, source)
    return timed


def timing_statistics(segments: Sequence[TimedSegment]) -> dict[str, Any]:
    token_count = sum(len(segment.tokens) for segment in segments)
    start_ms = segments[0].start_ms
    end_ms = segments[-1].end_ms
    duration_ms = end_ms - start_ms
    return {
        "segment_count": len(segments),
        "token_count": token_count,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "mean_tokens_per_segment": round(mean(len(segment.tokens) for segment in segments), 6),
        "effective_tokens_per_second": round(token_count * 1000 / duration_ms, 6) if duration_ms else None,
    }


def validate_timed_source(timed: TimedSource, source: ParsedTranscript) -> None:
    validate_parsed_transcript(source)
    if timed.schema_version != SCHEMA_VERSION or not timed.talk_id:
        raise ValueError("Invalid timed-source identity or schema version")
    if timed.talk_id != source.talk_id or timed.language != "en" or source.language != "en":
        raise ValueError("Timed source must match an English parsed transcript")
    if not timed.segments or len(timed.segments) != len(source.segments):
        raise ValueError("Timed source must preserve all parsed segments")

    token_ids: set[str] = set()
    expected_global = 0
    previous_emit = -1
    for expected_index, (segment, parsed) in enumerate(zip(timed.segments, source.segments)):
        if (segment.segment_id, segment.index, segment.text) != (parsed.segment_id, expected_index, parsed.text):
            raise ValueError(f"Parsed segment identity, order, or text changed at index {expected_index}")
        if segment.start_ms < 0 or segment.end_ms < segment.start_ms:
            raise ValueError(f"Invalid timed interval at segment {segment.segment_id}")
        expected_texts = lexical_tokens(parsed.text)
        if [token.text for token in segment.tokens] != expected_texts:
            raise ValueError(f"Token reconstruction mismatch at segment {segment.segment_id}")
        for expected_segment_index, token in enumerate(segment.tokens):
            if not token.text or token.token_id in token_ids:
                raise ValueError("Empty lexical token or duplicate token ID")
            token_ids.add(token.token_id)
            if token.token_id != f"tok-{expected_global + 1:06d}":
                raise ValueError("Non-deterministic token ID")
            if token.global_index != expected_global or token.segment_index != expected_segment_index:
                raise ValueError("Non-contiguous token index")
            if token.source_segment_id != segment.segment_id:
                raise ValueError("Unresolved token source segment reference")
            if not segment.start_ms <= token.emit_ms <= segment.end_ms:
                raise ValueError("Token emit_ms lies outside its segment")
            if token.emit_ms < previous_emit:
                raise ValueError("Token emit_ms values are not globally monotonic")
            previous_emit = token.emit_ms
            expected_global += 1
        if segment.tokens and segment.tokens[-1].emit_ms != segment.end_ms:
            raise ValueError("Final token does not reach segment end_ms")
    if dict(timed.statistics) != timing_statistics(timed.segments):
        raise ValueError("Timing statistics do not match content")


def write_timed_source(path: Path, timed: TimedSource) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_timed_source(timed), encoding="utf-8", newline="\n")


def serialize_timed_source(timed: TimedSource) -> str:
    return json.dumps(timed.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "TimedSegment",
    "TimedSource",
    "TimedToken",
    "build_timed_source",
    "load_parsed_transcript",
    "serialize_timed_source",
    "validate_timed_source",
    "write_timed_source",
]
