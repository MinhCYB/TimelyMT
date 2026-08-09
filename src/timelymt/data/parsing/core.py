"""Shared parsed-transcript records, normalization, validation, and writing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from html import unescape
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol


SCHEMA_VERSION = "1.0.0"
PARSER_VERSION = "1.0.0"
SUPPORTED_LANGUAGES = {"en", "vi"}
TIMING_SOURCES = {"none", "wit3_seekvideo", "original_caption", "other"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


@dataclass(frozen=True)
class ParsedSegment:
    segment_id: str
    index: int
    text: str
    start_ms: int | None
    end_ms: int | None
    timing_source: str


@dataclass(frozen=True)
class ParsedTranscript:
    talk_id: str
    language: str
    provider: str
    segmentation_method: str
    segments: tuple[ParsedSegment, ...]
    provenance: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "talk_id": self.talk_id,
            "language": self.language,
            "provider": self.provider,
            "segmentation": {"method": self.segmentation_method},
            "segments": [asdict(segment) for segment in self.segments],
            "provenance": dict(self.provenance),
        }


class TranscriptParser(Protocol):
    provider: str

    def parse(self, input_path: Path, *, talk_id: str | None, language: str) -> ParsedTranscript:
        """Parse one language of one talk into the shared representation."""

        ...


def normalize_text(text: str) -> str:
    """Decode entities and collapse formatting whitespace without changing linguistic content."""
    return re.sub(r"\s+", " ", unescape(text.replace("\r\n", "\n").replace("\r", "\n"))).strip()


def make_segments(
    texts: list[str],
    language: str,
    *,
    starts_ms: list[int | None] | None = None,
    timing_source: str,
) -> tuple[ParsedSegment, ...]:
    starts = starts_ms if starts_ms is not None else [None] * len(texts)
    if len(starts) != len(texts):
        raise ValueError("Segment text and timestamp counts differ")
    return tuple(
        ParsedSegment(
            segment_id=f"{language}-{index + 1:06d}",
            index=index,
            text=text,
            start_ms=starts[index],
            end_ms=None,
            timing_source=timing_source,
        )
        for index, text in enumerate(texts)
    )


def make_provenance(
    input_path: Path,
    raw_bytes: bytes,
    parser_name: str,
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "raw_input_path": input_path.as_posix(),
        "source_checksum_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "parser_name": parser_name,
        "parser_version": PARSER_VERSION,
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    if source_metadata:
        provenance["source_metadata"] = dict(source_metadata)
    return provenance


def validate_parsed_transcript(transcript: ParsedTranscript) -> None:
    if transcript.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported parsed transcript schema version: {transcript.schema_version!r}")
    if not IDENTIFIER.fullmatch(transcript.talk_id):
        raise ValueError(f"Invalid or missing talk_id: {transcript.talk_id!r}")
    if transcript.language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {transcript.language!r}")
    if not transcript.provider or not transcript.segmentation_method:
        raise ValueError("Provider and segmentation method must be non-empty")
    if not transcript.segments:
        raise ValueError("Parsed transcript contains no segments")

    seen: set[str] = set()
    for expected_index, segment in enumerate(transcript.segments):
        if segment.segment_id in seen:
            raise ValueError(f"Duplicate segment_id: {segment.segment_id}")
        seen.add(segment.segment_id)
        if segment.segment_id != f"{transcript.language}-{expected_index + 1:06d}":
            raise ValueError(f"Non-deterministic segment_id at index {expected_index}")
        if segment.index != expected_index:
            raise ValueError(f"Non-contiguous segment index: {segment.index}")
        if not segment.text.strip():
            raise ValueError(f"Empty segment text at index {expected_index}")
        if segment.timing_source not in TIMING_SOURCES:
            raise ValueError(f"Unsupported timing_source: {segment.timing_source!r}")
        for name, value in (("start_ms", segment.start_ms), ("end_ms", segment.end_ms)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"Invalid {name} at index {expected_index}: {value!r}")
        if segment.start_ms is not None and segment.end_ms is not None and segment.end_ms < segment.start_ms:
            raise ValueError(f"end_ms precedes start_ms at index {expected_index}")


def write_parsed_transcript(path: Path, transcript: ParsedTranscript) -> None:
    validate_parsed_transcript(transcript)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
