"""Aligned-transcript records, loading, validation, statistics, and output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

from timelymt.data.parsing.core import ParsedSegment, ParsedTranscript, validate_parsed_transcript


SCHEMA_VERSION = "1.0.0"
ALIGNER_VERSION = "1.0.0"
ALIGNMENT_TYPES = ("1:1", "1:2", "2:1", "2:2", "1:3", "3:1", "other")


@dataclass(frozen=True)
class AlignmentUnit:
    alignment_id: str
    source_segment_ids: tuple[str, ...]
    target_segment_ids: tuple[str, ...]
    source_text: str
    target_text: str
    score: float
    features: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_segment_ids"] = list(self.source_segment_ids)
        value["target_segment_ids"] = list(self.target_segment_ids)
        value["features"] = dict(self.features)
        return value


@dataclass(frozen=True)
class AlignedTranscript:
    talk_id: str
    source_artifact: str
    target_artifact: str
    method: Mapping[str, Any]
    alignments: tuple[AlignmentUnit, ...]
    unaligned_source_segment_ids: tuple[str, ...]
    unaligned_target_segment_ids: tuple[str, ...]
    statistics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_language: str = "en"
    target_language: str = "vi"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "talk_id": self.talk_id,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "source_artifact": self.source_artifact,
            "target_artifact": self.target_artifact,
            "method": dict(self.method),
            "alignments": [unit.to_dict() for unit in self.alignments],
            "unaligned_source_segment_ids": list(self.unaligned_source_segment_ids),
            "unaligned_target_segment_ids": list(self.unaligned_target_segment_ids),
            "statistics": dict(self.statistics),
            "provenance": dict(self.provenance),
        }


def load_parsed_transcript(path: Path) -> ParsedTranscript:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        transcript = ParsedTranscript(
            talk_id=document["talk_id"],
            language=document["language"],
            provider=document["provider"],
            segmentation_method=document["segmentation"]["method"],
            segments=tuple(ParsedSegment(**segment) for segment in document["segments"]),
            provenance=document["provenance"],
            schema_version=document["schema_version"],
        )
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot load parsed transcript {path}: {error}") from error
    validate_parsed_transcript(transcript)
    return transcript


def alignment_statistics(
    source_count: int,
    target_count: int,
    alignments: Sequence[AlignmentUnit],
    unaligned_source_count: int,
    unaligned_target_count: int,
) -> dict[str, Any]:
    type_counts = {name: 0 for name in ALIGNMENT_TYPES}
    scores: list[float] = []
    for unit in alignments:
        alignment_type = f"{len(unit.source_segment_ids)}:{len(unit.target_segment_ids)}"
        type_counts[alignment_type if alignment_type in type_counts else "other"] += 1
        scores.append(unit.score)
    return {
        "source_segment_count": source_count,
        "target_segment_count": target_count,
        "alignment_unit_count": len(alignments),
        "alignment_type_counts": type_counts,
        "unaligned_source_count": unaligned_source_count,
        "unaligned_target_count": unaligned_target_count,
        "mean_score": _rounded(mean(scores)) if scores else None,
        "median_score": _rounded(median(scores)) if scores else None,
        "minimum_score": _rounded(min(scores)) if scores else None,
        "maximum_score": _rounded(max(scores)) if scores else None,
    }


def validate_aligned_transcript(
    aligned: AlignedTranscript,
    source: ParsedTranscript,
    target: ParsedTranscript,
) -> None:
    validate_parsed_transcript(source)
    validate_parsed_transcript(target)
    if source.talk_id != target.talk_id or aligned.talk_id != source.talk_id:
        raise ValueError("Source, target, and alignment talk IDs must match")
    if source.language != "en" or aligned.source_language != "en":
        raise ValueError("Alignment source language must be en")
    if target.language != "vi" or aligned.target_language != "vi":
        raise ValueError("Alignment target language must be vi")
    if not source.segments or not target.segments:
        raise ValueError("Cannot validate alignment for an empty transcript")

    source_by_id = {segment.segment_id: segment for segment in source.segments}
    target_by_id = {segment.segment_id: segment for segment in target.segments}
    aligned_source: list[str] = []
    aligned_target: list[str] = []
    alignment_ids: set[str] = set()
    previous_source = -1
    previous_target = -1
    for index, unit in enumerate(aligned.alignments):
        if unit.alignment_id != f"a-{index + 1:06d}" or unit.alignment_id in alignment_ids:
            raise ValueError(f"Invalid or duplicate alignment ID: {unit.alignment_id}")
        alignment_ids.add(unit.alignment_id)
        if not unit.source_segment_ids or not unit.target_segment_ids:
            raise ValueError(f"Alignment {unit.alignment_id} must be bilingual")
        try:
            source_segments = [source_by_id[identifier] for identifier in unit.source_segment_ids]
            target_segments = [target_by_id[identifier] for identifier in unit.target_segment_ids]
        except KeyError as error:
            raise ValueError(f"Unresolved segment reference: {error.args[0]}") from error
        source_indices = [segment.index for segment in source_segments]
        target_indices = [segment.index for segment in target_segments]
        if source_indices != list(range(source_indices[0], source_indices[0] + len(source_indices))):
            raise ValueError(f"Non-contiguous source group in {unit.alignment_id}")
        if target_indices != list(range(target_indices[0], target_indices[0] + len(target_indices))):
            raise ValueError(f"Non-contiguous target group in {unit.alignment_id}")
        if source_indices[0] <= previous_source or target_indices[0] <= previous_target:
            raise ValueError(f"Non-monotonic alignment order at {unit.alignment_id}")
        previous_source = source_indices[-1]
        previous_target = target_indices[-1]
        if unit.source_text != " ".join(segment.text for segment in source_segments):
            raise ValueError(f"Source text reconstruction mismatch in {unit.alignment_id}")
        if unit.target_text != " ".join(segment.text for segment in target_segments):
            raise ValueError(f"Target text reconstruction mismatch in {unit.alignment_id}")
        aligned_source.extend(unit.source_segment_ids)
        aligned_target.extend(unit.target_segment_ids)

    _validate_accounting(
        "source", source_by_id, aligned_source, list(aligned.unaligned_source_segment_ids)
    )
    _validate_accounting(
        "target", target_by_id, aligned_target, list(aligned.unaligned_target_segment_ids)
    )
    expected_statistics = alignment_statistics(
        len(source.segments),
        len(target.segments),
        aligned.alignments,
        len(aligned.unaligned_source_segment_ids),
        len(aligned.unaligned_target_segment_ids),
    )
    if dict(aligned.statistics) != expected_statistics:
        raise ValueError("Alignment statistics do not match alignment content")


def _validate_accounting(
    side: str,
    available: Mapping[str, ParsedSegment],
    aligned_ids: list[str],
    unaligned_ids: list[str],
) -> None:
    if len(aligned_ids) != len(set(aligned_ids)):
        raise ValueError(f"Duplicate aligned {side} segment reference")
    if len(unaligned_ids) != len(set(unaligned_ids)):
        raise ValueError(f"Duplicate unaligned {side} segment reference")
    unknown = (set(aligned_ids) | set(unaligned_ids)) - available.keys()
    if unknown:
        raise ValueError(f"Unresolved {side} segment reference: {sorted(unknown)[0]}")
    overlap = set(aligned_ids) & set(unaligned_ids)
    if overlap:
        raise ValueError(f"{side.capitalize()} segment both aligned and unaligned: {sorted(overlap)[0]}")
    if set(aligned_ids) | set(unaligned_ids) != set(available):
        raise ValueError(f"Not all {side} segments are accounted for")
    expected_unaligned = sorted(unaligned_ids, key=lambda identifier: available[identifier].index)
    if unaligned_ids != expected_unaligned:
        raise ValueError(f"Unaligned {side} IDs are not in transcript order")


def make_provenance(source_path: Path, target_path: Path) -> dict[str, str]:
    return {
        "source_checksum_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "target_checksum_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
        "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def write_alignment(path: Path, aligned: AlignedTranscript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(aligned.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_review(path: Path, alignments: Sequence[AlignmentUnit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("alignment_id", "source_ids", "target_ids", "source_text", "target_text", "score", "alignment_type")
        )
        for unit in alignments:
            writer.writerow(
                (
                    unit.alignment_id,
                    ",".join(unit.source_segment_ids),
                    ",".join(unit.target_segment_ids),
                    unit.source_text,
                    unit.target_text,
                    f"{unit.score:.6f}",
                    f"{len(unit.source_segment_ids)}:{len(unit.target_segment_ids)}",
                )
            )


def _rounded(value: float) -> float:
    return round(value, 6)
