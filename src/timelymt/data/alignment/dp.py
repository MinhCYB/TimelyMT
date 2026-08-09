"""Bounded monotonic dynamic programming for bilingual segment alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from timelymt.data.parsing.core import ParsedTranscript, validate_parsed_transcript

from .core import (
    ALIGNER_VERSION,
    AlignedTranscript,
    AlignmentUnit,
    alignment_statistics,
    make_provenance,
    validate_aligned_transcript,
)
from .scoring import GroupScorer


@dataclass(frozen=True)
class AlignmentParameters:
    max_group_size: int = 3
    skip_penalty: float = 1.6
    group_penalty: float = 0.65

    def validate(self) -> None:
        if self.max_group_size < 1 or self.max_group_size > 4:
            raise ValueError("max_group_size must be between 1 and 4")
        if not math.isfinite(self.skip_penalty) or self.skip_penalty <= 0:
            raise ValueError("skip_penalty must be a positive finite number")
        if not math.isfinite(self.group_penalty) or self.group_penalty < 0:
            raise ValueError("group_penalty must be a non-negative finite number")


@dataclass(frozen=True)
class _Step:
    source_size: int
    target_size: int
    cost: float
    features: dict[str, float | int | None] | None


def align_transcripts(
    source: ParsedTranscript,
    target: ParsedTranscript,
    *,
    source_path: Path,
    target_path: Path,
    parameters: AlignmentParameters = AlignmentParameters(),
) -> AlignedTranscript:
    parameters.validate()
    validate_parsed_transcript(source)
    validate_parsed_transcript(target)
    if source.talk_id != target.talk_id:
        raise ValueError("Source and target talk IDs must match")
    if source.language != "en":
        raise ValueError("Alignment source language must be en")
    if target.language != "vi":
        raise ValueError("Alignment target language must be vi")
    if not source.segments or not target.segments:
        raise ValueError("Cannot align an empty transcript")

    scorer = GroupScorer(source, target, group_penalty=parameters.group_penalty)
    source_count = len(source.segments)
    target_count = len(target.segments)
    costs = [[math.inf] * (target_count + 1) for _ in range(source_count + 1)]
    previous: list[list[tuple[int, int, _Step] | None]] = [
        [None] * (target_count + 1) for _ in range(source_count + 1)
    ]
    costs[0][0] = 0.0
    transitions = _transitions(parameters.max_group_size)
    for source_index in range(source_count + 1):
        for target_index in range(target_count + 1):
            if not math.isfinite(costs[source_index][target_index]):
                continue
            for source_size, target_size in transitions:
                next_source = source_index + source_size
                next_target = target_index + target_size
                if next_source > source_count or next_target > target_count:
                    continue
                if source_size == 0 or target_size == 0:
                    step = _Step(source_size, target_size, parameters.skip_penalty, None)
                else:
                    scored = scorer.score(source_index, source_size, target_index, target_size)
                    step = _Step(source_size, target_size, scored.cost, scored.features)
                candidate = costs[source_index][target_index] + step.cost
                if candidate < costs[next_source][next_target] - 1e-12:
                    costs[next_source][next_target] = candidate
                    previous[next_source][next_target] = (source_index, target_index, step)

    steps = _backtrack(previous, source_count, target_count)
    alignments: list[AlignmentUnit] = []
    unaligned_source: list[str] = []
    unaligned_target: list[str] = []
    source_index = 0
    target_index = 0
    for step in steps:
        source_group = source.segments[source_index : source_index + step.source_size]
        target_group = target.segments[target_index : target_index + step.target_size]
        if step.source_size == 0:
            unaligned_target.extend(segment.segment_id for segment in target_group)
        elif step.target_size == 0:
            unaligned_source.extend(segment.segment_id for segment in source_group)
        else:
            alignments.append(
                AlignmentUnit(
                    alignment_id=f"a-{len(alignments) + 1:06d}",
                    source_segment_ids=tuple(segment.segment_id for segment in source_group),
                    target_segment_ids=tuple(segment.segment_id for segment in target_group),
                    source_text=" ".join(segment.text for segment in source_group),
                    target_text=" ".join(segment.text for segment in target_group),
                    score=step.cost,
                    features=step.features or {},
                )
            )
        source_index += step.source_size
        target_index += step.target_size

    statistics = alignment_statistics(
        source_count, target_count, alignments, len(unaligned_source), len(unaligned_target)
    )
    aligned = AlignedTranscript(
        talk_id=source.talk_id,
        source_artifact=source_path.as_posix(),
        target_artifact=target_path.as_posix(),
        method={
            "name": "monotonic_length_position_dp",
            "version": ALIGNER_VERSION,
            "parameters": {
                "max_group_size": parameters.max_group_size,
                "skip_penalty": parameters.skip_penalty,
                "group_penalty": parameters.group_penalty,
            },
        },
        alignments=tuple(alignments),
        unaligned_source_segment_ids=tuple(unaligned_source),
        unaligned_target_segment_ids=tuple(unaligned_target),
        statistics=statistics,
        provenance=make_provenance(source_path, target_path),
    )
    validate_aligned_transcript(aligned, source, target)
    return aligned


def _transitions(max_group_size: int) -> tuple[tuple[int, int], ...]:
    bilingual = tuple(
        (source_size, target_size)
        for source_size in range(1, max_group_size + 1)
        for target_size in range(1, max_group_size + 1)
    )
    # Fixed ordering makes equal-cost paths deterministic and prefers simple bilingual steps.
    return tuple(sorted(bilingual, key=lambda sizes: (sum(sizes), abs(sizes[0] - sizes[1]), sizes))) + (
        (1, 0),
        (0, 1),
    )


def _backtrack(
    previous: list[list[tuple[int, int, _Step] | None]], source_count: int, target_count: int
) -> list[_Step]:
    steps: list[_Step] = []
    state = (source_count, target_count)
    while state != (0, 0):
        predecessor = previous[state[0]][state[1]]
        if predecessor is None:
            raise RuntimeError("No monotonic alignment path found")
        source_index, target_index, step = predecessor
        steps.append(step)
        state = (source_index, target_index)
    steps.reverse()
    return steps
