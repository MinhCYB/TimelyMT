"""Resumable composition of the existing per-talk data stages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from timelymt.data.acquisition.core import Candidate, SourceAdapter, acquire_candidates
from timelymt.data.alignment.core import load_parsed_transcript, write_alignment, write_review
from timelymt.data.alignment.dp import AlignmentParameters, align_transcripts
from timelymt.data.canonical.builder import build_canonical_talk
from timelymt.data.canonical.core import canonical_content_checksum, validate_canonical_talk, write_canonical_talk
from timelymt.data.parsing.core import write_parsed_transcript
from timelymt.data.parsing.ted import TedContinuousTranscriptParser
from timelymt.data.timing.core import build_timed_source, write_timed_source


STAGES = ("acquire", "parse", "align", "time", "canonical")


@dataclass(frozen=True)
class PipelinePaths:
    raw_root: Path = Path("data/streaming/raw")
    parsed_root: Path = Path("data/streaming/parsed")
    aligned_root: Path = Path("data/streaming/aligned")
    timed_root: Path = Path("data/streaming/timed")
    processed_root: Path = Path("data/streaming/processed")
    acquisition_results: Path = Path("data/manifests/acquisition-results.jsonl")


def process_talk(
    candidate: Candidate,
    *,
    adapters: Mapping[str, SourceAdapter],
    paths: PipelinePaths = PipelinePaths(),
    alignment_parameters: AlignmentParameters = AlignmentParameters(),
    resume: bool = True,
    force_stage: str | None = None,
) -> dict[str, Any]:
    if force_stage is not None and force_stage not in STAGES:
        raise ValueError(f"force_stage must be one of {', '.join(STAGES)}")
    record: dict[str, Any] = {
        "talk_id": candidate.id,
        "provider": candidate.provider,
        "status": "processing",
        "stages": {},
        "updated_at": _timestamp(),
    }
    force_index = STAGES.index(force_stage) if force_stage else len(STAGES)
    try:
        for index, stage in enumerate(STAGES):
            output_paths = _stage_outputs(stage, candidate, paths)
            complete = all(path.is_file() for path in output_paths)
            should_run = index >= force_index or not (resume and complete)
            if should_run:
                _run_stage(stage, candidate, adapters, paths, alignment_parameters)
                record["stages"][stage] = "completed"
            else:
                record["stages"][stage] = "resumed"
        canonical_path = paths.processed_root / candidate.id / "streaming-talk.json"
        document = json.loads(canonical_path.read_text(encoding="utf-8"))
        validate_canonical_talk(document)
        record["status"] = "accepted"
        record["canonical_path"] = canonical_path.as_posix()
        record["content_checksum"] = canonical_content_checksum(document)
    except Exception as error:  # Continue safely at the next talk.
        record["status"] = "failed"
        record["failed_stage"] = next(
            (stage for stage in STAGES if stage not in record["stages"]),
            STAGES[-1],
        )
        record["failure_reason"] = f"{type(error).__name__}: {error}"
    record["updated_at"] = _timestamp()
    return record


def prepare_dataset(
    candidates: Sequence[Candidate],
    *,
    adapters: Mapping[str, SourceAdapter],
    paths: PipelinePaths = PipelinePaths(),
    alignment_parameters: AlignmentParameters = AlignmentParameters(),
    resume: bool = True,
    force_stage: str | None = None,
    on_result: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        result = process_talk(
            candidate,
            adapters=adapters,
            paths=paths,
            alignment_parameters=alignment_parameters,
            resume=resume,
            force_stage=force_stage,
        )
        records.append(result)
        if on_result:
            on_result(result)
    return records


def write_pipeline_results(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": "1.0.0", "talks": list(records)}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_stage(
    stage: str,
    candidate: Candidate,
    adapters: Mapping[str, SourceAdapter],
    paths: PipelinePaths,
    parameters: AlignmentParameters,
) -> None:
    talk_id = candidate.id
    if stage == "acquire":
        result = acquire_candidates(
            [candidate], adapters, paths.raw_root, paths.acquisition_results, skip_existing=False
        )[0]
        if result.status != "available":
            raise ValueError(f"Bilingual acquisition status is {result.status}")
        return
    if stage == "parse":
        if candidate.provider != "ted":
            raise ValueError(f"Batch parsing is not configured for provider {candidate.provider}")
        parser = TedContinuousTranscriptParser()
        raw = paths.raw_root / candidate.provider / talk_id
        for language, filename in (("en", "source.en.txt"), ("vi", "target.vi.txt")):
            transcript = parser.parse(raw / filename, talk_id=talk_id, language=language)
            output = paths.parsed_root / talk_id / ("source.en.json" if language == "en" else "target.vi.json")
            write_parsed_transcript(output, transcript)
        return
    source_path = paths.parsed_root / talk_id / "source.en.json"
    if stage == "align":
        target_path = paths.parsed_root / talk_id / "target.vi.json"
        aligned = align_transcripts(
            load_parsed_transcript(source_path),
            load_parsed_transcript(target_path),
            source_path=source_path,
            target_path=target_path,
            parameters=parameters,
        )
        output = paths.aligned_root / talk_id
        write_alignment(output / "alignment.json", aligned)
        write_review(output / "review.tsv", aligned.alignments)
        return
    if stage == "time":
        timed = build_timed_source(load_parsed_transcript(source_path), source_path=source_path)
        write_timed_source(paths.timed_root / talk_id / "source.en.json", timed)
        return
    document = build_canonical_talk(
        talk_id,
        raw_root=paths.raw_root,
        parsed_root=paths.parsed_root,
        aligned_root=paths.aligned_root,
        timed_root=paths.timed_root,
    )
    write_canonical_talk(paths.processed_root / talk_id / "streaming-talk.json", document)


def _stage_outputs(stage: str, candidate: Candidate, paths: PipelinePaths) -> tuple[Path, ...]:
    talk_id = candidate.id
    raw = paths.raw_root / candidate.provider / talk_id
    return {
        "acquire": (raw / "acquisition.json", raw / "source.en.txt", raw / "target.vi.txt"),
        "parse": (paths.parsed_root / talk_id / "source.en.json", paths.parsed_root / talk_id / "target.vi.json"),
        "align": (paths.aligned_root / talk_id / "alignment.json",),
        "time": (paths.timed_root / talk_id / "source.en.json",),
        "canonical": (paths.processed_root / talk_id / "streaming-talk.json",),
    }[stage]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
