"""Dataset-level acceptance, diagnostics, aggregate statistics, and snapshots."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from timelymt.data.canonical.core import canonical_content_checksum, load_canonical_talk
from timelymt.data.manifest.core import dataset_manifest_checksum, validate_dataset_manifest


def qa_flags(document: Mapping[str, Any], alignment: Mapping[str, Any] | None = None) -> list[str]:
    source = document["source"]["segments"]
    target = document["target_reference"]["segments"]
    alignments = document["alignments"]
    tokens = document["stream"]["tokens"]
    duration = source[-1]["end_ms"] - source[0]["start_ms"]
    flags: list[str] = []
    ratio = len(source) / len(target)
    if ratio < 0.6 or ratio > 1.67:
        flags.append("segment_count_ratio")
    large = sum(max(len(unit["source_segment_ids"]), len(unit["target_segment_ids"])) >= 3 for unit in alignments)
    if alignments and large / len(alignments) > 0.1:
        flags.append("frequent_3n_mappings")
    if duration < 120_000:
        flags.append("extremely_short_talk")
    if duration > 3_600_000:
        flags.append("extremely_long_talk")
    if not tokens:
        flags.append("zero_lexical_tokens")
    talk = document["talk"]
    if any(not talk.get(field) for field in ("speaker", "domain", "provider")):
        flags.append("missing_metadata")
    if alignment:
        statistics = alignment.get("statistics", {})
        unaligned = statistics.get("unaligned_source_count", 0) + statistics.get("unaligned_target_count", 0)
        if unaligned / max(1, len(source) + len(target)) > 0.05:
            flags.append("high_unaligned_rate")
        if statistics.get("maximum_score") is not None and statistics["maximum_score"] > 3.0:
            flags.append("very_high_alignment_cost")
    return flags


def inspect_talk(canonical_path: Path, *, aligned_root: Path = Path("data/streaming/aligned")) -> dict[str, Any]:
    document = load_canonical_talk(canonical_path)
    talk_id = document["talk"]["talk_id"]
    alignment_path = aligned_root / talk_id / "alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8")) if alignment_path.is_file() else None
    source = document["source"]["segments"]
    target = document["target_reference"]["segments"]
    units = document["alignments"]
    type_distribution: dict[str, int] = {}
    for unit in units:
        key = f"{len(unit['source_segment_ids'])}:{len(unit['target_segment_ids'])}"
        type_distribution[key] = type_distribution.get(key, 0) + 1
    statistics = alignment.get("statistics", {}) if alignment else {}
    return {
        "talk_id": talk_id,
        "speaker": document["talk"].get("speaker"),
        "domain": document["talk"].get("domain"),
        "provider": document["talk"].get("provider"),
        "timing_mode": document["stream"]["timing_mode"],
        "source_segments": len(source),
        "target_segments": len(target),
        "alignment_units": len(units),
        "stream_tokens": len(document["stream"]["tokens"]),
        "duration_ms": source[-1]["end_ms"] - source[0]["start_ms"],
        "alignment_type_distribution": dict(sorted(type_distribution.items())),
        "unaligned_source_count": statistics.get("unaligned_source_count", 0),
        "unaligned_target_count": statistics.get("unaligned_target_count", 0),
        "mean_alignment_cost": statistics.get("mean_score"),
        "maximum_alignment_cost": statistics.get("maximum_score"),
        "qa_flags": qa_flags(document, alignment),
        "processing_status": "accepted",
        "content_checksum": canonical_content_checksum(document),
    }


def build_quality_report(
    canonical_paths: Sequence[Path],
    *,
    failed_records: Sequence[Mapping[str, Any]] = (),
    aligned_root: Path = Path("data/streaming/aligned"),
) -> dict[str, Any]:
    talks = [inspect_talk(path, aligned_root=aligned_root) for path in sorted(canonical_paths)]
    totals = {
        field: sum(talk[field] for talk in talks)
        for field in ("source_segments", "target_segments", "alignment_units", "stream_tokens", "duration_ms")
    }
    return {
        "version": "1.0.0",
        "generated_at": _timestamp(),
        "summary": {"accepted": len(talks), "failed_or_rejected": len(failed_records), **totals},
        "talks": talks,
        "failed_or_rejected": list(failed_records),
        "cost_note": "Alignment cost is a structural optimization diagnostic, not confidence.",
    }


def validate_dataset(manifest: Mapping[str, Any], *, project_root: Path = Path.cwd()) -> dict[str, Any]:
    validate_dataset_manifest(manifest)
    paths: set[str] = set()
    checksums: list[str] = []
    for entry in manifest["talks"]:
        path_value = entry["canonical_path"]
        if path_value in paths:
            raise ValueError(f"Duplicate canonical path: {path_value}")
        paths.add(path_value)
        path = Path(path_value)
        if not path.is_absolute():
            path = project_root / path
        document = load_canonical_talk(path)
        if document["talk"]["talk_id"] != entry["talk_id"]:
            raise ValueError(f"Canonical talk ID mismatch: {entry['talk_id']}")
        checksum = canonical_content_checksum(document)
        if checksum != entry["content_checksum"]:
            raise ValueError(f"Canonical checksum mismatch: {entry['talk_id']}")
        checksums.append(checksum)
    return {
        "talk_count": len(manifest["talks"]),
        "manifest_checksum": dataset_manifest_checksum(manifest),
        "canonical_checksums_resolved": len(checksums),
        "speaker_metadata_complete": all(talk.get("speaker") for talk in manifest["talks"]),
    }


def build_snapshot(
    manifest: Mapping[str, Any],
    *,
    split: Mapping[str, Any] | None,
    alignment_config: Mapping[str, Any],
    known_limitations: Sequence[str],
) -> dict[str, Any]:
    talks = manifest["talks"]
    timing_distribution = _distribution(talks, "timing_mode")
    provider_distribution = _distribution(talks, "provider")
    split_counts = {name: len(split["splits"][name]) for name in ("train", "dev", "test")} if split else None
    return {
        "snapshot_version": "1.0.0",
        "dataset_name": "TimelyMT Streaming Dataset v1",
        "created_at": _timestamp(),
        "manifest_checksum": dataset_manifest_checksum(manifest),
        "split_manifest_checksum": stable_checksum(split) if split else None,
        "talk_count": len(talks),
        "source_language": "en",
        "target_language": "vi",
        "alignment_config_version": alignment_config["version"],
        "alignment_parameters": {key: alignment_config[key] for key in ("max_group_size", "skip_penalty", "group_penalty")},
        "timing_modes": timing_distribution,
        "provider_distribution": provider_distribution,
        "split_counts": split_counts,
        "known_limitations": list(known_limitations),
    }


def stable_checksum(document: Mapping[str, Any] | None) -> str | None:
    if document is None:
        return None
    stable = dict(copy.deepcopy(document))
    stable.pop("created_at", None)
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _distribution(talks: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for talk in talks:
        key = str(talk.get(field, "unknown"))
        values[key] = values.get(key, 0) + 1
    return dict(sorted(values.items()))
