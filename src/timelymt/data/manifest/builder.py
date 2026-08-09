"""Provider-neutral dataset manifest construction from canonical talk artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from timelymt.data.canonical.core import canonical_content_checksum, load_canonical_talk

from .core import SCHEMA_VERSION, validate_dataset_manifest


BUILDER_NAME = "timelymt.data.manifest"
BUILDER_VERSION = "1.0.0"
CALIBRATION_TALK_IDS = {
    "ted-alona-fyshe-ai-understand",
    "ted-jeff-dean-ai-smart",
    "ted-yejin-choi-ai-smart-stupid",
}


def build_dataset_manifest(canonical_paths: Iterable[Path] | None = None, *, processed_root: Path = Path("data/streaming/processed")) -> dict[str, Any]:
    paths = list(canonical_paths) if canonical_paths is not None else sorted(processed_root.glob("*/streaming-talk.json"))
    entries = [_manifest_entry(path, processed_root) for path in paths]
    entries.sort(key=lambda entry: entry["talk_id"])
    if len({entry["talk_id"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate canonical talk IDs cannot enter the dataset manifest")
    document = {"schema_version": SCHEMA_VERSION, "dataset": {"name": "TimelyMT", "source_language": "en", "target_language": "vi", "status": "pilot" if entries and all(entry["cohort"] == "pilot" for entry in entries) else "unassigned"}, "talks": entries, "provenance": {"builder": BUILDER_NAME, "builder_version": BUILDER_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")}}
    validate_dataset_manifest(document)
    return document


def _manifest_entry(path: Path, processed_root: Path) -> dict[str, Any]:
    document = load_canonical_talk(path)
    talk = document["talk"]
    talk_id = talk["talk_id"]
    if path.name != "streaming-talk.json" or path.parent.name != talk_id:
        raise ValueError(f"Canonical path identity does not match talk_id {talk_id}: {path}")
    try:
        canonical_path = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        canonical_path = path.relative_to(processed_root.parent.parent).as_posix()
    source_segments = document["source"]["segments"]
    statistics = {"source_segments": len(source_segments), "target_segments": len(document["target_reference"]["segments"]), "alignment_units": len(document["alignments"]), "stream_tokens": len(document["stream"]["tokens"]), "source_clock_duration_ms": source_segments[-1]["end_ms"] - source_segments[0]["start_ms"]}
    entry = {"talk_id": talk_id, "canonical_path": canonical_path, "content_checksum": canonical_content_checksum(document), "timing_mode": document["stream"]["timing_mode"], "statistics": statistics, "cohort": "pilot" if talk_id in CALIBRATION_TALK_IDS else "unassigned"}
    for field in ("speaker", "domain", "provider", "topics"):
        if field in talk:
            entry[field] = talk[field]
    return entry
