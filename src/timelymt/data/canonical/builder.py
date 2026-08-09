"""Provider-neutral assembly of a completed canonical streaming talk."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .core import SCHEMA_VERSION, load_json, validate_canonical_talk


BUILDER_NAME = "timelymt.data.canonical"
BUILDER_VERSION = "1.0.0"


def build_canonical_talk(talk_id: str, *, raw_root: Path = Path("data/streaming/raw"), parsed_root: Path = Path("data/streaming/parsed"), aligned_root: Path = Path("data/streaming/aligned"), timed_root: Path = Path("data/streaming/timed")) -> dict[str, Any]:
    paths = _artifact_paths(talk_id, raw_root, parsed_root, aligned_root, timed_root)
    metadata = load_json(paths["metadata"], "acquisition metadata")
    acquisition = load_json(paths["acquisition"], "acquisition record")
    parsed_source = load_json(paths["source"], "parsed source")
    parsed_target = load_json(paths["target"], "parsed target")
    aligned = load_json(paths["alignment"], "alignment")
    timed = load_json(paths["timed"], "timed source")
    _validate_upstream_identity(talk_id, acquisition, parsed_source, parsed_target, aligned, timed)
    _validate_timed_source(parsed_source, timed)
    _validate_alignment(parsed_source, parsed_target, aligned)
    candidate = _mapping(metadata.get("candidate"), "metadata.candidate")
    provider_metadata = metadata.get("provider_metadata")
    talk = {"talk_id": talk_id, "source_language": "en", "target_language": "vi", "provider": candidate.get("provider")}
    for key in ("title", "speaker", "domain", "source_url"):
        if isinstance(candidate.get(key), str) and candidate[key]:
            talk[key] = candidate[key]
    if isinstance(candidate.get("slug"), str) and candidate["slug"]:
        talk["original_source_id"] = candidate["slug"]
    duration_ms = _duration_ms(provider_metadata)
    if duration_ms is not None:
        talk["duration_ms"] = duration_ms
    source_segments = [{key: segment[key] for key in ("segment_id", "index", "text", "start_ms", "end_ms")} for segment in timed["segments"]]
    target_segments = []
    for segment in parsed_target["segments"]:
        value = {key: segment[key] for key in ("segment_id", "index", "text")}
        if segment.get("start_ms") is not None and segment.get("end_ms") is not None:
            value["start_ms"], value["end_ms"] = segment["start_ms"], segment["end_ms"]
        target_segments.append(value)
    alignments = [{"alignment_id": unit["alignment_id"], "source_segment_ids": unit["source_segment_ids"], "target_segment_ids": unit["target_segment_ids"], "method": aligned["method"]["name"], "metadata": {"method_version": aligned["method"]["version"], "method_parameters": aligned["method"]["parameters"]}} for unit in aligned["alignments"]]
    tokens = [{"token_id": token["token_id"], "index": token["global_index"], "text": token["text"], "source_segment_id": token["source_segment_id"], "segment_index": token["segment_index"], "emit_ms": token["emit_ms"]} for segment in timed["segments"] for token in segment["tokens"]]
    document = {"schema_version": SCHEMA_VERSION, "talk": talk, "source": {"language": "en", "segments": source_segments}, "target_reference": {"language": "vi", "segments": target_segments}, "alignments": alignments, "stream": {"timing_mode": timed["timing"]["mode"], "timing_parameters": timed["timing"]["parameters"], "tokens": tokens}, "provenance": _provenance(candidate, acquisition, parsed_source, parsed_target, aligned, timed, paths)}
    validate_canonical_talk(document)
    return document


def _artifact_paths(talk_id: str, raw_root: Path, parsed_root: Path, aligned_root: Path, timed_root: Path) -> dict[str, Path]:
    raw_candidates = sorted(path for path in raw_root.glob(f"*/{talk_id}") if path.is_dir())
    if len(raw_candidates) != 1:
        raise ValueError(f"Expected exactly one raw acquisition directory for {talk_id}, found {len(raw_candidates)}")
    raw = raw_candidates[0]
    paths = {"metadata": raw / "metadata.json", "acquisition": raw / "acquisition.json", "source": parsed_root / talk_id / "source.en.json", "target": parsed_root / talk_id / "target.vi.json", "alignment": aligned_root / talk_id / "alignment.json", "timed": timed_root / talk_id / "source.en.json"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required upstream artifact: {missing[0]}")
    return paths


def _validate_upstream_identity(talk_id: str, acquisition: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any], aligned: Mapping[str, Any], timed: Mapping[str, Any]) -> None:
    identifiers = [acquisition.get("candidate_id"), source.get("talk_id"), target.get("talk_id"), aligned.get("talk_id"), timed.get("talk_id")]
    if any(identifier != talk_id for identifier in identifiers):
        raise ValueError("Talk IDs must match across acquisition, parsed, alignment, and timing artifacts")
    if source.get("language") != "en" or target.get("language") != "vi" or aligned.get("source_language") != "en" or aligned.get("target_language") != "vi" or timed.get("language") != "en":
        raise ValueError("Upstream artifact languages must be English source and Vietnamese target")


def _validate_timed_source(source: Mapping[str, Any], timed: Mapping[str, Any]) -> None:
    parsed_segments, timed_segments = source.get("segments"), timed.get("segments")
    if not isinstance(parsed_segments, list) or not isinstance(timed_segments, list) or len(parsed_segments) != len(timed_segments):
        raise ValueError("Timed source must preserve every parsed source segment")
    for index, (parsed, finalized) in enumerate(zip(parsed_segments, timed_segments)):
        if (parsed.get("segment_id"), parsed.get("index"), parsed.get("text")) != (finalized.get("segment_id"), index, finalized.get("text")):
            raise ValueError(f"Timed source segment IDs, order, or text differ from parsed source at index {index}")


def _validate_alignment(source: Mapping[str, Any], target: Mapping[str, Any], aligned: Mapping[str, Any]) -> None:
    source_by_id = {segment["segment_id"]: segment for segment in source["segments"]}
    target_by_id = {segment["segment_id"]: segment for segment in target["segments"]}
    used_source: set[str] = set()
    used_target: set[str] = set()
    for unit in aligned.get("alignments", []):
        source_ids, target_ids = unit.get("source_segment_ids", []), unit.get("target_segment_ids", [])
        try:
            source_text = " ".join(source_by_id[item]["text"] for item in source_ids)
            target_text = " ".join(target_by_id[item]["text"] for item in target_ids)
        except KeyError as error:
            raise ValueError(f"Invalid alignment reference: {error.args[0]}") from error
        if source_text != unit.get("source_text") or target_text != unit.get("target_text"):
            raise ValueError(f"Alignment text reconstruction mismatch: {unit.get('alignment_id')}")
        if used_source.intersection(source_ids) or used_target.intersection(target_ids):
            raise ValueError("Alignment reuses a source or target segment")
        used_source.update(source_ids)
        used_target.update(target_ids)
    unaligned_source = set(aligned.get("unaligned_source_segment_ids", []))
    unaligned_target = set(aligned.get("unaligned_target_segment_ids", []))
    if used_source | unaligned_source != set(source_by_id) or used_target | unaligned_target != set(target_by_id):
        raise ValueError("Alignment does not account for every source and target segment")


def _provenance(candidate: Mapping[str, Any], acquisition: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any], aligned: Mapping[str, Any], timed: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    acquired_at = acquisition.get("acquired_at")
    return {"provider": candidate.get("provider"), "source_identifier": candidate.get("slug", candidate.get("id")), "acquired_on": acquired_at[:10] if isinstance(acquired_at, str) else None, "parser": {"name": source["provenance"]["parser_name"], "version": source["provenance"]["parser_version"]}, "alignment": {"name": aligned["method"]["name"], "version": aligned["method"]["version"], "metadata": {"parameters": aligned["method"]["parameters"]}}, "timing": {"name": timed["provenance"]["timing_tool"], "version": timed["provenance"]["timing_version"], "metadata": {"mode": timed["timing"]["mode"], "parameters": timed["timing"]["parameters"]}}, "processing_version": BUILDER_VERSION, "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"), "metadata": {"builder": BUILDER_NAME, "artifacts": {name: {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in paths.items()}, "parsed_checksums": {"source": source["provenance"]["source_checksum_sha256"], "target": target["provenance"]["source_checksum_sha256"]}, "unaligned_source_segment_ids": aligned["unaligned_source_segment_ids"], "unaligned_target_segment_ids": aligned["unaligned_target_segment_ids"]}}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object for {name}")
    return value


def _duration_ms(value: Any) -> int | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("duration"), str):
        return None
    duration = value["duration"]
    import re
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return None
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return (hours * 3600 + minutes * 60 + seconds) * 1000


__all__ = ["BUILDER_NAME", "BUILDER_VERSION", "build_canonical_talk"]
