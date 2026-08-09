"""Canonical talk serialization and cross-artifact semantic validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from timelymt.data.timing.tokenization import lexical_tokens


SCHEMA_VERSION = "1.0.0"


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def load_canonical_talk(path: Path) -> dict[str, Any]:
    document = load_json(path, "canonical talk")
    validate_canonical_talk(document)
    return document


def serialize_canonical_talk(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_canonical_talk(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_canonical_talk(document), encoding="utf-8", newline="\n")


def canonical_content_checksum(document: Mapping[str, Any]) -> str:
    """Hash canonical content after removing the intentionally volatile assembly time."""
    stable = copy.deepcopy(document)
    provenance = stable.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("processed_at", None)
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_canonical_talk(document: Mapping[str, Any]) -> None:
    required = {"schema_version", "talk", "source", "target_reference", "alignments", "stream", "provenance"}
    if set(document) != required:
        raise ValueError("Canonical talk must contain exactly the canonical top-level fields")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported canonical schema version")
    talk = _object(document["talk"], "talk")
    source = _object(document["source"], "source")
    target = _object(document["target_reference"], "target_reference")
    stream = _object(document["stream"], "stream")
    if talk.get("talk_id") in (None, "") or talk.get("source_language") != "en" or talk.get("target_language") != "vi":
        raise ValueError("Canonical talk identity must be English-to-Vietnamese")
    if source.get("language") != "en" or target.get("language") != "vi":
        raise ValueError("Canonical source and target languages must be en and vi")
    source_segments = _segments(source.get("segments"), "source", timed=True)
    target_segments = _segments(target.get("segments"), "target", timed=False)
    source_by_id = {segment["segment_id"]: segment for segment in source_segments}
    target_by_id = {segment["segment_id"]: segment for segment in target_segments}
    _validate_alignments(document["alignments"], source_by_id, target_by_id)
    _validate_stream(stream, source_segments, source_by_id)
    provenance = _object(document["provenance"], "provenance")
    if not isinstance(provenance.get("processing_version"), str) or not provenance["processing_version"]:
        raise ValueError("Canonical provenance requires processing_version")


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Canonical {name} must be an object")
    return value


def _segments(value: Any, side: str, *, timed: bool) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Canonical {side} segments must be a non-empty array")
    identifiers: set[str] = set()
    segments: list[Mapping[str, Any]] = []
    previous_end = -1
    for index, segment in enumerate(value):
        segment = _object(segment, f"{side} segment")
        allowed = {"segment_id", "index", "text", "metadata"}
        if timed:
            allowed |= {"start_ms", "end_ms"}
        else:
            allowed |= {"start_ms", "end_ms"}
        if set(segment) - allowed:
            raise ValueError(f"Canonical {side} segment contains unsupported fields")
        identifier, text = segment.get("segment_id"), segment.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"Canonical {side} segment IDs must be unique")
        if segment.get("index") != index or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Canonical {side} segments must have ordered indices and non-empty text")
        identifiers.add(identifier)
        if timed:
            start_ms, end_ms = segment.get("start_ms"), segment.get("end_ms")
            if not isinstance(start_ms, int) or not isinstance(end_ms, int) or start_ms < 0 or end_ms < start_ms:
                raise ValueError(f"Invalid source timing at {identifier}")
            if start_ms < previous_end:
                raise ValueError("Canonical source clock moves backward")
            previous_end = end_ms
        elif ("start_ms" in segment) != ("end_ms" in segment):
            raise ValueError("Target timestamps must be paired")
        segments.append(segment)
    return segments


def _validate_alignments(value: Any, source: Mapping[str, Mapping[str, Any]], target: Mapping[str, Mapping[str, Any]]) -> None:
    if not isinstance(value, list):
        raise ValueError("Canonical alignments must be an array")
    alignment_ids: set[str] = set()
    used_source: set[str] = set()
    used_target: set[str] = set()
    previous_source = previous_target = -1
    for alignment in value:
        alignment = _object(alignment, "alignment")
        if set(alignment) - {"alignment_id", "source_segment_ids", "target_segment_ids", "method", "confidence", "metadata"}:
            raise ValueError("Canonical alignment contains unsupported fields")
        identifier = alignment.get("alignment_id")
        source_ids, target_ids = alignment.get("source_segment_ids"), alignment.get("target_segment_ids")
        if not isinstance(identifier, str) or not identifier or identifier in alignment_ids:
            raise ValueError("Canonical alignment IDs must be unique")
        if not isinstance(source_ids, list) or not source_ids or not isinstance(target_ids, list) or not target_ids:
            raise ValueError(f"Canonical alignment {identifier} must be bilingual")
        if len(source_ids) != len(set(source_ids)) or len(target_ids) != len(set(target_ids)):
            raise ValueError(f"Duplicate reference within canonical alignment {identifier}")
        try:
            source_segments = [source[item] for item in source_ids]
            target_segments = [target[item] for item in target_ids]
        except KeyError as error:
            raise ValueError(f"Unresolved canonical alignment reference: {error.args[0]}") from error
        source_indices = [item["index"] for item in source_segments]
        target_indices = [item["index"] for item in target_segments]
        if source_indices != list(range(source_indices[0], source_indices[0] + len(source_indices))) or target_indices != list(range(target_indices[0], target_indices[0] + len(target_indices))):
            raise ValueError(f"Non-contiguous canonical alignment {identifier}")
        if source_indices[0] <= previous_source or target_indices[0] <= previous_target:
            raise ValueError(f"Non-monotonic canonical alignment {identifier}")
        if used_source.intersection(source_ids) or used_target.intersection(target_ids):
            raise ValueError(f"Reused canonical segment reference in {identifier}")
        alignment_ids.add(identifier)
        used_source.update(source_ids)
        used_target.update(target_ids)
        previous_source, previous_target = source_indices[-1], target_indices[-1]


def _validate_stream(stream: Mapping[str, Any], source_segments: list[Mapping[str, Any]], source: Mapping[str, Mapping[str, Any]]) -> None:
    if stream.get("timing_mode") not in {"simulated", "recovered_from_caption_starts"}:
        raise ValueError("Canonical stream timing mode must preserve the timed-source mode")
    tokens = stream.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("Canonical stream tokens must be an array")
    token_ids: set[str] = set()
    previous_emit = -1
    previous_source_index = -1
    per_segment: dict[str, list[Mapping[str, Any]]] = {item["segment_id"]: [] for item in source_segments}
    for index, token in enumerate(tokens):
        token = _object(token, "stream token")
        if set(token) != {"token_id", "index", "text", "source_segment_id", "segment_index", "emit_ms"}:
            raise ValueError("Canonical stream token contains unsupported fields")
        identifier, segment_id, text = token.get("token_id"), token.get("source_segment_id"), token.get("text")
        if not isinstance(identifier, str) or not identifier or identifier in token_ids or token.get("index") != index:
            raise ValueError("Canonical stream token IDs and indices must be unique and contiguous")
        if not isinstance(segment_id, str) or segment_id not in source or not isinstance(text, str) or not text.strip():
            raise ValueError("Canonical stream token has invalid source-only content or reference")
        emit_ms = token.get("emit_ms")
        segment = source[segment_id]
        if segment["index"] < previous_source_index:
            raise ValueError("Canonical stream source segments are out of order")
        if not isinstance(emit_ms, int) or not segment["start_ms"] <= emit_ms <= segment["end_ms"] or emit_ms < previous_emit:
            raise ValueError("Canonical stream emit_ms is outside its segment or non-monotonic")
        token_ids.add(identifier)
        previous_emit = emit_ms
        previous_source_index = segment["index"]
        per_segment[segment_id].append(token)
    for segment in source_segments:
        tokens_for_segment = per_segment[segment["segment_id"]]
        if [token.get("segment_index") for token in tokens_for_segment] != list(range(len(tokens_for_segment))):
            raise ValueError(f"Invalid segment-local token indices for {segment['segment_id']}")
        if [token["text"] for token in tokens_for_segment] != lexical_tokens(segment["text"]):
            raise ValueError(f"Stream tokens do not match timed-source tokenization for {segment['segment_id']}")


__all__ = ["SCHEMA_VERSION", "canonical_content_checksum", "load_canonical_talk", "load_json", "serialize_canonical_talk", "validate_canonical_talk", "write_canonical_talk"]
