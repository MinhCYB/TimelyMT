"""Standalone prepared-context contract, provenance checks, and serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlparse


SCHEMA_VERSION = "prepared-context-v0"
SPLITS = frozenset({"train", "dev"})
SOURCE_TYPES = frozenset({
    "paper", "official_article", "project_documentation", "event_abstract",
    "slides", "speaker_notes", "glossary", "other",
})
CLASSIFICATIONS = frozenset({
    "SAFE_PRETALK_CONFIRMED", "SAFE_PRETALK_PLAUSIBLE", "PUBLIC_POST_TALK",
    "TRANSCRIPT_DERIVED", "REFERENCE_DERIVED", "QUESTIONABLE", "UNAVAILABLE",
})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_CHECKSUM = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class PreparedContextMetadata:
    """Informational talk metadata; never a prepared source."""

    title: str | None = None
    speaker: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class PreparedContextSource:
    source_id: str
    source_type: str
    text: str
    source_uri: str
    language: str
    published_at: str | None
    acquired_at: str | None
    available_before_talk: bool
    classification: str
    relationship: str
    transcript_used: bool
    reference_used: bool
    checksum: str

    @property
    def model_eligible(self) -> bool:
        return (
            self.classification == "SAFE_PRETALK_CONFIRMED"
            and self.available_before_talk
            and not self.transcript_used
            and not self.reference_used
        )


@dataclass(frozen=True)
class PreparedContextPool:
    schema_version: str
    talk_id: str
    split: str
    metadata: PreparedContextMetadata
    sources: tuple[PreparedContextSource, ...]

    def eligible_sources(self) -> tuple[PreparedContextSource, ...]:
        """Return only strictly confirmed, non-leaking pre-talk sources."""

        return tuple(source for source in self.sources if source.model_eligible)


def source_text_checksum(text: str) -> str:
    """Hash the exact UTF-8 source text without normalization."""

    if not isinstance(text, str):
        raise TypeError("prepared source text must be a string")
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def validate_prepared_context(document: Mapping[str, Any]) -> None:
    """Validate the closed v0 structure and its provenance/leakage invariants."""

    if not isinstance(document, Mapping):
        raise ValueError("Prepared context pool must be an object")
    if set(document) != {"schema_version", "talk_id", "split", "metadata", "sources"}:
        raise ValueError("Prepared context pool has unsupported or missing fields")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported prepared context schema version")
    _validate_identifier(document.get("talk_id"), "talk_id")
    if document.get("split") not in SPLITS:
        raise ValueError("Prepared context split must be train or dev")

    metadata = _mapping(document.get("metadata"), "metadata")
    if set(metadata) - {"title", "speaker", "domain"}:
        raise ValueError("Prepared context metadata contains unsupported fields")
    if any(not isinstance(value, str) for value in metadata.values()):
        raise ValueError("Prepared context metadata values must be strings")

    sources = document.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Prepared context sources must be an array")
    source_ids: set[str] = set()
    for source in sources:
        source = _mapping(source, "source")
        _validate_source(source)
        source_id = source["source_id"]
        if source_id in source_ids:
            raise ValueError(f"Duplicate prepared context source_id: {source_id}")
        source_ids.add(source_id)


def load_prepared_context(path: Path | str) -> PreparedContextPool:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load prepared context pool {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"Prepared context pool must be a JSON object: {path}")
    validate_prepared_context(document)
    return _pool_from_document(document)


def serialize_prepared_context(pool: PreparedContextPool) -> str:
    document = _pool_to_document(pool)
    validate_prepared_context(document)
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_prepared_context(path: Path | str, pool: PreparedContextPool) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_prepared_context(pool), encoding="utf-8", newline="\n")


def _validate_source(source: Mapping[str, Any]) -> None:
    required = {
        "source_id", "source_type", "text", "source_uri", "language",
        "published_at", "acquired_at", "available_before_talk", "classification",
        "relationship", "transcript_used", "reference_used", "checksum",
    }
    if set(source) != required:
        raise ValueError("Prepared context source has unsupported or missing fields")
    source_id = _validate_identifier(source.get("source_id"), "source_id")
    if source.get("source_type") not in SOURCE_TYPES:
        raise ValueError(f"Prepared context source_type is invalid for {source_id}")
    text = source.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Prepared context text must be a string for {source_id}")
    source_uri = source.get("source_uri")
    if not isinstance(source_uri, str) or not source_uri or not urlparse(source_uri).scheme:
        raise ValueError(f"Prepared context source_uri is invalid for {source_id}")
    if source.get("language") != "en":
        raise ValueError(f"Prepared context language must be en for {source_id}")
    _validate_timestamp(source.get("published_at"), "published_at", source_id, nullable=True)
    classification = source.get("classification")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"Prepared context classification is invalid for {source_id}")
    if classification != "UNAVAILABLE" and not text:
        raise ValueError(f"Prepared context text is required for {source_id}")
    acquired_at = source.get("acquired_at")
    _validate_timestamp(acquired_at, "acquired_at", source_id, nullable=classification == "UNAVAILABLE")
    for field in ("available_before_talk", "transcript_used", "reference_used"):
        if not isinstance(source.get(field), bool):
            raise ValueError(f"Prepared context {field} must be boolean for {source_id}")
    if not isinstance(source.get("relationship"), str) or not source["relationship"]:
        raise ValueError(f"Prepared context relationship is required for {source_id}")
    checksum = source.get("checksum")
    if not isinstance(checksum, str) or not _CHECKSUM.fullmatch(checksum):
        raise ValueError(f"Prepared context checksum syntax is invalid for {source_id}")
    if checksum != source_text_checksum(text):
        raise ValueError(f"Prepared context checksum mismatch for {source_id}")
    _validate_provenance_consistency(source, source_id)


def _validate_provenance_consistency(source: Mapping[str, Any], source_id: str) -> None:
    classification = source["classification"]
    if classification == "SAFE_PRETALK_CONFIRMED" and (
        not source["available_before_talk"] or source["transcript_used"] or source["reference_used"]
    ):
        raise ValueError(f"Confirmed prepared source has contradictory provenance: {source_id}")
    if classification == "PUBLIC_POST_TALK" and source["available_before_talk"]:
        raise ValueError(f"Post-talk source cannot be available before talk: {source_id}")
    if classification == "TRANSCRIPT_DERIVED" and not source["transcript_used"]:
        raise ValueError(f"Transcript-derived source must declare transcript use: {source_id}")
    if classification == "REFERENCE_DERIVED" and not source["reference_used"]:
        raise ValueError(f"Reference-derived source must declare reference use: {source_id}")
    if classification == "UNAVAILABLE" and (source["acquired_at"] is not None or source["available_before_talk"]):
        raise ValueError(f"Unavailable source has contradictory acquisition provenance: {source_id}")


def _validate_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Prepared context {field} is invalid")
    return value


def _validate_timestamp(value: Any, field: str, source_id: str, *, nullable: bool) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ValueError(f"Prepared context {field} is required for {source_id}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Prepared context {field} is invalid for {source_id}") from error
    if "T" not in value or parsed.tzinfo is None:
        raise ValueError(f"Prepared context {field} must be an ISO-8601 date-time with timezone for {source_id}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Prepared context {label} must be an object")
    return value


def _pool_from_document(document: Mapping[str, Any]) -> PreparedContextPool:
    metadata = document["metadata"]
    sources = document["sources"]
    assert isinstance(metadata, Mapping) and isinstance(sources, list)
    return PreparedContextPool(
        schema_version=document["schema_version"],
        talk_id=document["talk_id"],
        split=document["split"],
        metadata=PreparedContextMetadata(
            title=metadata.get("title"), speaker=metadata.get("speaker"), domain=metadata.get("domain"),
        ),
        sources=tuple(PreparedContextSource(**source) for source in sources),
    )


def _pool_to_document(pool: PreparedContextPool) -> dict[str, Any]:
    if not isinstance(pool, PreparedContextPool):
        raise TypeError("pool must be a PreparedContextPool")
    metadata = {key: value for key, value in asdict(pool.metadata).items() if value is not None}
    return {
        "schema_version": pool.schema_version,
        "talk_id": pool.talk_id,
        "split": pool.split,
        "metadata": metadata,
        "sources": [asdict(source) for source in pool.sources],
    }


__all__ = [
    "CLASSIFICATIONS", "SCHEMA_VERSION", "SOURCE_TYPES", "SPLITS",
    "PreparedContextMetadata", "PreparedContextPool", "PreparedContextSource",
    "load_prepared_context", "serialize_prepared_context", "source_text_checksum",
    "validate_prepared_context", "write_prepared_context",
]
