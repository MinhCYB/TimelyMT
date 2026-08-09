"""Leakage-safe causal translation requests and derived hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from timelymt.data.canonical.core import canonical_content_checksum, load_canonical_talk, validate_canonical_talk
from timelymt.data.manifest.core import lookup_split_for_talk
from timelymt.translator.core import Translator
from timelymt.translator.envit5 import EnViT5Config


ARTIFACT_SCHEMA_VERSION = "1.0.0"
REQUEST_ID_VERSION = "1.0.0"
DATASET_NAME = "TimelyMT Streaming Dataset v1"
FORBIDDEN_FIELD_FRAGMENTS = (
    "alignment",
    "bleu",
    "chrf",
    "commit",
    "future",
    "policy",
    "pseudo",
    "reference",
    "target_segment",
)


@dataclass(frozen=True)
class RuntimeSourceToken:
    """One arrived lexical token, stripped of segment-boundary provenance."""

    talk_id: str
    token_id: str
    token_index: int
    text: str
    emit_ms: int


@dataclass(frozen=True)
class RuntimeTalk:
    """A source-only observed prefix of one canonical talk."""

    talk_id: str
    split: str
    tokens: tuple[RuntimeSourceToken, ...]

    @property
    def latest_observed_token_index(self) -> int:
        if not self.tokens:
            raise ValueError(f"Runtime talk has no observed tokens: {self.talk_id}")
        return self.tokens[-1].token_index


@dataclass(frozen=True)
class TranslatorIdentity:
    model_id: str
    model_revision: str
    config_version: str
    config_fingerprint: str
    generation_config_fingerprint: str


@dataclass(frozen=True)
class TranslationRequest:
    artifact_schema_version: str
    request_id: str
    talk_id: str
    split: str
    start_token_index: int
    end_token_index: int
    observation_emit_ms: int
    source_text: str
    translator_model_id: str
    translator_revision: str
    translator_config_version: str
    translator_config_fingerprint: str
    generation_config_fingerprint: str


@dataclass(frozen=True)
class TranslationHypothesis:
    artifact_schema_version: str
    request_id: str
    talk_id: str
    split: str
    start_token_index: int
    end_token_index: int
    observation_emit_ms: int
    source_text: str
    translated_text: str
    source_token_count: int
    target_token_count: int | None
    translator_model_id: str
    translator_revision: str
    translator_config_version: str
    translator_config_fingerprint: str
    generation_config_fingerprint: str
    device: str | None
    dtype: str | None
    cache_hit: bool | None


@dataclass(frozen=True)
class ArtifactProvenance:
    dataset_name: str
    dataset_snapshot_version: str
    dataset_snapshot_checksum: str
    dataset_manifest_path: str
    split_manifest_path: str
    split_checksum: str
    translator_model_id: str
    translator_revision: str
    translator_config_version: str
    translator_config_fingerprint: str
    generation_config_fingerprint: str


@dataclass(frozen=True)
class DerivedArtifactManifest:
    artifact_schema_version: str
    artifact_type: str
    artifact_path: str
    artifact_checksum: str
    record_count: int
    source_talk_ids: tuple[str, ...]
    provenance: ArtifactProvenance
    created_at: str


def stable_fingerprint(value: Any) -> str:
    """Hash JSON-compatible semantic content with canonical JSON encoding."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def translator_identity(config: EnViT5Config) -> TranslatorIdentity:
    """Derive identity from the complete frozen config and generation subset."""

    if not config.frozen or config.model_revision is None:
        raise ValueError("translator identity requires a frozen, pinned configuration")
    config_document = asdict(config)
    return TranslatorIdentity(
        model_id=config.model_id,
        model_revision=config.model_revision,
        config_version=config.config_version,
        config_fingerprint=stable_fingerprint(config_document),
        generation_config_fingerprint=stable_fingerprint(config.generation_parameters),
    )


def load_runtime_talk(
    canonical_path: Path | str,
    *,
    dataset_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    observed_through_token_index: int,
) -> RuntimeTalk:
    """Load only the arrived source-token prefix from a Dataset v1 talk."""

    document = load_canonical_talk(Path(canonical_path))
    talk_id = document["talk"]["talk_id"]
    entries = {entry["talk_id"]: entry for entry in dataset_manifest.get("talks", [])}
    if talk_id not in entries:
        raise ValueError(f"Talk does not exist in Dataset v1 manifest: {talk_id}")
    if canonical_content_checksum(document) != entries[talk_id].get("content_checksum"):
        raise ValueError(f"Canonical content checksum mismatch for Dataset v1 talk: {talk_id}")
    return runtime_talk_from_canonical(
        document,
        split_manifest=split_manifest,
        observed_through_token_index=observed_through_token_index,
    )


def runtime_talk_from_canonical(
    document: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any],
    observed_through_token_index: int,
) -> RuntimeTalk:
    """Sanitize a validated canonical mapping into a narrow runtime view."""

    validate_canonical_talk(document)
    if isinstance(observed_through_token_index, bool) or not isinstance(observed_through_token_index, int):
        raise TypeError("observed_through_token_index must be an integer")
    talk_id = document.get("talk", {}).get("talk_id")
    if not isinstance(talk_id, str) or not talk_id:
        raise ValueError("Canonical talk has no valid talk_id")
    split = lookup_split_for_talk(split_manifest, talk_id)
    canonical_tokens = document.get("stream", {}).get("tokens")
    if not isinstance(canonical_tokens, list) or not canonical_tokens:
        raise ValueError(f"Canonical talk has no stream tokens: {talk_id}")
    if observed_through_token_index < 0 or observed_through_token_index >= len(canonical_tokens):
        raise ValueError(
            f"observed_through_token_index {observed_through_token_index} is outside talk {talk_id}"
        )
    tokens = tuple(
        RuntimeSourceToken(
            talk_id=talk_id,
            token_id=token["token_id"],
            token_index=token["index"],
            text=token["text"],
            emit_ms=token["emit_ms"],
        )
        for token in canonical_tokens[: observed_through_token_index + 1]
    )
    _validate_runtime_tokens(talk_id, tokens)
    return RuntimeTalk(talk_id=talk_id, split=split, tokens=tokens)


def reconstruct_source_text(tokens: Sequence[RuntimeSourceToken]) -> str:
    """Join consecutive runtime lexical token texts with one ASCII space."""

    if not tokens:
        raise ValueError("source span must contain at least one runtime token")
    first = tokens[0]
    for offset, token in enumerate(tokens):
        if token.talk_id != first.talk_id or token.token_index != first.token_index + offset:
            raise ValueError("source span tokens must be consecutive and from one talk")
        if not isinstance(token.text, str) or not token.text.strip():
            raise ValueError("runtime lexical token text must be non-empty")
    return " ".join(token.text for token in tokens)


def make_translation_request(
    runtime_talk: RuntimeTalk,
    start_token_index: int,
    end_token_index: int,
    *,
    translator: TranslatorIdentity,
) -> TranslationRequest:
    """Create one exact caller-selected inclusive causal source-span request."""

    _validate_span_indices(start_token_index, end_token_index)
    if end_token_index > runtime_talk.latest_observed_token_index:
        raise ValueError(
            f"end_token_index {end_token_index} is a future token; latest observed is "
            f"{runtime_talk.latest_observed_token_index}"
        )
    if start_token_index < runtime_talk.tokens[0].token_index:
        raise ValueError("start_token_index is not available in this runtime view")
    by_index = {token.token_index: token for token in runtime_talk.tokens}
    try:
        span = tuple(by_index[index] for index in range(start_token_index, end_token_index + 1))
    except KeyError as error:
        raise ValueError(f"source token index is unavailable: {error.args[0]}") from error
    source_text = reconstruct_source_text(span)
    request_id = deterministic_request_id(
        talk_id=runtime_talk.talk_id,
        start_token_index=start_token_index,
        end_token_index=end_token_index,
        translator=translator,
    )
    return TranslationRequest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        request_id=request_id,
        talk_id=runtime_talk.talk_id,
        split=runtime_talk.split,
        start_token_index=start_token_index,
        end_token_index=end_token_index,
        observation_emit_ms=span[-1].emit_ms,
        source_text=source_text,
        translator_model_id=translator.model_id,
        translator_revision=translator.model_revision,
        translator_config_version=translator.config_version,
        translator_config_fingerprint=translator.config_fingerprint,
        generation_config_fingerprint=translator.generation_config_fingerprint,
    )


def deterministic_request_id(
    *,
    talk_id: str,
    start_token_index: int,
    end_token_index: int,
    translator: TranslatorIdentity,
) -> str:
    identity = {
        "request_id_version": REQUEST_ID_VERSION,
        "talk_id": talk_id,
        "start_token_index": start_token_index,
        "end_token_index": end_token_index,
        "translator": asdict(translator),
    }
    return f"trq-{stable_fingerprint(identity)}"


def validate_translation_request(
    request: TranslationRequest,
    *,
    runtime_talk: RuntimeTalk,
    translator: TranslatorIdentity,
) -> None:
    _reject_forbidden_fields(asdict(request))
    if request.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported translation request schema version")
    if request.talk_id != runtime_talk.talk_id or request.split != runtime_talk.split:
        raise ValueError("request talk or split does not match the runtime talk")
    expected = make_translation_request(
        runtime_talk,
        request.start_token_index,
        request.end_token_index,
        translator=translator,
    )
    if request != expected:
        raise ValueError("translation request does not match its causal source span and translator identity")


def translate_requests(
    translator: Translator,
    requests: Sequence[TranslationRequest],
    *,
    translator_identity: TranslatorIdentity,
    batch_size: int = 8,
) -> list[TranslationHypothesis]:
    """Translate validated request rows in order without deriving new spans."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    rows = list(requests)
    _check_duplicate_request_ids(rows)
    for request in rows:
        _validate_request_identity(request, translator_identity)
        _reject_forbidden_fields(asdict(request))
    hypotheses: list[TranslationHypothesis] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        results = translator.translate_batch([request.source_text for request in batch])
        if len(results) != len(batch):
            raise RuntimeError("translator returned a different number of request translations")
        for request, result in zip(batch, results, strict=True):
            if result.source_text != request.source_text:
                raise ValueError("translator result changed the exact request source text")
            metadata = result.metadata
            hypothesis = TranslationHypothesis(
                artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
                request_id=request.request_id,
                talk_id=request.talk_id,
                split=request.split,
                start_token_index=request.start_token_index,
                end_token_index=request.end_token_index,
                observation_emit_ms=request.observation_emit_ms,
                source_text=request.source_text,
                translated_text=result.translated_text,
                source_token_count=request.end_token_index - request.start_token_index + 1,
                target_token_count=result.target_token_count,
                translator_model_id=translator_identity.model_id,
                translator_revision=translator_identity.model_revision,
                translator_config_version=translator_identity.config_version,
                translator_config_fingerprint=translator_identity.config_fingerprint,
                generation_config_fingerprint=translator_identity.generation_config_fingerprint,
                device=_optional_string(metadata.get("device")),
                dtype=_optional_string(metadata.get("dtype")),
                cache_hit=_optional_bool(metadata.get("cache_hit")),
            )
            validate_translation_hypothesis(
                hypothesis,
                request=request,
                translator=translator_identity,
            )
            hypotheses.append(hypothesis)
    return hypotheses


def validate_translation_hypothesis(
    hypothesis: TranslationHypothesis,
    *,
    request: TranslationRequest,
    translator: TranslatorIdentity,
) -> None:
    _reject_forbidden_fields(asdict(hypothesis))
    if hypothesis.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported translation hypothesis schema version")
    shared = (
        "request_id",
        "talk_id",
        "split",
        "start_token_index",
        "end_token_index",
        "observation_emit_ms",
        "source_text",
    )
    if any(getattr(hypothesis, field) != getattr(request, field) for field in shared):
        raise ValueError("hypothesis does not resolve to the exact translation request")
    if hypothesis.source_token_count != request.end_token_index - request.start_token_index + 1:
        raise ValueError("hypothesis source_token_count does not match the inclusive request span")
    if not isinstance(hypothesis.translated_text, str):
        raise ValueError("hypothesis translated_text must be a string")
    _validate_request_identity(hypothesis, translator)


def build_artifact_provenance(
    *,
    dataset_snapshot_manifest: Mapping[str, Any],
    dataset_manifest_path: Path | str,
    split_manifest: Mapping[str, Any],
    split_manifest_path: Path | str,
    translator: TranslatorIdentity,
) -> ArtifactProvenance:
    if dataset_snapshot_manifest.get("dataset_name") != DATASET_NAME:
        raise ValueError(f"expected frozen dataset identity {DATASET_NAME!r}")
    snapshot_checksum = dataset_snapshot_manifest.get("manifest_checksum")
    if not isinstance(snapshot_checksum, str):
        raise ValueError("frozen dataset snapshot has no valid manifest checksum")
    split_checksum = stable_fingerprint(split_manifest)
    if snapshot_checksum != split_manifest.get("dataset_manifest_checksum"):
        raise ValueError("dataset snapshot and split manifest checksums do not match")
    if split_checksum != dataset_snapshot_manifest.get("split_manifest_checksum"):
        raise ValueError("frozen dataset split checksum does not match split manifest")
    return ArtifactProvenance(
        dataset_name=DATASET_NAME,
        dataset_snapshot_version=dataset_snapshot_manifest["snapshot_version"],
        dataset_snapshot_checksum=snapshot_checksum,
        dataset_manifest_path=Path(dataset_manifest_path).as_posix(),
        split_manifest_path=Path(split_manifest_path).as_posix(),
        split_checksum=split_checksum,
        translator_model_id=translator.model_id,
        translator_revision=translator.model_revision,
        translator_config_version=translator.config_version,
        translator_config_fingerprint=translator.config_fingerprint,
        generation_config_fingerprint=translator.generation_config_fingerprint,
    )


Record = TypeVar("Record", TranslationRequest, TranslationHypothesis)


def write_artifact_jsonl(
    path: Path | str,
    records: Sequence[TranslationRequest] | Sequence[TranslationHypothesis],
    *,
    provenance: ArtifactProvenance,
    artifact_type: str,
    manifest_path: Path | str,
    created_at: str | None = None,
) -> DerivedArtifactManifest:
    """Write deterministic ordered JSONL plus a small provenance manifest."""

    if artifact_type not in {"translation_requests", "translation_hypotheses"}:
        raise ValueError("unsupported derived artifact type")
    rows = list(records)
    _check_duplicate_request_ids(rows)
    expected_class = TranslationRequest if artifact_type == "translation_requests" else TranslationHypothesis
    if any(not isinstance(row, expected_class) for row in rows):
        raise TypeError(f"{artifact_type} contains an incompatible record")
    for row in rows:
        _reject_forbidden_fields(asdict(row))
        if _identity_tuple(row) != _identity_tuple(provenance):
            raise ValueError("artifact row translator identity does not match manifest provenance")
    serialized = "".join(
        json.dumps(asdict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(serialized, encoding="utf-8", newline="\n")
    manifest = DerivedArtifactManifest(
        artifact_schema_version=ARTIFACT_SCHEMA_VERSION,
        artifact_type=artifact_type,
        artifact_path=artifact_path.as_posix(),
        artifact_checksum=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        record_count=len(rows),
        source_talk_ids=tuple(sorted({row.talk_id for row in rows})),
        provenance=provenance,
        created_at=created_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    manifest_document = asdict(manifest)
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def read_artifact_jsonl(path: Path | str, record_type: type[Record]) -> list[Record]:
    """Strictly reconstruct request or hypothesis dataclasses from JSONL."""

    if record_type not in {TranslationRequest, TranslationHypothesis}:
        raise TypeError("record_type must be TranslationRequest or TranslationHypothesis")
    rows: list[Record] = []
    expected_fields = set(record_type.__dataclass_fields__)
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            document = json.loads(line)
            if not isinstance(document, dict) or set(document) != expected_fields:
                raise ValueError(f"invalid fields on JSONL line {line_number}")
            _reject_forbidden_fields(document)
            rows.append(record_type(**document))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read derived artifact {path}: {error}") from error
    _check_duplicate_request_ids(rows)
    return rows


def validate_artifact_manifest(
    manifest: DerivedArtifactManifest,
    *,
    records: Sequence[TranslationRequest] | Sequence[TranslationHypothesis],
    artifact_bytes: bytes,
) -> None:
    if manifest.artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported derived artifact manifest schema version")
    if manifest.artifact_checksum != hashlib.sha256(artifact_bytes).hexdigest():
        raise ValueError("derived artifact checksum mismatch")
    if manifest.record_count != len(records):
        raise ValueError("derived artifact record count mismatch")
    if manifest.source_talk_ids != tuple(sorted({record.talk_id for record in records})):
        raise ValueError("derived artifact source talk IDs mismatch")
    _check_duplicate_request_ids(records)


def _validate_runtime_tokens(talk_id: str, tokens: Sequence[RuntimeSourceToken]) -> None:
    previous_emit = -1
    for expected_index, token in enumerate(tokens):
        if token.talk_id != talk_id or token.token_index != expected_index:
            raise ValueError("runtime tokens must preserve contiguous canonical talk indices")
        if token.emit_ms < previous_emit:
            raise ValueError("runtime token timestamps must be nondecreasing")
        previous_emit = token.emit_ms


def _validate_span_indices(start: int, end: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
        raise TypeError("source span indices must be integers")
    if start < 0 or end < 0:
        raise ValueError("source span indices must be nonnegative")
    if start > end:
        raise ValueError("start_token_index must be <= end_token_index")


def _validate_request_identity(record: Any, translator: TranslatorIdentity) -> None:
    if _identity_tuple(record) != _identity_tuple(translator):
        raise ValueError("translator identity does not match the frozen request identity")
    expected_id = deterministic_request_id(
        talk_id=record.talk_id,
        start_token_index=record.start_token_index,
        end_token_index=record.end_token_index,
        translator=translator,
    )
    if record.request_id != expected_id:
        raise ValueError("request_id is not deterministic for this span and translator")


def _identity_tuple(value: Any) -> tuple[Any, ...]:
    def field(name: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(name)
        return getattr(value, name)

    return (
        field("translator_model_id") if hasattr(value, "translator_model_id") or isinstance(value, Mapping) else field("model_id"),
        field("translator_revision") if hasattr(value, "translator_revision") or isinstance(value, Mapping) else field("model_revision"),
        field("translator_config_version") if hasattr(value, "translator_config_version") or isinstance(value, Mapping) else field("config_version"),
        field("translator_config_fingerprint") if hasattr(value, "translator_config_fingerprint") or isinstance(value, Mapping) else field("config_fingerprint"),
        field("generation_config_fingerprint"),
    )


def _check_duplicate_request_ids(records: Iterable[Any]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.request_id in seen:
            raise ValueError(f"duplicate request_id: {record.request_id}")
        seen.add(record.request_id)


def _reject_forbidden_fields(document: Mapping[str, Any]) -> None:
    for name in document:
        lowered = name.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_FIELD_FRAGMENTS):
            raise ValueError(f"unsafe field is forbidden in translation artifacts: {name}")


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactProvenance",
    "DerivedArtifactManifest",
    "RuntimeSourceToken",
    "RuntimeTalk",
    "TranslationHypothesis",
    "TranslationRequest",
    "TranslatorIdentity",
    "build_artifact_provenance",
    "deterministic_request_id",
    "load_runtime_talk",
    "make_translation_request",
    "read_artifact_jsonl",
    "reconstruct_source_text",
    "runtime_talk_from_canonical",
    "stable_fingerprint",
    "translate_requests",
    "translator_identity",
    "validate_artifact_manifest",
    "validate_translation_hypothesis",
    "validate_translation_request",
    "write_artifact_jsonl",
]
