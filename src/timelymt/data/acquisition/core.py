"""Shared acquisition records, manifest loading, writing, and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse


STATUSES = {"pending", "available", "partial", "unavailable", "failed"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


class ManifestError(ValueError):
    """Raised when a candidate manifest violates the acquisition contract."""


@dataclass(frozen=True)
class Candidate:
    id: str
    slug: str
    title: str
    speaker: str
    domain: str
    priority: str
    provider: str
    source_url: str


@dataclass(frozen=True)
class Discovery:
    english_available: bool = False
    vietnamese_available: bool = False
    transcript_available: bool = False
    subtitle_timing_available: bool = False


@dataclass(frozen=True)
class AdapterArtifact:
    filename: str
    content: str


@dataclass(frozen=True)
class AdapterResponse:
    discovery: Discovery
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Sequence[AdapterArtifact] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class AcquisitionResult:
    candidate_id: str
    provider: str
    source_url: str
    acquired_at: str
    status: str
    discovered: Discovery
    artifact_paths: Sequence[str] = field(default_factory=tuple)
    discovered_metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        if record["failure_reason"] is None:
            record.pop("failure_reason")
        return record


class SourceAdapter(Protocol):
    provider: str

    def acquire(self, candidate: Candidate) -> AdapterResponse:
        """Discover and return public raw artifacts for one candidate."""

        ...


def load_manifest(path: Path) -> list[Candidate]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"Cannot parse manifest {path}: {error}") from error

    records = document.get("candidates") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise ManifestError("Manifest must contain a 'candidates' array")

    required = set(Candidate.__dataclass_fields__)
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ManifestError(f"Candidate {index} must be an object")
        missing = required - record.keys()
        if missing:
            raise ManifestError(f"Candidate {index} missing fields: {', '.join(sorted(missing))}")
        candidate = Candidate(**{key: record[key] for key in required})
        _validate_candidate(candidate, index)
        if candidate.id in seen:
            raise ManifestError(f"Duplicate candidate id: {candidate.id}")
        seen.add(candidate.id)
        candidates.append(candidate)
    return candidates


def _validate_candidate(candidate: Candidate, index: int) -> None:
    if not IDENTIFIER.fullmatch(candidate.id):
        raise ManifestError(f"Candidate {index} has invalid id: {candidate.id!r}")
    if not candidate.slug or not candidate.title or not candidate.provider:
        raise ManifestError(f"Candidate {index} has an empty required string")
    parsed = urlparse(candidate.source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ManifestError(f"Candidate {candidate.id} has invalid source_url")


def artifact_directory(raw_root: Path, candidate: Candidate) -> Path:
    return raw_root / candidate.provider / candidate.id


def acquire_candidates(
    candidates: Sequence[Candidate],
    adapters: Mapping[str, SourceAdapter],
    raw_root: Path,
    results_path: Path,
    *,
    skip_existing: bool = True,
) -> list[AcquisitionResult]:
    results: list[AcquisitionResult] = []
    for candidate in candidates:
        try:
            result = _acquire_one(candidate, adapters, raw_root, skip_existing)
        except Exception as error:  # A single provider/talk failure must not stop a batch.
            result = AcquisitionResult(
                candidate_id=candidate.id,
                provider=candidate.provider,
                source_url=candidate.source_url,
                acquired_at=_timestamp(),
                status="failed",
                discovered=Discovery(),
                failure_reason=f"{type(error).__name__}: {error}",
            )
            try:
                output_dir = artifact_directory(raw_root, candidate)
                output_dir.mkdir(parents=True, exist_ok=True)
                _write_json(output_dir / "acquisition.json", result.to_dict())
            except OSError:
                pass
        _append_result(results_path, result)
        results.append(result)
    return results


def _acquire_one(
    candidate: Candidate,
    adapters: Mapping[str, SourceAdapter],
    raw_root: Path,
    skip_existing: bool,
) -> AcquisitionResult:
    adapter = adapters.get(candidate.provider)
    if adapter is None:
        raise ValueError(f"No adapter registered for provider {candidate.provider!r}")

    output_dir = artifact_directory(raw_root, candidate)
    acquisition_path = output_dir / "acquisition.json"
    if skip_existing and acquisition_path.is_file():
        existing = json.loads(acquisition_path.read_text(encoding="utf-8"))
        if existing.get("status") in {"available", "partial", "unavailable"}:
            return _result_from_dict(existing)

    response = adapter.acquire(candidate)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for artifact in response.artifacts:
        if Path(artifact.filename).name != artifact.filename:
            raise ValueError(f"Unsafe artifact filename: {artifact.filename!r}")
        content = _normalize_text(artifact.content)
        if not content.strip():
            raise ValueError(f"Empty transcript artifact rejected: {artifact.filename}")
        path = output_dir / artifact.filename
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path.as_posix())

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "candidate": asdict(candidate),
        "discovered": asdict(response.discovery),
        "provider_metadata": dict(response.metadata),
    }
    _write_json(metadata_path, metadata)
    paths.insert(0, metadata_path.as_posix())

    status = _status_for(response.discovery)
    result = AcquisitionResult(
        candidate_id=candidate.id,
        provider=candidate.provider,
        source_url=candidate.source_url,
        acquired_at=_timestamp(),
        status=status,
        discovered=response.discovery,
        artifact_paths=tuple(paths),
        discovered_metadata=dict(response.metadata),
        warnings=tuple(response.warnings),
    )
    _write_json(acquisition_path, result.to_dict())
    return result


def _status_for(discovery: Discovery) -> str:
    if discovery.english_available and discovery.vietnamese_available:
        return "available"
    if discovery.english_available or discovery.vietnamese_available:
        return "partial"
    return "unavailable"


def _append_result(path: Path, result: AcquisitionResult) -> None:
    if result.status not in STATUSES:
        raise ValueError(f"Invalid acquisition status: {result.status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _result_from_dict(record: Mapping[str, Any]) -> AcquisitionResult:
    return AcquisitionResult(
        candidate_id=record["candidate_id"],
        provider=record["provider"],
        source_url=record["source_url"],
        acquired_at=record["acquired_at"],
        status=record["status"],
        discovered=Discovery(**record["discovered"]),
        artifact_paths=tuple(record.get("artifact_paths", ())),
        discovered_metadata=record.get("discovered_metadata", {}),
        warnings=tuple(record.get("warnings", ())),
        failure_reason=record.get("failure_reason"),
    )


def validate_artifacts(results: Sequence[AcquisitionResult], project_root: Path) -> list[str]:
    errors: list[str] = []
    for result in results:
        if result.status not in STATUSES:
            errors.append(f"{result.candidate_id}: missing or invalid status")
        if result.status not in {"available", "partial"}:
            continue
        for artifact_path in result.artifact_paths:
            path = Path(artifact_path)
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                errors.append(f"{result.candidate_id}: missing artifact {artifact_path}")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeError as error:
                errors.append(f"{result.candidate_id}: unreadable UTF-8 {artifact_path}: {error}")
                continue
            if path.name.startswith(("source.", "target.")) and not text.strip():
                errors.append(f"{result.candidate_id}: empty transcript {artifact_path}")
    return errors
