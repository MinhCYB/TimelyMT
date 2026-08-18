"""Safe, persistent checkpoint packages for researcher-operated P3-GLOBAL runs.

This module deliberately manages only experiment artifacts.  It never trains,
rolls out, evaluates, or reads TEST data.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, cast

import torch

from timelymt.data.translation_artifacts import stable_fingerprint
from .policy_p3_global import (
    P3_VARIANT,
    prepared_manifest_fingerprint,
    validate_p3_checkpoint_metadata,
)
from .policy_v2 import sha256_file, validate_v1_supervision


P3_PACKAGE_SCHEMA_VERSION = "1.0.0"
P3_CHECKPOINT_ROOT = PurePosixPath("checkpoints/policy_p3_global")
UPSTREAM_SUPERVISION_ROOT = PurePosixPath("data/policy/pseudo_labels/train")
P3_STAGE_ORDER = {"NONE": 0, "TRAINED": 1, "DEV_SINGLE_ROLLOUT": 2, "DEV_GRID_ROLLOUT": 3, "DEV_EVALUATED": 4}
_PACKAGE_METADATA = PurePosixPath("checkpoint-metadata.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def config_fingerprint(path: Path) -> str:
    return stable_fingerprint(json.loads(path.read_text(encoding="utf-8")))


def repository_identity(root: Path) -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    try:
        return {
            "repo_commit": git("rev-parse", "HEAD"),
            "repo_remote": git("remote", "get-url", "origin"),
            "working_tree_dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("P3 checkpoint persistence requires a git checkout with origin") from error


def _safe_relative(name: str, roots: Iterable[PurePosixPath]) -> PurePosixPath:
    if not name or "\\" in name or name.startswith("/"):
        raise RuntimeError(f"unsafe checkpoint path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe checkpoint path: {name!r}")
    if path == _PACKAGE_METADATA:
        return path
    if not any(path == root or root in path.parents for root in roots):
        raise RuntimeError(f"unexpected checkpoint path: {name!r}")
    if any(part.lower() == "test" for part in path.parts):
        raise RuntimeError(f"TEST path is forbidden in checkpoint package: {name!r}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _validate_internal_checkpoint(root: Path, manifest_path: Path) -> dict[str, Any]:
    checkpoint = root / Path(*P3_CHECKPOINT_ROOT.parts) / "P3_GLOBAL.pt"
    metadata_path = root / Path(*P3_CHECKPOINT_ROOT.parts) / "P3_GLOBAL.metadata.json"
    metadata = _read_json(metadata_path)
    if not checkpoint.is_file() or metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("P3 checkpoint file is missing or its SHA-256 does not match internal metadata")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError("P3 checkpoint payload is unreadable") from error
    # This is the repository's official P3 metadata validator.  It does not
    # instantiate MiniLM, so package inspection remains cache-independent.
    validate_p3_checkpoint_metadata(metadata, payload, manifest_path=manifest_path, cache=cast(Any, SimpleNamespace(dimension=384)))
    return metadata


def _validate_package_metadata(metadata: Mapping[str, Any], *, config_path: Path, manifest_path: Path) -> None:
    expected = {
        "schema_version": P3_PACKAGE_SCHEMA_VERSION,
        "experiment": P3_VARIANT,
        "config_fingerprint": config_fingerprint(config_path),
        "prepared_context_manifest_fingerprint": prepared_manifest_fingerprint(manifest_path),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P3 package metadata is incompatible with this experiment")
    if metadata.get("completed_stage") not in P3_STAGE_ORDER:
        raise RuntimeError("P3 package has an invalid persistence stage")
    if not isinstance(metadata.get("created_at"), str) or not metadata.get("repo_commit") or not metadata.get("repo_remote"):
        raise RuntimeError("P3 package metadata lacks repository identity")


def _validate_expanded_root(root: Path, *, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"unsafe expanded checkpoint root: {root}")
    files = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"symlink forbidden in expanded checkpoint: {path}")
        _safe_relative(path.relative_to(root).as_posix(), (P3_CHECKPOINT_ROOT,))
    expected_files = {
        _PACKAGE_METADATA,
        P3_CHECKPOINT_ROOT / "P3_GLOBAL.pt",
        P3_CHECKPOINT_ROOT / "P3_GLOBAL.metadata.json",
    }
    actual_files = {PurePosixPath(path.relative_to(root).as_posix()) for path in files}
    if actual_files != expected_files:
        raise RuntimeError("P3 checkpoint package must contain only persistence metadata and P3 weights/metadata")
    package = _read_json(root / _PACKAGE_METADATA)
    _validate_package_metadata(package, config_path=config_path, manifest_path=manifest_path)
    internal = _validate_internal_checkpoint(root, manifest_path)
    if package.get("checkpoint_sha256") != internal.get("checkpoint_sha256"):
        raise RuntimeError("P3 package and internal checkpoint hashes differ")
    return package


def validate_p3_candidate(candidate: Path, *, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate a raw archive or expanded package without touching repository artifacts."""
    candidate = Path(candidate)
    if candidate.is_dir():
        return _validate_expanded_root(candidate, config_path=config_path, manifest_path=manifest_path)
    if not candidate.is_file() or not candidate.name.endswith(".tar.gz"):
        raise RuntimeError(f"unsupported P3 checkpoint candidate: {candidate}")
    with tempfile.TemporaryDirectory(prefix="p3-checkpoint-validate-") as temporary:
        destination = Path(temporary)
        try:
            with tarfile.open(candidate, "r:gz") as archive:
                members = archive.getmembers()
                names = [_safe_relative(member.name, (P3_CHECKPOINT_ROOT,)) for member in members]
                if len(names) != len(set(names)):
                    raise RuntimeError("checkpoint archive has duplicate paths")
                if any(not member.isfile() for member in members):
                    raise RuntimeError("checkpoint archive may contain files only")
                for member, relative in zip(members, names):
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read checkpoint archive member: {member.name}")
                    target = destination / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
        except tarfile.TarError as error:
            raise RuntimeError(f"invalid P3 checkpoint archive: {candidate}") from error
        return _validate_expanded_root(destination, config_path=config_path, manifest_path=manifest_path)


def discover_p3_candidates(root: Path, *, config_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    """Return compatible candidates ordered by metadata timestamp then stable path."""
    root = Path(root)
    discovered = sorted(root.rglob("*.tar.gz"))
    discovered += sorted({path.parent for path in root.rglob("checkpoint-metadata.json")})
    compatible = []
    for path in discovered:
        try:
            metadata = validate_p3_candidate(path, config_path=config_path, manifest_path=manifest_path)
        except RuntimeError as error:
            continue
        compatible.append({"path": path, "metadata": metadata})
    return sorted(compatible, key=lambda item: (item["metadata"]["created_at"], item["metadata"].get("checkpoint_sha256", ""), str(item["path"])), reverse=True)


def _copy_selected(source_root: Path, destination_root: Path, roots: Iterable[PurePosixPath]) -> None:
    selected = [source_root / _PACKAGE_METADATA]
    for root in roots:
        selected.extend(sorted((source_root / Path(*root.parts)).rglob("*")))
    for source in selected:
        if not source.is_file() or source.is_symlink():
            continue
        relative = _safe_relative(source.relative_to(source_root).as_posix(), roots)
        target = destination_root / Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def restore_p3_candidate(candidate: Path, repo_root: Path, *, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Validate completely in isolation, then atomically replace only P3 files."""
    package = validate_p3_candidate(candidate, config_path=config_path, manifest_path=manifest_path)
    with tempfile.TemporaryDirectory(prefix="p3-checkpoint-restore-") as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        if Path(candidate).is_dir():
            _copy_selected(Path(candidate), staged, (P3_CHECKPOINT_ROOT,))
        else:
            with tarfile.open(candidate, "r:gz") as archive:
                for member in archive.getmembers():
                    relative = _safe_relative(member.name, (P3_CHECKPOINT_ROOT,))
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read checkpoint archive member: {member.name}")
                    target = staged / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
        _validate_expanded_root(staged, config_path=config_path, manifest_path=manifest_path)
        target = Path(repo_root) / Path(*P3_CHECKPOINT_ROOT.parts)
        replacement = Path(temporary) / "replacement"
        shutil.copytree(staged / Path(*P3_CHECKPOINT_ROOT.parts), replacement)
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(replacement), str(target))
    return package


def local_p3_checkpoint(repo_root: Path, *, config_path: Path, manifest_path: Path) -> dict[str, Any] | None:
    root = Path(repo_root)
    metadata_path = root / P3_CHECKPOINT_ROOT / "P3_GLOBAL.metadata.json"
    checkpoint = root / P3_CHECKPOINT_ROOT / "P3_GLOBAL.pt"
    if not metadata_path.is_file() and not checkpoint.is_file():
        return None
    # Local files have no outer package. Validate the official P3 contract.
    internal = _validate_internal_checkpoint(root, manifest_path)
    return {"created_at": metadata_path.stat().st_mtime_ns, "checkpoint_sha256": internal["checkpoint_sha256"], "internal": internal}


def resolve_local_conflict(local: Mapping[str, Any] | None, persistent: Mapping[str, Any]) -> str:
    """Return keep-local, restore-persistent, or conflict; never overwrite ambiguously."""
    if local is None:
        return "restore-persistent"
    if local["checkpoint_sha256"] == persistent["checkpoint_sha256"]:
        return "keep-local"
    local_time, remote_time = str(local["created_at"]), str(persistent["created_at"])
    if local_time > remote_time:
        return "keep-local"
    if remote_time > local_time:
        return "restore-persistent"
    return "conflict"


def build_p3_package(repo_root: Path, destination: Path, *, config_path: Path, manifest_path: Path, stage: str) -> dict[str, Any]:
    if stage not in P3_STAGE_ORDER or stage == "NONE":
        raise ValueError(f"invalid P3 persistence stage: {stage}")
    repo_root, destination = Path(repo_root), Path(destination)
    internal = _validate_internal_checkpoint(repo_root, manifest_path)
    identity = repository_identity(repo_root)
    if identity["working_tree_dirty"]:
        print("WARNING: repository working tree is dirty; checkpoint package records the commit but not uncommitted scientific-code changes.")
    metadata = {
        "schema_version": P3_PACKAGE_SCHEMA_VERSION, "experiment": P3_VARIANT,
        "created_at": utc_now(), "completed_stage": stage,
        "repo_commit": identity["repo_commit"], "repo_remote": identity["repo_remote"],
        "working_tree_dirty": identity["working_tree_dirty"], "config_fingerprint": config_fingerprint(config_path),
        "prepared_context_manifest_fingerprint": prepared_manifest_fingerprint(manifest_path),
        "checkpoint_sha256": internal["checkpoint_sha256"],
    }
    with tempfile.TemporaryDirectory(prefix="p3-checkpoint-package-") as temporary:
        staged = Path(temporary) / "package"
        staged.mkdir()
        (staged / _PACKAGE_METADATA).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _copy_selected(repo_root, staged, (P3_CHECKPOINT_ROOT,))
        _validate_expanded_root(staged, config_path=config_path, manifest_path=manifest_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    info = archive.gettarinfo(str(path), arcname=path.relative_to(staged).as_posix())
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)
    return metadata


def restore_upstream_supervision(candidate: Path, repo_root: Path) -> None:
    """Restore only frozen V1 TRAIN pseudo-labels, never P3 or TEST artifacts."""
    candidate, repo_root = Path(candidate), Path(repo_root)
    with tempfile.TemporaryDirectory(prefix="p3-upstream-restore-") as temporary:
        staged = Path(temporary) / "root"
        staged.mkdir()
        if candidate.is_dir():
            sources = [(path, path.relative_to(candidate).as_posix()) for path in candidate.rglob("*") if path.is_file()]
            for source, name in sources:
                # Validate traversal/symlinks globally but retain only the narrow
                # upstream whitelist from a broader historical package.
                relative = _safe_relative(name, (UPSTREAM_SUPERVISION_ROOT,)) if name.startswith(str(UPSTREAM_SUPERVISION_ROOT)) else PurePosixPath(name)
                if relative == _PACKAGE_METADATA or not (relative == UPSTREAM_SUPERVISION_ROOT or UPSTREAM_SUPERVISION_ROOT in relative.parents):
                    continue
                target = staged / Path(*relative.parts); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        else:
            with tarfile.open(candidate, "r:gz") as archive:
                for member in archive.getmembers():
                    name = member.name
                    if not name or "\\" in name or name.startswith("/"):
                        raise RuntimeError(f"unsafe upstream archive path: {name!r}")
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                        raise RuntimeError(f"unsafe upstream archive path: {name!r}")
                    if relative == _PACKAGE_METADATA or not (relative == UPSTREAM_SUPERVISION_ROOT or UPSTREAM_SUPERVISION_ROOT in relative.parents):
                        continue
                    if not member.isfile():
                        raise RuntimeError("unsafe upstream archive member")
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read upstream archive member: {member.name}")
                    target = staged / Path(*relative.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as handle: shutil.copyfileobj(source, handle)
        validate_v1_supervision(staged / Path(*UPSTREAM_SUPERVISION_ROOT.parts), "train")
        target = repo_root / Path(*UPSTREAM_SUPERVISION_ROOT.parts)
        if target.exists(): shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged / Path(*UPSTREAM_SUPERVISION_ROOT.parts)), str(target))
