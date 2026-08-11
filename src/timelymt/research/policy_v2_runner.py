"""Artifact-oriented CLI stages for exploratory Policy V2."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from timelymt.data.canonical.core import load_canonical_talk
from .evaluation import latency_metrics, quality_metrics
from .policy import VARIANTS
from .policy_v2 import (
    DATASET_CHECKSUM, ENCODER_MODEL_ID, ENCODER_REVISION, EXPERIMENT_LABEL, EXPERIMENT_STATUS,
    POOLING_VERSION, SPLIT_CHECKSUM, THRESHOLDS, TRANSLATOR_FINGERPRINT, V1_SOURCE_COMMIT,
    EmbeddingCache, FrozenMiniLMEncoder, atomic_json, current_git_commit, load_v2_checkpoint,
    make_checkpoint_metadata, metrics_are_complete, restore_v1_artifacts, save_v2_checkpoint,
    select_v2_configuration, sha256_file, train_v2_policy, validate_prediction_record,
    validate_v1_supervision, v1_identity_document, reject_test_split,
)
from .streaming import learned_rollout, prediction_record


ROOT = Path(__file__).parents[3]
V2_EXPERIMENT = ROOT / "outputs/experiments/policy-v2"
V2_CHECKPOINTS = ROOT / "checkpoints/policy_v2"
V2_CACHE = ROOT / "outputs/experiments/policy-v2/embedding-cache"
PSEUDO = ROOT / "data/policy/pseudo_labels"
V1_SOURCE = V2_EXPERIMENT / "v1-source"
V1_METRICS = V1_SOURCE / "all.json"
V1_SELECTION = V1_SOURCE / "dev-selection.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy(variant: str, threshold: float) -> str:
    if variant not in VARIANTS or threshold not in THRESHOLDS:
        raise ValueError("V2 strategy variant/threshold is outside the frozen grid")
    return f"v2_{variant}_{threshold:.2f}"


def all_v2_strategies(variants: Sequence[str] = VARIANTS, thresholds: Sequence[float] = THRESHOLDS) -> list[str]:
    return [_strategy(variant, threshold) for variant in variants for threshold in thresholds]


def _encoder_cache(cache_dir: Path = V2_CACHE, *, batch_size: int = 256):
    encoder = FrozenMiniLMEncoder(batch_size=batch_size)
    return encoder, EmbeddingCache(cache_dir, encoder)


def _valid_checkpoint(variant: str, checkpoint_dir: Path = V2_CHECKPOINTS) -> bool:
    path, metadata_path = checkpoint_dir / f"V2{variant}.pt", checkpoint_dir / f"V2{variant}.metadata.json"
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = _load_json(metadata_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    expected = {
        "artifact_status": "full", "experiment_status": EXPERIMENT_STATUS, "variant": variant,
        "encoder_model_id": ENCODER_MODEL_ID, "encoder_revision": ENCODER_REVISION,
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator_fingerprint": TRANSLATOR_FINGERPRINT, "v1_source_commit": V1_SOURCE_COMMIT,
    }
    try:
        expected_train_ids = set(_load_json(ROOT / "data/splits/experimental.json")["splits"]["train"])
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError):
        return False
    return (
        all(metadata.get(key) == value for key, value in expected.items())
        and set(metadata.get("train_talk_ids", [])) == expected_train_ids
        and metadata.get("numeric_scaler_fit_split") == "train"
        and metadata.get("checkpoint_sha256") == sha256_file(path)
    )


def train_v2(variant: str, pseudo_manifest: Path | None = None, *, cache_dir: Path = V2_CACHE) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"unknown V2 variant: {variant}")
    if _valid_checkpoint(variant):
        print(f"V2{variant}: validated resume hit")
        return
    manifest_path = pseudo_manifest or PSEUDO / "train/manifest.json"
    if manifest_path.name != "manifest.json":
        raise RuntimeError("V2 training requires a V1 TRAIN manifest")
    manifest, rows = validate_v1_supervision(manifest_path.parent, "train")
    _, cache = _encoder_cache(cache_dir)
    started = time.perf_counter()
    policy, training = train_v2_policy(rows, variant, cache)
    checkpoint_path = V2_CHECKPOINTS / f"V2{variant}.pt"
    checkpoint_hash = save_v2_checkpoint(checkpoint_path, policy)
    metadata = make_checkpoint_metadata(
        variant=variant, manifest=manifest, training=training, checkpoint_hash=checkpoint_hash,
        v2_code_commit=current_git_commit(ROOT), embedding_dimension=cache.dimension,
    )
    metadata["training_seconds"] = time.perf_counter() - started
    atomic_json(V2_CHECKPOINTS / f"V2{variant}.metadata.json", metadata)
    load_v2_checkpoint(checkpoint_path, V2_CHECKPOINTS / f"V2{variant}.metadata.json", cache)
    print(f"trained V2{variant}: states={len(rows)} hash={checkpoint_hash} output={checkpoint_path}")


def _prediction_path(strategy: str, talk_id: str) -> Path:
    return V2_EXPERIMENT / "predictions/dev" / strategy / f"{talk_id}.json"


def rollout_v2(variant: str, thresholds: Sequence[float], batch_size: int = 1) -> None:
    from .cli import _manifests, _runtime_talk, _translator

    reject_test_split("dev")
    if not _valid_checkpoint(variant):
        raise RuntimeError(f"V2{variant} checkpoint is not full/hash-valid")
    _, split = _manifests()
    expected_talks = list(split["splits"]["dev"])
    metadata_path = V2_CHECKPOINTS / f"V2{variant}.metadata.json"
    metadata = _load_json(metadata_path)
    model_hash = metadata["checkpoint_sha256"]
    _, cache = _encoder_cache()
    policy = load_v2_checkpoint(V2_CHECKPOINTS / f"V2{variant}.pt", metadata_path, cache)
    _, provider = _translator(batch_size)
    for threshold in thresholds:
        strategy = _strategy(variant, float(threshold))
        for index, talk_id in enumerate(expected_talks, start=1):
            path = _prediction_path(strategy, talk_id)
            if path.is_file():
                try:
                    validate_prediction_record(_load_json(path), strategy=strategy, talk_id=talk_id, model_hash=model_hash)
                    print(f"{strategy} talk {index}/{len(expected_talks)}: validated resume hit")
                    continue
                except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError):
                    path.unlink()
            talk = _runtime_talk(talk_id, split)
            record = prediction_record(strategy, talk, learned_rollout(talk, provider, policy, float(threshold)))
            record.update({
                "artifact_status": "full", "publishable": False, "experiment_status": EXPERIMENT_STATUS,
                "experiment_label": EXPERIMENT_LABEL, "model_sha256": model_hash,
                "dataset_checksum": DATASET_CHECKSUM, "encoder_revision": ENCODER_REVISION,
            })
            atomic_json(path, record)
            validate_prediction_record(record, strategy=strategy, talk_id=talk_id, model_hash=model_hash)
            print(f"{strategy} talk {index}/{len(expected_talks)}: commits={len(record['commits'])} output={path}")


def evaluate_v2(strategies: Sequence[str] | None = None) -> None:
    from .cli import _talk_paths

    reject_test_split("dev")
    selected = list(strategies or all_v2_strategies())
    all_path = V2_EXPERIMENT / "metrics/dev/all.json"
    split = _load_json(ROOT / "data/splits/experimental.json")
    expected = set(split["splits"]["dev"])
    resumable_predictions = True
    for strategy in selected:
        _, variant, threshold_text = strategy.split("_")
        _strategy(variant, float(threshold_text))
        metadata = _load_json(V2_CHECKPOINTS / f"V2{variant}.metadata.json")
        paths = sorted((V2_EXPERIMENT / "predictions/dev" / strategy).glob("*.json"))
        try:
            records = [_load_json(path) for path in paths]
            if {record.get("talk_id") for record in records} != expected:
                resumable_predictions = False
                break
            for record in records:
                validate_prediction_record(record, strategy=strategy, talk_id=record["talk_id"], model_hash=metadata["checkpoint_sha256"])
        except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, KeyError):
            resumable_predictions = False
            break
    if metrics_are_complete(all_path, selected) and resumable_predictions:
        print("V2 DEV metrics and exact DEV predictions: validated resume hit")
        return
    if not resumable_predictions:
        raise RuntimeError("V2 evaluation cannot resume: predictions lack valid exact DEV coverage")
    all_metrics: dict[str, Any] = {}
    for strategy in selected:
        _, variant, threshold_text = strategy.split("_")
        _strategy(variant, float(threshold_text))
        metadata = _load_json(V2_CHECKPOINTS / f"V2{variant}.metadata.json")
        paths = sorted((V2_EXPERIMENT / "predictions/dev" / strategy).glob("*.json"))
        records = [_load_json(path) for path in paths]
        if {record.get("talk_id") for record in records} != expected:
            raise RuntimeError(f"V2 evaluation requires exact DEV coverage: {strategy}")
        for record in records:
            validate_prediction_record(record, strategy=strategy, talk_id=record["talk_id"], model_hash=metadata["checkpoint_sha256"])
        references = {}
        for record in records:
            document = load_canonical_talk(_talk_paths()[record["talk_id"]])
            references[record["talk_id"]] = " ".join(segment["text"] for segment in document["target_reference"]["segments"])
        metrics = {**quality_metrics(records, references), **latency_metrics(records, references)}
        metrics.update({
            "artifact_status": "full", "publishable": False, "experiment_status": EXPERIMENT_STATUS,
            "experiment_label": EXPERIMENT_LABEL, "dataset_checksum": DATASET_CHECKSUM,
            "encoder_revision": ENCODER_REVISION, "model_sha256": metadata["checkpoint_sha256"],
        })
        all_metrics[strategy] = metrics
        atomic_json(V2_EXPERIMENT / f"metrics/dev/{strategy}.json", metrics)
        print(f"{strategy}: BLEU={metrics['BLEU']:.3f} chrF2={metrics['chrF2']:.3f} AL={metrics['token_level_average_lagging']:.3f} LAAL={metrics['token_level_length_adaptive_average_lagging']:.3f}")
    atomic_json(all_path, all_metrics)


def compare_v1_v2() -> None:
    v1_metrics = _load_json(V1_METRICS)
    v2_path = V2_EXPERIMENT / "metrics/dev/all.json"
    v2_metrics = _load_json(v2_path)
    v1_selection = _load_json(V1_SELECTION)
    required = {"fixed_n_8", "fixed_time_1600", "fixed_time_3200", "local_agreement_la2", "mu_zhang2020"}
    required.update(name for name in v1_metrics if name.startswith("learned_P"))
    required.add(v1_selection["selected_strategy"])
    if not required.issubset(v1_metrics):
        raise RuntimeError("immutable V1 metrics lack required comparison strategies")
    comparison = {
        "experiment_status": EXPERIMENT_STATUS, "experiment_label": EXPERIMENT_LABEL,
        "v1_checkpoint_stage": v1_identity_document()["checkpoint_stage"],
        "v1_selected_strategy": v1_selection["selected_strategy"],
        "v1_metrics_sha256": sha256_file(V1_METRICS), "v2_metrics_sha256": sha256_file(v2_path),
        "v1_metrics": {name: v1_metrics[name] for name in sorted(required)}, "v2_metrics": v2_metrics,
    }
    output = V2_EXPERIMENT / "comparison-v1-v2.json"
    if output.is_file():
        existing = _load_json(output)
        if (
            existing.get("experiment_status") == EXPERIMENT_STATUS
            and existing.get("v1_metrics_sha256") == comparison["v1_metrics_sha256"]
            and existing.get("v2_metrics_sha256") == comparison["v2_metrics_sha256"]
            and set(existing.get("v2_metrics", {})) == set(v2_metrics)
        ):
            print("V1/V2 comparison: validated resume hit")
            return
    atomic_json(output, comparison)


def select_v2() -> None:
    v1_metrics, v2_metrics = _load_json(V1_METRICS), _load_json(V2_EXPERIMENT / "metrics/dev/all.json")
    if any(row.get("artifact_status") != "full" for row in v2_metrics.values()):
        raise RuntimeError("V2 selection refuses partial metrics")
    metrics_hash = sha256_file(V2_EXPERIMENT / "metrics/dev/all.json")
    output = V2_EXPERIMENT / "dev-selection.json"
    if output.is_file():
        existing = _load_json(output)
        if (
            existing.get("experiment_status") == EXPERIMENT_STATUS
            and existing.get("selected_strategy") in v2_metrics
            and existing.get("v1_metrics_sha256") == sha256_file(V1_METRICS)
            and existing.get("v2_metrics_sha256") == metrics_hash
        ):
            print("V2 DEV selection: validated resume hit")
            return
    result = select_v2_configuration(v1_metrics, v2_metrics)
    result.update({
        "v1_metrics_sha256": sha256_file(V1_METRICS),
        "v2_metrics_sha256": metrics_hash,
    })
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


def freeze_v2() -> None:
    path = V2_EXPERIMENT / "v2-frozen-config.json"
    if path.exists():
        existing = _load_json(path)
        if existing.get("artifact_status") == "v2-dev-frozen-complete" and existing.get("experiment_status") == EXPERIMENT_STATUS:
            print("V2 freeze: validated resume hit")
            return
        raise RuntimeError("refusing to overwrite invalid V2 frozen config")
    train_manifest, _ = validate_v1_supervision(PSEUDO / "train", "train")
    dev_manifest, _ = validate_v1_supervision(PSEUDO / "dev", "dev")
    hashes = {}
    for variant in VARIANTS:
        if not _valid_checkpoint(variant):
            raise RuntimeError(f"freeze requires a valid V2{variant} checkpoint")
        hashes[variant] = sha256_file(V2_CHECKPOINTS / f"V2{variant}.pt")
    metrics_path, selection_path = V2_EXPERIMENT / "metrics/dev/all.json", V2_EXPERIMENT / "dev-selection.json"
    if not metrics_are_complete(metrics_path, all_v2_strategies()) or not selection_path.is_file():
        raise RuntimeError("freeze requires full V2 DEV metrics and selection")
    selection = _load_json(selection_path)
    if (
        selection.get("experiment_status") != EXPERIMENT_STATUS
        or selection.get("selected_strategy") not in all_v2_strategies()
        or selection.get("v2_metrics_sha256") != sha256_file(metrics_path)
    ):
        raise RuntimeError("freeze requires an identity-valid V2 DEV selection")
    document = {
        "artifact_status": "v2-dev-frozen-complete", "publishable": False,
        "experiment_status": EXPERIMENT_STATUS, "experiment_label": EXPERIMENT_LABEL,
        "v1_source_identity": v1_identity_document(), "v2_code_commit": current_git_commit(ROOT),
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator_fingerprint": TRANSLATOR_FINGERPRINT,
        "encoder": {"model_id": ENCODER_MODEL_ID, "revision": ENCODER_REVISION, "pooling": POOLING_VERSION, "frozen": True},
        "thresholds": list(THRESHOLDS), "selection_rule_identity": "V1 select_dev_configuration (unchanged adapter)",
        "checkpoint_hashes": hashes, "metrics_sha256": sha256_file(metrics_path),
        "selection_sha256": sha256_file(selection_path), "train_manifest_checksum": train_manifest["config_checksum"],
        "dev_manifest_checksum": dev_manifest["config_checksum"],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "test_status": "UNTOUCHED",
    }
    atomic_json(path, document)
    atomic_json(V2_EXPERIMENT / "run-provenance.json", {
        "experiment_status": EXPERIMENT_STATUS, "v1_source_commit": V1_SOURCE_COMMIT,
        "v2_code_commit": document["v2_code_commit"], "frozen_config_sha256": sha256_file(path),
    })


def import_v1(source: Path) -> None:
    identity = restore_v1_artifacts(source, ROOT)
    print(json.dumps(identity, indent=2))


def emergency_checkpoint(stage: str) -> dict[str, Any]:
    allowed = {"v2-models-trained", "v2-dev-rollouts-complete", "v2-dev-frozen-complete"}
    if stage not in allowed:
        raise ValueError(f"invalid V2 checkpoint stage: {stage}")
    return {
        "checkpoint_stage": stage, "experiment_status": EXPERIMENT_STATUS,
        "v1_source_identity": v1_identity_document(), "v2_code_commit": current_git_commit(ROOT),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
