"""Granular operator stages for P3-GLOBAL. No stage accesses TEST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from timelymt.data.prepared_context import load_prepared_context
from .policy_v2 import EmbeddingCache, FrozenMiniLMEncoder, atomic_json, validate_v1_supervision
from .policy_p3_global import (
    P3_VARIANT, build_prepared_global_embedding, load_matching_pool, load_p3_checkpoint,
    make_p3_checkpoint_metadata, prepared_manifest_fingerprint, save_p3_checkpoint,
    train_p3_global_policy, validate_pool_identity,
)
from .streaming import learned_rollout, prediction_record


ROOT = Path(__file__).parents[3]
P3_ROOT = ROOT / "outputs/experiments/policy-p3-global"
P3_CHECKPOINTS = ROOT / "checkpoints/policy_p3_global"
P3_CACHE = P3_ROOT / "embedding-cache"
PREPARED_ROOT = ROOT / "data/prepared_context"
PREPARED_MANIFEST = PREPARED_ROOT / "manifest.json"
PSEUDO = ROOT / "data/policy/pseudo_labels"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def _encoder_cache():
    encoder = FrozenMiniLMEncoder(device="cpu", dtype="float32")
    return encoder, EmbeddingCache(P3_CACHE, encoder)


def validate_prepared_p3() -> None:
    manifest = json.loads(PREPARED_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "prepared-context-v0":
        raise RuntimeError("unsupported prepared context manifest schema")
    for entry in manifest.get("pools", []):
        split, talk_id = entry["split"], entry["talk_id"]
        if split not in {"train", "dev"}:
            raise RuntimeError("P3_GLOBAL prepared context manifest contains forbidden split")
        pool = load_prepared_context(PREPARED_ROOT / entry["path"])
        validate_pool_identity(pool, talk_id=talk_id, split=split)
    print(json.dumps({"manifest": str(PREPARED_MANIFEST), "fingerprint": prepared_manifest_fingerprint(PREPARED_MANIFEST), "pool_count": manifest.get("pool_count"), "eligible_source_count": manifest.get("eligible_source_count")}, indent=2))


def inspect_p3(talk_id: str, split: str) -> None:
    pool = load_matching_pool(PREPARED_ROOT, talk_id=talk_id, split=split)
    sources = sorted(pool.eligible_sources(), key=lambda source: source.source_id)
    print(json.dumps({"talk_id": talk_id, "split": split, "prepared_pool": str(PREPARED_ROOT / split / f"{talk_id}.json"), "eligible_source_ids": [source.source_id for source in sources], "source_count": len(sources), "prepared_embedding_dimension": 384, "feature_dimension": 1547, "note": "embedding norm requires the pinned MiniLM and is intentionally not loaded by inspect"}, indent=2))


def _prepared_by_talk(rows: Sequence[dict[str, Any]], cache: EmbeddingCache) -> dict[str, Any]:
    encoder = cache.encoder
    result = {}
    for talk_id in sorted({row["talk_id"] for row in rows}):
        split = next(row["split"] for row in rows if row["talk_id"] == talk_id)
        pool = load_matching_pool(PREPARED_ROOT, talk_id=talk_id, split=split)
        result[talk_id] = build_prepared_global_embedding(pool, encoder, cache)
    return result


def train_p3() -> None:
    manifest, rows = validate_v1_supervision(PSEUDO / "train", "train")
    _, cache = _encoder_cache()
    prepared = _prepared_by_talk(rows, cache)
    policy, training = train_p3_global_policy(rows, prepared, cache)
    checkpoint = P3_CHECKPOINTS / "P3_GLOBAL.pt"
    digest = save_p3_checkpoint(checkpoint, policy)
    metadata = make_p3_checkpoint_metadata(checkpoint_hash=digest, prepared_manifest=PREPARED_MANIFEST, train_talk_ids=manifest["talk_ids"], training=training, scaler=policy.scaler)
    atomic_json(P3_CHECKPOINTS / "P3_GLOBAL.metadata.json", metadata)
    print(f"trained {P3_VARIANT}: states={len(rows)} output={checkpoint}")


def inspect_p3_checkpoint() -> None:
    path = P3_CHECKPOINTS / "P3_GLOBAL.metadata.json"
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, sort_keys=True))


def rollout_p3(split: str, thresholds: Sequence[float], *, talk_id: str | None = None, batch_size: int = 1) -> None:
    if split not in {"train", "dev"} or batch_size != 1:
        raise RuntimeError("P3_GLOBAL rollout permits TRAIN/DEV only and requires translator batch size 1")
    if any(threshold not in THRESHOLDS for threshold in thresholds):
        raise RuntimeError("P3_GLOBAL threshold is outside the frozen grid")
    from .cli import _manifests, _runtime_talk, _translator
    _, split_manifest = _manifests()
    talks = list(split_manifest["splits"][split])
    if talk_id is not None:
        if talk_id not in talks:
            raise ValueError(f"talk {talk_id} does not belong to {split}")
        talks = [talk_id]
    encoder, cache = _encoder_cache()
    _, provider = _translator(batch_size, device="cuda")
    metadata_path = P3_CHECKPOINTS / "P3_GLOBAL.metadata.json"
    for selected_id in talks:
        prepared = build_prepared_global_embedding(load_matching_pool(PREPARED_ROOT, talk_id=selected_id, split=split), encoder, cache)
        policy = load_p3_checkpoint(P3_CHECKPOINTS / "P3_GLOBAL.pt", metadata_path, cache, prepared, manifest_path=PREPARED_MANIFEST)
        for threshold in thresholds:
            strategy = f"p3_global_{threshold:.2f}"
            talk = _runtime_talk(selected_id, split_manifest)
            record = prediction_record(strategy, talk, learned_rollout(talk, provider, policy, threshold))
            record["prepared_context"] = prepared.provenance()
            atomic_json(P3_ROOT / "predictions" / split / strategy / f"{selected_id}.json", record)


def evaluate_p3(strategies: Sequence[str] | None = None) -> None:
    from .evaluation import latency_metrics, quality_metrics
    from .cli import _talk_paths
    from timelymt.data.canonical.core import load_canonical_talk
    selected = list(strategies or [f"p3_global_{threshold:.2f}" for threshold in THRESHOLDS])
    result = {}
    for strategy in selected:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((P3_ROOT / "predictions/dev" / strategy).glob("*.json"))]
        if not records:
            raise RuntimeError(f"no P3_GLOBAL DEV predictions for {strategy}")
        references = {record["talk_id"]: " ".join(segment["text"] for segment in load_canonical_talk(_talk_paths()[record["talk_id"]])["target_reference"]["segments"]) for record in records}
        result[strategy] = {**quality_metrics(records, references), **latency_metrics(records, references)}
    atomic_json(P3_ROOT / "metrics/dev/all.json", result)
