"""Granular operator stages for P3-GLOBAL. No stage accesses TEST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from timelymt.data.prepared_context import load_prepared_context
from .policy_v2 import EmbeddingCache, FrozenMiniLMEncoder, atomic_json, validate_v1_supervision
from .policy_p3_global import (
    P3_VARIANT, build_prepared_global_embedding, load_matching_pool, load_p3_checkpoint,
    make_p3_checkpoint_metadata, prepared_manifest_fingerprint, save_p3_checkpoint,
    prepare_p3_text_embeddings, train_p3_global_policy, validate_pool_identity,
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


def p3_runtime(config_path: Path = ROOT / "configs/experiments/policy-p3-global.json") -> dict[str, Any]:
    runtime = json.loads(config_path.read_text(encoding="utf-8")).get("runtime", {})
    def resolve(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("P3 CUDA was explicitly requested but is unavailable")
        if requested not in {"cpu", "cuda"}:
            raise ValueError(f"invalid P3 device: {requested}")
        return torch.device(requested)
    encoder_device, policy_device = resolve(runtime.get("encoder_device", "auto")), resolve(runtime.get("policy_device", "auto"))
    batch_size = int(runtime.get("encoder_batch_size", 256))
    if batch_size <= 0:
        raise ValueError("P3 encoder_batch_size must be positive")
    return {"encoder_device": encoder_device, "policy_device": policy_device, "encoder_batch_size": batch_size}


def p3_runtime_provenance(runtime: dict[str, Any]) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {"encoder_device_used": str(runtime["encoder_device"]), "policy_device_used": str(runtime["policy_device"]),
            "torch_version": torch.__version__, "cuda_available": cuda_available,
            "cuda_device_name": torch.cuda.get_device_name(runtime["encoder_device"]) if cuda_available and runtime["encoder_device"].type == "cuda" else None}


def _encoder_cache(runtime: dict[str, Any] | None = None):
    runtime = runtime or p3_runtime()
    encoder = FrozenMiniLMEncoder(device=runtime["encoder_device"], dtype="float32", batch_size=runtime["encoder_batch_size"], allow_cuda_float32=True)
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


def _pools_by_talk(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {talk_id: load_matching_pool(PREPARED_ROOT, talk_id=talk_id, split=next(row["split"] for row in rows if row["talk_id"] == talk_id)) for talk_id in sorted({row["talk_id"] for row in rows})}


def _prepared_by_talk(pools: dict[str, Any], cache: EmbeddingCache) -> dict[str, Any]:
    encoder = cache.encoder
    return {talk_id: build_prepared_global_embedding(pool, encoder, cache) for talk_id, pool in pools.items()}


def train_p3() -> None:
    print("P3 TRAIN")
    runtime = p3_runtime()
    provenance = p3_runtime_provenance(runtime)
    print(f"device encoder: {provenance['encoder_device_used']}")
    print(f"device policy: {provenance['policy_device_used']}")
    print(f"torch.cuda.is_available(): {provenance['cuda_available']}")
    print(f"CUDA device: {provenance['cuda_device_name']}")
    manifest, rows = validate_v1_supervision(PSEUDO / "train", "train")
    _, cache = _encoder_cache(runtime)
    print(f"MiniLM parameter device: {next(cache.encoder.model.parameters()).device}")
    pools = _pools_by_talk(rows)
    print("Preparing text embeddings...")
    embedding_stats = prepare_p3_text_embeddings(rows, list(pools.values()), cache)
    for key in ("unique_texts", "cache_hits", "cache_misses", "batches"):
        print(f"{key.replace('_', ' ')}: {embedding_stats[key]}")
    prepared = _prepared_by_talk(pools, cache)
    print("Building feature matrix...")
    print(f"shape: ({len(rows)}, 1547)")
    print("Training...")
    policy, training = train_p3_global_policy(rows, prepared, cache, device=runtime["policy_device"], progress=lambda epoch, total, loss: print(f"epoch {epoch}/{total} loss={loss:.6f}"))
    print(f"P3 MLP device: {next(policy.model.parameters()).device}")
    checkpoint = P3_CHECKPOINTS / "P3_GLOBAL.pt"
    digest = save_p3_checkpoint(checkpoint, policy)
    metadata = make_p3_checkpoint_metadata(checkpoint_hash=digest, prepared_manifest=PREPARED_MANIFEST, train_talk_ids=manifest["talk_ids"], training=training, scaler=policy.scaler, runtime=provenance)
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
