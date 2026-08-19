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


def _strategy(threshold: float, prepared_context_mode: str) -> str:
    return f"p3_global_{threshold:.2f}" if prepared_context_mode == "real" else f"p3_global_zeroctx_{threshold:.2f}"


def attach_prepared_context_provenance(record: dict[str, Any], prepared: Any, prepared_context_mode: str) -> None:
    """Record source eligibility separately from the vector injected into P3."""
    provenance = prepared.provenance(prepared_context_mode)
    record["prepared_context"] = provenance
    record["prepared_context_mode"] = prepared_context_mode
    record["prepared_context_effective_embedding_norm"] = provenance["prepared_context_effective_embedding_norm"]


def rollout_p3(split: str, thresholds: Sequence[float], *, talk_id: str | None = None, batch_size: int = 1,
               prepared_context_mode: str = "real", encoder_device: str | None = None,
               encoder_dtype: str = "float32", policy_device: str | None = None,
               policy_dtype: str = "float32", translator_device: str = "cuda",
               translator_dtype: str = "float16", trace_output: Path | None = None) -> None:
    if split not in {"train", "dev"} or batch_size != 1:
        raise RuntimeError("P3_GLOBAL rollout permits TRAIN/DEV only and requires translator batch size 1")
    if trace_output is not None:
        if split != "dev":
            raise RuntimeError("P3 demo traces permit DEV only")
        if talk_id is None:
            raise RuntimeError("P3 demo traces require exactly one --talk-id")
        if len(thresholds) != 1:
            raise RuntimeError("P3 demo traces require exactly one threshold")
    if any(threshold not in THRESHOLDS for threshold in thresholds):
        raise RuntimeError("P3_GLOBAL threshold is outside the frozen grid")
    if prepared_context_mode not in {"real", "zero"}:
        raise ValueError("P3_GLOBAL prepared context mode must be real or zero")
    if encoder_dtype != "float32" or policy_dtype != "float32" or translator_dtype != "float16":
        raise RuntimeError("P3_GLOBAL rollout requires float32 encoder/policy and float16 translator")
    from .cli import _manifests, _runtime_talk, _translator
    _, split_manifest = _manifests()
    talks = list(split_manifest["splits"][split])
    if talk_id is not None:
        if talk_id not in talks:
            raise ValueError(f"talk {talk_id} does not belong to {split}")
        talks = [talk_id]
    runtime = p3_runtime()
    if encoder_device is not None:
        runtime["encoder_device"] = torch.device(encoder_device)
    encoder, cache = _encoder_cache(runtime)
    _, provider = _translator(batch_size, device=translator_device)
    metadata_path = P3_CHECKPOINTS / "P3_GLOBAL.metadata.json"
    for selected_id in talks:
        prepared = build_prepared_global_embedding(load_matching_pool(PREPARED_ROOT, talk_id=selected_id, split=split), encoder, cache)
        policy = load_p3_checkpoint(
            P3_CHECKPOINTS / "P3_GLOBAL.pt", metadata_path, cache, prepared,
            # Existing canonical P3 rollout loaded inference policy on CPU.
            manifest_path=PREPARED_MANIFEST, device=policy_device or "cpu",
            prepared_context_mode=prepared_context_mode,
        )
        for threshold in thresholds:
            strategy = _strategy(threshold, prepared_context_mode)
            talk = _runtime_talk(selected_id, split_manifest)
            events: list[dict[str, Any]] = []
            trace_sink = (lambda event: events.append(dict(event))) if trace_output is not None else None
            commits = learned_rollout(talk, provider, policy, threshold, trace_sink)
            record = prediction_record(strategy, talk, commits)
            attach_prepared_context_provenance(record, prepared, prepared_context_mode)
            if trace_output is None:
                atomic_json(P3_ROOT / "predictions" / split / strategy / f"{selected_id}.json", record)
            else:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                trace_commits = [event for event in events if event["decision"] == "COMMIT"]
                if len(trace_commits) != len(commits):
                    raise RuntimeError("P3 trace commit events do not match canonical commits")
                for event, commit in zip(trace_commits, commits):
                    if (
                        event["candidate_source_start"], event["candidate_source_end"],
                        event["committed_target_text"], event["observation_ms"], event["decision_reason"],
                    ) != (
                        commit.source_start, commit.source_end, commit.translated_text,
                        commit.observation_emit_ms, commit.reason,
                    ):
                        raise RuntimeError("P3 trace commit event differs from canonical commit")
                trace = {
                    "artifact_version": "demo-policy-trace-v1",
                    "talk_id": talk.talk_id,
                    "split": talk.split,
                    "strategy": strategy,
                    "threshold": threshold,
                    "prepared_context_mode": prepared_context_mode,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "prepared_context": prepared.provenance(prepared_context_mode),
                    "source_token_count": len(talk.tokens),
                    "source_final_emit_ms": talk.tokens[-1].emit_ms,
                    "source_stream": {
                        "source_token_count": len(talk.tokens),
                        "source_final_emit_ms": talk.tokens[-1].emit_ms,
                        "clock": "simulated_source_emit_ms",
                        "observation_key": "source_token_end",
                    },
                    "events": events,
                }
                atomic_json(trace_output, trace)


def evaluate_p3(strategies: Sequence[str] | None = None) -> None:
    from .evaluation import latency_metrics, quality_metrics
    from .cli import _talk_paths
    from timelymt.data.canonical.core import load_canonical_talk
    selected = list(strategies or [f"p3_global_{threshold:.2f}" for threshold in THRESHOLDS])
    zero_context = [strategy.startswith("p3_global_zeroctx_") for strategy in selected]
    if any(zero_context) and not all(zero_context):
        raise RuntimeError("P3 evaluation cannot mix real- and zero-context strategies")
    result = {}
    for strategy in selected:
        records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((P3_ROOT / "predictions/dev" / strategy).glob("*.json"))]
        if not records:
            raise RuntimeError(f"no P3_GLOBAL DEV predictions for {strategy}")
        references = {record["talk_id"]: " ".join(segment["text"] for segment in load_canonical_talk(_talk_paths()[record["talk_id"]])["target_reference"]["segments"]) for record in records}
        result[strategy] = {**quality_metrics(records, references), **latency_metrics(records, references)}
    output = P3_ROOT / "metrics/dev/all-zeroctx.json" if all(zero_context) else P3_ROOT / "metrics/dev/all.json"
    atomic_json(output, result)
