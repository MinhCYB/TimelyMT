"""P3-GLOBAL prepared-context policy, isolated from frozen P0/P1/P2 semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from timelymt.data.prepared_context import PreparedContextPool, load_prepared_context
from timelymt.data.translation_artifacts import stable_fingerprint
from .policy import NUMERIC_FEATURES
from .policy_v2 import (
    ENCODER_MODEL_ID, ENCODER_REVISION, EXPECTED_EMBEDDING_DIMENSION, POOLING_VERSION,
    EmbeddingCache, NumericScaler, V2MLP, _atomic_torch_save, class_weight, numeric_vector,
    sha256_file, validate_causal_state,
)


P3_VARIANT = "P3_GLOBAL"
PREPARED_REPRESENTATION_VERSION = "prepared-global-v0"
P3_FEATURE_ORDER = (
    "current_source_embedding",
    "previous_committed_source_embedding",
    "previous_generated_target_embedding",
    "prepared_global_embedding",
    "scaled_numeric_features",
)
P3_INPUT_DIMENSION = EXPECTED_EMBEDDING_DIMENSION * 4 + len(NUMERIC_FEATURES)


@dataclass(frozen=True)
class PreparedGlobalEmbedding:
    talk_id: str
    split: str
    eligible_source_ids: tuple[str, ...]
    eligible_source_checksums: tuple[str, ...]
    embedding: np.ndarray
    representation_version: str = PREPARED_REPRESENTATION_VERSION

    @property
    def source_count(self) -> int:
        return len(self.eligible_source_ids)

    @property
    def has_eligible_context(self) -> bool:
        return bool(self.eligible_source_ids)

    @property
    def embedding_dimension(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def embedding_norm(self) -> float:
        return float(np.linalg.norm(self.embedding))

    def provenance(self) -> dict[str, Any]:
        return {
            "talk_id": self.talk_id, "split": self.split,
            "eligible_source_ids": list(self.eligible_source_ids),
            "eligible_source_checksums": list(self.eligible_source_checksums),
            "source_count": self.source_count, "representation_version": self.representation_version,
            "embedding_dimension": self.embedding_dimension, "embedding_norm": self.embedding_norm,
            "has_eligible_context": self.has_eligible_context,
        }


def validate_pool_identity(pool: PreparedContextPool, *, talk_id: str, split: str) -> None:
    if split.lower() == "test":
        raise RuntimeError("P3_GLOBAL prepared context forbids TEST")
    if pool.talk_id != talk_id:
        raise RuntimeError(f"prepared context talk_id mismatch: expected={talk_id} actual={pool.talk_id}")
    if pool.split != split:
        raise RuntimeError(f"prepared context split mismatch: expected={split} actual={pool.split}")


def load_matching_pool(root: Path, *, talk_id: str, split: str) -> PreparedContextPool:
    if split.lower() == "test":
        raise RuntimeError("P3_GLOBAL prepared context forbids TEST")
    path = root / split / f"{talk_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared context pool is missing: {path}")
    pool = load_prepared_context(path)
    validate_pool_identity(pool, talk_id=talk_id, split=split)
    return pool


def build_prepared_global_embedding(pool: PreparedContextPool, encoder: Any, cache: Any | None = None) -> PreparedGlobalEmbedding:
    """Encode each eligible source independently, then equal-average and normalize."""
    dimension = int(encoder.dimension)
    if dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise RuntimeError(f"P3_GLOBAL requires {EXPECTED_EMBEDDING_DIMENSION}-d MiniLM embeddings")
    sources = tuple(sorted(pool.eligible_sources(), key=lambda source: source.source_id))
    if not sources:
        embedding = np.zeros(dimension, dtype=np.float32)
    else:
        embedder = cache if cache is not None else encoder
        values = np.asarray(embedder.encode([source.text for source in sources]), dtype=np.float32)
        if values.shape != (len(sources), dimension):
            raise RuntimeError("prepared source encoder returned an invalid embedding matrix")
        if len(sources) == 1:
            embedding = values[0].astype(np.float32, copy=True)
        else:
            mean = values.mean(axis=0, dtype=np.float32)
            norm = float(np.linalg.norm(mean))
            embedding = np.zeros(dimension, dtype=np.float32) if norm == 0.0 else (mean / norm).astype(np.float32)
    return PreparedGlobalEmbedding(
        talk_id=pool.talk_id, split=pool.split,
        eligible_source_ids=tuple(source.source_id for source in sources),
        eligible_source_checksums=tuple(source.checksum for source in sources), embedding=embedding,
    )


def prepare_p3_text_embeddings(rows: Sequence[Mapping[str, Any]], pools: Sequence[PreparedContextPool], cache: EmbeddingCache) -> dict[str, int]:
    """Materialize every exact P3 text identity once before feature construction."""
    texts: list[str] = []
    for pool in pools:
        texts.extend(source.text for source in pool.eligible_sources())
    for row in rows:
        state = row["causal"]
        validate_causal_state(state)
        texts.extend((state["current_source_text"], state["previous_committed_source_text"], state["previous_committed_target_text"]))
    unique = list(dict.fromkeys(text for text in texts if text != ""))
    hits = sum(cache._read(text) is not None for text in unique)
    cache.encode(unique)
    batch_size = int(getattr(cache.encoder, "batch_size", len(unique) or 1))
    return {"unique_texts": len(unique), "cache_hits": hits, "cache_misses": len(unique) - hits, "batches": (len(unique) - hits + batch_size - 1) // batch_size}


def p3_feature_vector(state: Mapping[str, Any], prepared: PreparedGlobalEmbedding, cache: EmbeddingCache, scaler: NumericScaler) -> np.ndarray:
    validate_causal_state(state)
    if prepared.embedding.shape != (cache.dimension,) or prepared.embedding.dtype != np.float32:
        raise RuntimeError("invalid P3 prepared embedding")
    texts = (
        state["current_source_text"], state["previous_committed_source_text"],
        state["previous_committed_target_text"],
    )
    feature = np.concatenate([
        *(cache.encode([text])[0] for text in texts), prepared.embedding,
        scaler.transform(numeric_vector(state)[None, :])[0],
    ]).astype(np.float32, copy=False)
    if feature.shape != (P3_INPUT_DIMENSION,):
        raise RuntimeError("P3_GLOBAL feature dimension mismatch")
    return feature


def build_p3_feature_matrix(rows: Sequence[Mapping[str, Any]], prepared_by_talk: Mapping[str, PreparedGlobalEmbedding], cache: EmbeddingCache, scaler: NumericScaler) -> np.ndarray:
    texts = [
        text
        for row in rows
        for text in (
            row["causal"]["current_source_text"], row["causal"]["previous_committed_source_text"], row["causal"]["previous_committed_target_text"],
        )
    ]
    encoded = cache.encode(list(dict.fromkeys(texts)))
    embeddings = dict(zip(dict.fromkeys(texts), encoded, strict=True))
    features = []
    for row in rows:
        talk_id = row.get("talk_id")
        if not isinstance(talk_id, str):
            raise RuntimeError("P3 supervision row is missing talk_id")
        prepared = prepared_by_talk.get(talk_id)
        if prepared is None:
            raise RuntimeError(f"missing P3 prepared embedding for supervision talk: {talk_id}")
        if prepared.talk_id != talk_id or prepared.split != row.get("split"):
            raise RuntimeError(f"P3 prepared embedding identity mismatch for supervision talk: {talk_id}")
        state = row["causal"]
        validate_causal_state(state)
        feature = np.concatenate([
            embeddings[state["current_source_text"]], embeddings[state["previous_committed_source_text"]],
            embeddings[state["previous_committed_target_text"]], prepared.embedding,
            scaler.transform(numeric_vector(state)[None, :])[0],
        ]).astype(np.float32, copy=False)
        if feature.shape != (P3_INPUT_DIMENSION,):
            raise RuntimeError("P3_GLOBAL feature dimension mismatch")
        features.append(feature)
    return np.stack(features).astype(np.float32, copy=False)


@dataclass
class P3GlobalPolicy:
    model: V2MLP
    scaler: NumericScaler
    cache: EmbeddingCache
    prepared: PreparedGlobalEmbedding
    device: torch.device
    variant: str = P3_VARIANT

    def predict_commit_probability(self, state: Mapping[str, Any]) -> float:
        self.model.eval()
        features = p3_feature_vector(state, self.prepared, self.cache, self.scaler)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)))[0].cpu())


def train_p3_global_policy(rows: Sequence[Mapping[str, Any]], prepared_by_talk: Mapping[str, PreparedGlobalEmbedding], cache: EmbeddingCache, *, epochs: int = 20, batch_size: int = 256, device: str | torch.device = "cpu", progress: Callable[[int, int, float], None] | None = None) -> tuple[P3GlobalPolicy, dict[str, Any]]:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("P3_GLOBAL training accepts TRAIN supervision only")
    numeric = np.stack([numeric_vector(row["causal"]) for row in rows])
    scaler = NumericScaler.fit(numeric, split_name="train")
    features = build_p3_feature_matrix(rows, prepared_by_talk, cache, scaler)
    labels = np.asarray([1.0 if row["label"] == "COMMIT" else 0.0 for row in rows], dtype=np.float32)
    counts = {"LISTEN": int((labels == 0).sum()), "COMMIT": int((labels == 1).sum())}
    weight = class_weight(counts)
    torch.manual_seed(20260809)
    model = V2MLP(P3_INPUT_DIMENSION).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(20260809))
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            seen += len(batch_labels)
        epoch_loss = total_loss / seen
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        if progress is not None:
            progress(epoch + 1, epochs, epoch_loss)
    first = prepared_by_talk[rows[0]["talk_id"]]
    return P3GlobalPolicy(model, scaler, cache, first, torch.device(device)), {"label_counts": counts, "positive_weight": weight, "training_history": history}


def prepared_manifest_fingerprint(path: Path) -> str:
    return sha256_file(path)


def checkpoint_payload(policy: P3GlobalPolicy) -> dict[str, Any]:
    return {"format_version": "1.0.0", "variant": P3_VARIANT, "input_dimension": P3_INPUT_DIMENSION,
            "embedding_dimension": policy.cache.dimension, "scaler": asdict(policy.scaler),
            "model_state_dict": {name: value.detach().cpu().float() for name, value in policy.model.state_dict().items()}}


def save_p3_checkpoint(path: Path, policy: P3GlobalPolicy) -> str:
    _atomic_torch_save(path, checkpoint_payload(policy))
    return sha256_file(path)


def make_p3_checkpoint_metadata(*, checkpoint_hash: str, prepared_manifest: Path, train_talk_ids: Sequence[str], training: Mapping[str, Any], scaler: NumericScaler, runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    metadata = {"variant": P3_VARIANT, "input_dimension": P3_INPUT_DIMENSION,
            "prepared_context_schema_version": "prepared-context-v0", "prepared_representation_version": PREPARED_REPRESENTATION_VERSION,
            "prepared_context_manifest_fingerprint": prepared_manifest_fingerprint(prepared_manifest),
            "encoder_model_id": ENCODER_MODEL_ID, "encoder_revision": ENCODER_REVISION, "pooling_version": POOLING_VERSION,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION, "numeric_feature_ordering": list(NUMERIC_FEATURES),
            "numeric_scaler_fit_split": "train", "numeric_scaler_fingerprint": stable_fingerprint(asdict(scaler)),
            "train_talk_ids": sorted(train_talk_ids), "checkpoint_sha256": checkpoint_hash,
            "mlp_architecture": ["Linear(input,256)", "GELU", "Dropout(0.20)", "Linear(256,64)", "GELU", "Dropout(0.10)", "Linear(64,1)"],
            "training_hyperparameters": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "epochs": 20, "seed": 20260809, "weighted_bce": "TRAIN_LISTEN/TRAIN_COMMIT"},
            "label_counts": dict(training["label_counts"]), "positive_weight": training["positive_weight"]}
    if runtime is not None:
        metadata["runtime"] = dict(runtime)
    return metadata


def validate_p3_checkpoint_metadata(metadata: Mapping[str, Any], payload: Mapping[str, Any], *, manifest_path: Path, cache: EmbeddingCache) -> None:
    expected = {"variant": P3_VARIANT, "input_dimension": P3_INPUT_DIMENSION, "prepared_representation_version": PREPARED_REPRESENTATION_VERSION,
                "prepared_context_schema_version": "prepared-context-v0", "prepared_context_manifest_fingerprint": prepared_manifest_fingerprint(manifest_path), "encoder_model_id": ENCODER_MODEL_ID,
                "encoder_revision": ENCODER_REVISION, "pooling_version": POOLING_VERSION, "embedding_dimension": cache.dimension,
                "numeric_feature_ordering": list(NUMERIC_FEATURES), "numeric_scaler_fit_split": "train",
                "mlp_architecture": ["Linear(input,256)", "GELU", "Dropout(0.20)", "Linear(256,64)", "GELU", "Dropout(0.10)", "Linear(64,1)"],
                "training_hyperparameters": {"optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 256, "epochs": 20, "seed": 20260809, "weighted_bce": "TRAIN_LISTEN/TRAIN_COMMIT"}}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P3_GLOBAL checkpoint metadata identity mismatch")
    if payload.get("variant") != P3_VARIANT or payload.get("input_dimension") != P3_INPUT_DIMENSION or payload.get("embedding_dimension") != cache.dimension:
        raise RuntimeError("P3_GLOBAL checkpoint payload identity mismatch")
    scaler = payload.get("scaler", {})
    if scaler.get("fitted_split") != "train" or len(scaler.get("mean", ())) != len(NUMERIC_FEATURES) or len(scaler.get("scale", ())) != len(NUMERIC_FEATURES):
        raise RuntimeError("P3_GLOBAL checkpoint scaler identity mismatch")
    if not all(np.isfinite(value) for value in (*scaler["mean"], *scaler["scale"])) or any(float(value) <= 0 for value in scaler["scale"]):
        raise RuntimeError("P3_GLOBAL checkpoint scaler values are invalid")
    if metadata.get("numeric_scaler_fingerprint") != stable_fingerprint(scaler):
        raise RuntimeError("P3_GLOBAL checkpoint scaler fingerprint mismatch")
    try:
        V2MLP(P3_INPUT_DIMENSION).load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise RuntimeError("P3_GLOBAL checkpoint MLP architecture mismatch") from error


def load_p3_checkpoint(path: Path, metadata_path: Path, cache: EmbeddingCache, prepared: PreparedGlobalEmbedding, *, manifest_path: Path, device: str | torch.device = "cpu") -> P3GlobalPolicy:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError("P3_GLOBAL checkpoint SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_p3_checkpoint_metadata(metadata, payload, manifest_path=manifest_path, cache=cache)
    scaler_data = payload["scaler"]
    scaler = NumericScaler(tuple(scaler_data["mean"]), tuple(scaler_data["scale"]), scaler_data["fitted_split"])
    model = V2MLP(P3_INPUT_DIMENSION)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval().to(device)
    return P3GlobalPolicy(model, scaler, cache, prepared, torch.device(device))
