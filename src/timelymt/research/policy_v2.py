"""Post-hoc exploratory contextual/nonlinear policy extension.

V1 pseudo-labels are immutable upstream supervision.  This module never
generates pseudo-labels and rejects TEST for every V2 experimental operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
from torch import nn

from timelymt.data.translation_artifacts import stable_fingerprint
from .policy import NUMERIC_FEATURES, VARIANTS
from .streaming import select_dev_configuration


EXPERIMENT_STATUS = "post_hoc_exploratory"
EXPERIMENT_LABEL = "POST-HOC EXPLORATORY DEV EXTENSION"
ENCODER_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODER_REVISION = "e62509716f15c5fd03a6fd3156a4bc5e43f83f26"
POOLING_VERSION = "attention-mask-mean-l2-v1"
LOCAL_RUNTIME = {
    "encoder_device": "cpu", "encoder_dtype": "float32",
    "policy_device": "cpu", "policy_dtype": "float32",
    "translator_device": "cuda", "translator_dtype": "float16",
}
EXPECTED_EMBEDDING_DIMENSION = 384
DATASET_CHECKSUM = "6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce"
SPLIT_CHECKSUM = "aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4"
TRANSLATOR_FINGERPRINT = "a54ba8356642a7a696234453b3fc0a29d2dcf85db5299677c492ae967281bd1c"
V1_SOURCE_COMMIT = "6c75da5d60cc626ab79e7e82cae471e18be27531"
V1_CHECKPOINT_STAGE = "dev-frozen-complete"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
SEED = 20260809
FORBIDDEN_FEATURE_TERMS = ("oracle", "future", "reference", "gold", "alignment")
TEXT_FIELDS = {
    "P0": ("current_source_text",),
    "P1": ("current_source_text", "previous_committed_source_text"),
    "P2": ("current_source_text", "previous_committed_source_text", "previous_committed_target_text"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reject_test_split(split_name: str) -> None:
    if split_name.lower() == "test":
        raise RuntimeError("V2 is a post-hoc DEV extension; TEST access is forbidden")


def current_git_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("V2 metadata requires the current V2 code commit") from error


def v1_identity_document() -> dict[str, str]:
    return {
        "checkpoint_stage": V1_CHECKPOINT_STAGE,
        "v1_source_commit": V1_SOURCE_COMMIT,
        "dataset_checksum": DATASET_CHECKSUM,
        "split_checksum": SPLIT_CHECKSUM,
        "translator_config_fingerprint": TRANSLATOR_FINGERPRINT,
    }


def validate_v1_checkpoint_metadata(document: Mapping[str, Any]) -> None:
    expected = v1_identity_document()
    actual = {
        "checkpoint_stage": document.get("checkpoint_stage"),
        "v1_source_commit": document.get("git_commit"),
        "dataset_checksum": document.get("dataset_manifest_checksum"),
        "split_checksum": document.get("split_checksum"),
        "translator_config_fingerprint": document.get("translator_config_fingerprint"),
    }
    if actual != expected:
        raise RuntimeError(f"immutable V1 checkpoint identity mismatch: expected={expected!r} actual={actual!r}")


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(term in str(key).lower() for term in FORBIDDEN_FEATURE_TERMS)
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_causal_state(state: Mapping[str, Any]) -> None:
    expected = {*TEXT_FIELDS["P2"], "numeric"}
    if set(state) != expected:
        raise RuntimeError(f"V2 causal state fields differ from frozen V1 schema: {sorted(state)}")
    if set(state["numeric"]) != set(NUMERIC_FEATURES):
        raise RuntimeError("V2 numeric fields differ from the frozen V1 feature vector")
    if _contains_forbidden_key(state):
        raise RuntimeError("future/oracle/gold/reference/alignment data are forbidden V2 features")
    if any(not isinstance(state[name], str) for name in TEXT_FIELDS["P2"]):
        raise RuntimeError("V2 causal text fields must be exact strings")
    for name in NUMERIC_FEATURES:
        float(state["numeric"][name])


def validate_v1_supervision(directory: Path, split_name: str, *, require_full: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reject_test_split(split_name)
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid V1 supervision manifest: {manifest_path}") from error
    expected_manifest = {
        "split": split_name,
        "dataset_checksum": DATASET_CHECKSUM,
        "split_checksum": SPLIT_CHECKSUM,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise RuntimeError("V1 supervision manifest identity mismatch")
    translator = manifest.get("translator", {})
    if translator.get("config_fingerprint") != TRANSLATOR_FINGERPRINT:
        raise RuntimeError("V1 supervision translator identity mismatch")
    if require_full and (manifest.get("artifact_status") != "full" or not manifest.get("publishable")):
        raise RuntimeError("V2 final execution requires full frozen V1 supervision")
    paths = sorted(directory.glob("*.jsonl"))
    if {path.stem for path in paths} != set(manifest.get("talk_ids", [])):
        raise RuntimeError("V1 supervision talk files do not match manifest coverage")
    rows: list[dict[str, Any]] = []
    counts = {"LISTEN": 0, "COMMIT": 0}
    for path in paths:
        try:
            talk_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid V1 supervision file: {path}") from error
        if not talk_rows:
            raise RuntimeError(f"empty V1 supervision file: {path}")
        for row in talk_rows:
            if row.get("talk_id") != path.stem or row.get("split") != split_name:
                raise RuntimeError(f"V1 supervision row identity mismatch: {path}")
            label = row.get("label")
            if label not in counts:
                raise RuntimeError(f"invalid V1 LISTEN/COMMIT label: {path}")
            validate_causal_state(row.get("causal", {}))
            counts[label] += 1
        rows.extend(talk_rows)
    if len(rows) != manifest.get("state_count") or any(counts[name] != manifest.get(name) for name in counts):
        raise RuntimeError("V1 supervision state/label counts do not match manifest")
    if sorted({row["talk_id"] for row in rows}) != sorted(manifest.get("expected_talk_ids", [])):
        raise RuntimeError("V1 supervision is not exact full talk coverage")
    return manifest, rows


def state_texts(state: Mapping[str, Any], variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown V2 variant: {variant}")
    validate_causal_state(state)
    return tuple(state[name] for name in TEXT_FIELDS[variant])


def numeric_vector(state: Mapping[str, Any]) -> np.ndarray:
    validate_causal_state(state)
    return np.asarray([float(state["numeric"][name]) for name in NUMERIC_FEATURES], dtype=np.float32)


@dataclass(frozen=True)
class NumericScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_split: str = "train"

    @classmethod
    def fit(cls, values: np.ndarray, *, split_name: str) -> "NumericScaler":
        if split_name != "train":
            raise RuntimeError("numeric scaler may only be fitted on TRAIN")
        if values.ndim != 2 or values.shape[1] != len(NUMERIC_FEATURES) or values.shape[0] == 0:
            raise ValueError("invalid numeric TRAIN matrix")
        mean = values.astype(np.float64).mean(axis=0)
        scale = values.astype(np.float64).std(axis=0)
        scale[scale == 0.0] = 1.0
        return cls(tuple(mean.tolist()), tuple(scale.tolist()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - np.asarray(self.mean)) / np.asarray(self.scale)).astype(np.float32)


class FrozenMiniLMEncoder:
    """Pinned Transformers encoder with mask-aware mean pooling."""

    def __init__(
        self, *, device: str | torch.device = "cpu", dtype: str = "float32", batch_size: int = 256,
        allow_cuda_float32: bool = False,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        expected_dtype = "float16" if self.device.type == "cuda" else "float32"
        if dtype != expected_dtype and not (self.device.type == "cuda" and dtype == "float32" and allow_cuda_float32):
            raise ValueError(f"MiniLM {self.device.type} requires {expected_dtype}, not {dtype}")
        self.dtype = dtype
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID, revision=ENCODER_REVISION)
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        self.model = AutoModel.from_pretrained(ENCODER_MODEL_ID, revision=ENCODER_REVISION, torch_dtype=torch_dtype)
        resolved_revision = getattr(self.model.config, "_commit_hash", None)
        if resolved_revision != ENCODER_REVISION:
            raise RuntimeError(
                f"MiniLM resolved revision mismatch: expected={ENCODER_REVISION} actual={resolved_revision}"
            )
        self.model.eval().requires_grad_(False).to(self.device)
        hidden_size = int(self.model.config.hidden_size)
        if hidden_size != EXPECTED_EMBEDDING_DIMENSION:
            raise RuntimeError(f"unexpected MiniLM hidden dimension: {hidden_size}")
        self.dimension = hidden_size

    @staticmethod
    def pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(pooled.float(), p=2, dim=1)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + self.batch_size]), padding=True, truncation=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                batches.append(self.pool(output.last_hidden_state, encoded["attention_mask"]).cpu().numpy())
        result = np.concatenate(batches).astype(np.float32, copy=False)
        if result.shape[1] != self.dimension:
            raise RuntimeError("MiniLM pooled output dimension changed")
        return result


class EmbeddingCache:
    """Content-addressed float32 cache; deleting it changes no model semantics."""

    def __init__(self, root: Path, encoder: Any) -> None:
        self.root, self.encoder = root, encoder
        self.dimension = int(encoder.dimension)

    @staticmethod
    def key(text: str) -> str:
        identity = {
            "model_id": ENCODER_MODEL_ID, "revision": ENCODER_REVISION,
            "pooling": POOLING_VERSION, "text": text,
        }
        return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _path(self, text: str) -> Path:
        key = self.key(text)
        return self.root / key[:2] / f"{key}.pt"

    def _read(self, text: str) -> np.ndarray | None:
        path = self._path(text)
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            vector = payload["embedding"].numpy().astype(np.float32, copy=False)
        except (OSError, RuntimeError, KeyError, TypeError):
            return None
        if payload.get("key") != self.key(text) or vector.shape != (self.dimension,):
            return None
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        result: list[np.ndarray | None] = []
        missing: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            if text == "":
                result.append(np.zeros(self.dimension, dtype=np.float32))
                continue
            cached = self._read(text)
            result.append(cached)
            if cached is None:
                missing.setdefault(text, []).append(index)
        if missing:
            unique = list(missing)
            encoded = self.encoder.encode(unique)
            if encoded.shape != (len(unique), self.dimension):
                raise RuntimeError("encoder returned an invalid embedding matrix")
            for text, vector in zip(unique, encoded, strict=True):
                vector = np.asarray(vector, dtype=np.float32)
                path = self._path(text)
                _atomic_torch_save(path, {"key": self.key(text), "embedding": torch.from_numpy(vector)})
                for index in missing[text]:
                    result[index] = vector
        if any(vector is None for vector in result):
            raise RuntimeError("embedding cache failed to materialize all requested texts")
        finalized = [cast(np.ndarray, vector) for vector in result]
        return np.stack(finalized).astype(np.float32, copy=False)


class V2MLP(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def input_dimension(variant: str, embedding_dimension: int = EXPECTED_EMBEDDING_DIMENSION) -> int:
    if variant not in VARIANTS:
        raise ValueError(f"unknown V2 variant: {variant}")
    return len(TEXT_FIELDS[variant]) * embedding_dimension + len(NUMERIC_FEATURES)


def class_weight(label_counts: Mapping[str, int]) -> float:
    positive, negative = int(label_counts["COMMIT"]), int(label_counts["LISTEN"])
    if positive <= 0 or negative <= 0:
        raise RuntimeError("weighted BCE requires both TRAIN classes")
    return negative / positive


def _set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_feature_matrix(rows: Sequence[Mapping[str, Any]], variant: str, cache: EmbeddingCache, scaler: NumericScaler) -> np.ndarray:
    text_columns = list(zip(*(state_texts(row["causal"], variant) for row in rows), strict=True))
    embedded = [cache.encode(column) for column in text_columns]
    numeric = scaler.transform(np.stack([numeric_vector(row["causal"]) for row in rows]))
    matrix = np.concatenate([*embedded, numeric], axis=1).astype(np.float32, copy=False)
    if matrix.shape[1] != input_dimension(variant, cache.dimension):
        raise RuntimeError("V2 feature dimension mismatch")
    return matrix


@dataclass
class V2Policy:
    variant: str
    model: V2MLP
    scaler: NumericScaler
    cache: EmbeddingCache
    device: torch.device

    def predict_commit_probability(self, state: Mapping[str, Any]) -> float:
        self.model.eval()
        texts = state_texts(state, self.variant)
        features = np.concatenate([
            *(self.cache.encode([text])[0] for text in texts),
            self.scaler.transform(numeric_vector(state)[None, :])[0],
        ]).astype(np.float32)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)))[0].cpu())


def train_v2_policy(
    rows: Sequence[Mapping[str, Any]], variant: str, cache: EmbeddingCache, *,
    epochs: int = 20, batch_size: int = 256, device: str | torch.device | None = None,
) -> tuple[V2Policy, dict[str, Any]]:
    if not rows or any(row.get("split") != "train" for row in rows):
        raise RuntimeError("V2 training accepts TRAIN supervision only")
    _set_deterministic_seed()
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    numeric = np.stack([numeric_vector(row["causal"]) for row in rows])
    scaler = NumericScaler.fit(numeric, split_name="train")
    features = build_feature_matrix(rows, variant, cache, scaler)
    labels = np.asarray([1.0 if row["label"] == "COMMIT" else 0.0 for row in rows], dtype=np.float32)
    counts = {"LISTEN": int((labels == 0).sum()), "COMMIT": int((labels == 1).sum())}
    weight = class_weight(counts)
    model = V2MLP(features.shape[1]).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=target_device))
    generator = torch.Generator().manual_seed(SEED)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, seen = 0.0, 0
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(target_device), batch_labels.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_labels)
            seen += len(batch_labels)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(features).to(target_device))
            predicted = (logits >= 0).cpu().numpy()
        true_positive = int(((predicted == 1) & (labels == 1)).sum())
        recall_positive = true_positive / counts["COMMIT"]
        recall_negative = int(((predicted == 0) & (labels == 0)).sum()) / counts["LISTEN"]
        history.append({"epoch": epoch + 1, "train_loss": total_loss / seen, "train_balanced_accuracy": (recall_positive + recall_negative) / 2})
    policy = V2Policy(variant, model, scaler, cache, target_device)
    return policy, {"label_counts": counts, "positive_weight": weight, "training_history": history}


def checkpoint_payload(policy: V2Policy) -> dict[str, Any]:
    return {
        "format_version": "1.0.0", "variant": policy.variant,
        "input_dimension": input_dimension(policy.variant, policy.cache.dimension),
        "embedding_dimension": policy.cache.dimension,
        "scaler": asdict(policy.scaler),
        "model_state_dict": {name: value.detach().cpu().float() for name, value in policy.model.state_dict().items()},
    }


def save_v2_checkpoint(path: Path, policy: V2Policy) -> str:
    _atomic_torch_save(path, checkpoint_payload(policy))
    return sha256_file(path)


def load_v2_checkpoint(path: Path, metadata_path: Path, cache: EmbeddingCache, *, device: str | torch.device | None = None) -> V2Policy:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("checkpoint_sha256") != sha256_file(path):
        raise RuntimeError("V2 checkpoint SHA-256 mismatch")
    if metadata.get("experiment_status") != EXPERIMENT_STATUS or metadata.get("encoder_revision") != ENCODER_REVISION:
        raise RuntimeError("V2 checkpoint exploratory/encoder identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    variant = payload["variant"]
    if variant != metadata.get("variant") or payload["embedding_dimension"] != cache.dimension:
        raise RuntimeError("V2 checkpoint variant/embedding identity mismatch")
    scaler_value = payload["scaler"]
    scaler = NumericScaler(tuple(scaler_value["mean"]), tuple(scaler_value["scale"]), scaler_value["fitted_split"])
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = V2MLP(payload["input_dimension"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(target_device)
    return V2Policy(variant, model, scaler, cache, target_device)


def make_checkpoint_metadata(
    *, variant: str, manifest: Mapping[str, Any], training: Mapping[str, Any],
    checkpoint_hash: str, v2_code_commit: str, embedding_dimension: int,
) -> dict[str, Any]:
    fields = {name: list(TEXT_FIELDS[name]) for name in VARIANTS}
    numeric_identity = {"ordered_features": list(NUMERIC_FEATURES), "fingerprint": stable_fingerprint(list(NUMERIC_FEATURES))}
    pseudo_identity = {"config": manifest["config"], "fingerprint": manifest["config_checksum"]}
    return {
        "artifact_status": "full", "publishable": False, "experiment_status": EXPERIMENT_STATUS,
        "experiment_label": EXPERIMENT_LABEL, "variant": variant, "model_family": "frozen MiniLM embeddings + MLP",
        "encoder_model_id": ENCODER_MODEL_ID, "encoder_revision": ENCODER_REVISION,
        "pooling_version": POOLING_VERSION, "embedding_dimension": embedding_dimension,
        "input_field_definitions": fields, "numeric_feature_config": numeric_identity,
        "pseudo_label_config": pseudo_identity, "dataset_checksum": DATASET_CHECKSUM,
        "split_checksum": SPLIT_CHECKSUM, "translator_fingerprint": TRANSLATOR_FINGERPRINT,
        "train_talk_ids": manifest["talk_ids"], "label_counts": training["label_counts"], "seed": SEED,
        "mlp_architecture": ["Linear(input,256)", "GELU", "Dropout(0.20)", "Linear(256,64)", "GELU", "Dropout(0.10)", "Linear(64,1)"],
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4},
        "batch_size": 256, "epochs": 20,
        "weighted_bce": {"formula": "pos_weight = TRAIN_LISTEN / TRAIN_COMMIT", "positive_weight": training["positive_weight"]},
        "numeric_scaler_fit_split": "train", "checkpoint_sha256": checkpoint_hash,
        "v1_source_commit": V1_SOURCE_COMMIT, "v2_code_commit": v2_code_commit,
        "runtime": dict(LOCAL_RUNTIME),
        "training_history": training["training_history"],
    }


def validate_prediction_record(
    record: Mapping[str, Any], *, strategy: str, talk_id: str, model_hash: str, split_name: str = "dev",
) -> None:
    expected = {
        "strategy": strategy, "talk_id": talk_id, "split": split_name, "artifact_status": "full",
        "experiment_status": EXPERIMENT_STATUS, "model_sha256": model_hash,
        "dataset_checksum": DATASET_CHECKSUM, "encoder_revision": ENCODER_REVISION,
        "runtime": LOCAL_RUNTIME,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"invalid resumable V2 {split_name.upper()} prediction identity")
    if not isinstance(record.get("commits"), list) or not record["commits"]:
        raise RuntimeError(f"invalid resumable V2 {split_name.upper()} prediction commits")


def metrics_are_complete(path: Path, strategies: Sequence[str]) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return set(document) == set(strategies) and all(
        row.get("artifact_status") == "full"
        and row.get("experiment_status") == EXPERIMENT_STATUS
        and row.get("dataset_checksum") == DATASET_CHECKSUM
        and row.get("encoder_revision") == ENCODER_REVISION
        and row.get("runtime") == LOCAL_RUNTIME
        and isinstance(row.get("model_sha256"), str)
        for row in document.values()
    )


def select_v2_configuration(v1_metrics: Mapping[str, Mapping[str, float]], v2_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    adapted: dict[str, Mapping[str, float]] = {"fixed_n_8": v1_metrics["fixed_n_8"]}
    name_map = {}
    for name, row in v2_metrics.items():
        if not name.startswith("v2_P"):
            raise ValueError(f"invalid V2 strategy name: {name}")
        adapted_name = "learned_" + name.removeprefix("v2_")
        adapted[adapted_name] = row
        name_map[adapted_name] = name
    result = select_dev_configuration(adapted)
    selected = name_map[result["selected_strategy"]]
    result.update({
        "selected_strategy": selected, "selected_variant": selected.split("_")[1],
        "experiment_status": EXPERIMENT_STATUS, "experiment_label": EXPERIMENT_LABEL,
        "selection_rule_identity": "V1 select_dev_configuration (unchanged adapter)",
        "selected_dev_metrics": dict(v2_metrics[selected]),
    })
    return result


_RESTORE_FILES = {
    "checkpoint-metadata.json",
    "data/policy/pseudo_labels/train/manifest.json",
    "data/policy/pseudo_labels/dev/manifest.json",
    "outputs/experiments/research-mvp/metrics/dev/all.json",
    "outputs/experiments/research-mvp/dev-selection.json",
    "outputs/experiments/research-mvp/frozen-eval-config.json",
}
_RESTORE_PREFIXES = (
    "data/policy/pseudo_labels/train/", "data/policy/pseudo_labels/dev/",
)


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/").removeprefix("./")
    if normalized.startswith("timelymt-checkpoint/"):
        normalized = normalized.removeprefix("timelymt-checkpoint/")
    elif normalized.rstrip("/") == "timelymt-checkpoint":
        return "."
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe V1 checkpoint archive path: {name!r}")
    return normalized


def _required_restore_path(name: str) -> bool:
    return name in _RESTORE_FILES or (name.endswith(".jsonl") and name.startswith(_RESTORE_PREFIXES))


def _copy_without_overwrite(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"refusing to overwrite differing immutable V1 artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=destination.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        with source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
    temporary.replace(destination)


def restore_v1_artifacts(source: Path, root: Path) -> dict[str, Any]:
    """Restore only immutable supervision and comparison inputs, never V1 source."""
    staging_parent = Path(tempfile.mkdtemp(prefix="timelymt-v1-restore-"))
    try:
        staging = staging_parent / "timelymt-checkpoint"
        if source.is_file():
            staging.mkdir()
            with tarfile.open(source, "r:gz") as archive:
                selected = []
                for member in archive.getmembers():
                    name = _safe_member_name(member.name)
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise RuntimeError(f"unsafe V1 checkpoint archive member: {member.name!r}")
                    if member.isfile() and _required_restore_path(name):
                        member.name = name
                        selected.append(member)
                archive.extractall(staging, members=selected, filter="data")
        elif source.is_dir():
            candidate = source / "timelymt-checkpoint" if (source / "timelymt-checkpoint").is_dir() else source
            for path in candidate.rglob("*"):
                relative = path.relative_to(candidate).as_posix()
                _safe_member_name(relative)
                if path.is_symlink():
                    raise RuntimeError(f"symlink forbidden in expanded V1 checkpoint: {path}")
                if path.is_file() and _required_restore_path(relative):
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
        else:
            raise RuntimeError(f"V1 checkpoint source does not exist: {source}")
        metadata_path = staging / "checkpoint-metadata.json"
        validate_v1_checkpoint_metadata(json.loads(metadata_path.read_text(encoding="utf-8")))
        validate_v1_supervision(staging / "data/policy/pseudo_labels/train", "train")
        validate_v1_supervision(staging / "data/policy/pseudo_labels/dev", "dev")
        destinations: dict[Path, Path] = {}
        for path in staging.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(staging)
            if str(relative).replace("\\", "/").startswith("data/policy/pseudo_labels/"):
                destination = root / relative
            else:
                destination = root / "outputs/experiments/policy-v2/v1-source" / relative.name
            destinations[path] = destination
        for path, destination in destinations.items():
            _copy_without_overwrite(path, destination)
        identity = {**v1_identity_document(), "v2_code_commit": current_git_commit(root)}
        atomic_json(root / "outputs/experiments/policy-v2/v1-source/identity.json", identity)
        return identity
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
