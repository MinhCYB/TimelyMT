"""Resumable, stage-gated command line runner for the research MVP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from timelymt.data.canonical.core import load_canonical_talk
from timelymt.data.translation_artifacts import (
    RuntimeTalk, make_translation_request, runtime_talk_from_canonical, stable_fingerprint,
    translate_requests, translator_identity,
)
from timelymt.translator.cache import TranslationCache
from timelymt.translator.envit5 import EnViT5Translator, load_config
from .evaluation import latency_metrics, quality_metrics
from .meaningful_units import (
    MU_NUMERIC_FEATURES, MU_TEXT_FEATURES, MeaningfulUnitPolicy, flatten_mu_state,
    generate_mu_supervision, mu_config_document, mu_rollout, train_mu_policy,
)
from .policy import LearnedPolicy, NUMERIC_FEATURES, VARIANTS, flatten_state, train_policy
from .pseudo_labels import config_document, generate_pseudo_labels
from .streaming import (
    fixed_n, fixed_time, learned_rollout, local_agreement_la2, local_agreement_style, prediction_record,
    select_dev_configuration,
)


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "outputs/experiments/research-mvp"
PSEUDO = ROOT / "data/policy/pseudo_labels"
MU_SUPERVISION = ROOT / "data/policy/mu_zhang2020"
CHECKPOINTS = ROOT / "checkpoints/policy"
DATASET_CHECKSUM = "6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce"
SPLIT_CHECKSUM = "aabc06af1836e5d66a69d3b0305f6044892cbe0d3e45883ee7aeed53edd3ddc4"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
BASELINES = {
    "fixed_n_4": ("fixed_n", 4), "fixed_n_8": ("fixed_n", 8), "fixed_n_12": ("fixed_n", 12),
    "fixed_time_1600": ("fixed_time", 1600), "fixed_time_3200": ("fixed_time", 3200),
    "fixed_time_4800": ("fixed_time", 4800), "local_agreement_style_k2": ("local", 2),
    "local_agreement_style_k3": ("local", 3), "local_agreement_la2": ("la2", 2),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    split = _load_json(ROOT / "data/splits/experimental.json")
    dataset = _load_json(ROOT / "data/manifests/timelymt-streaming-dataset-v1.json")
    if dataset["manifest_checksum"] != DATASET_CHECKSUM or stable_fingerprint(split) != SPLIT_CHECKSUM:
        raise RuntimeError("frozen Dataset v1 or split checksum changed")
    return dataset, split


def _talk_paths() -> dict[str, Path]:
    manifest = _load_json(ROOT / "data/manifests/streaming-dataset.json")
    return {row["talk_id"]: ROOT / row["canonical_path"] for row in manifest["talks"]}


def _runtime_talk(talk_id: str, split: Mapping[str, Any], max_source_tokens: int | None = None) -> RuntimeTalk:
    document = load_canonical_talk(_talk_paths()[talk_id])
    count = len(document["stream"]["tokens"])
    if max_source_tokens is not None:
        count = min(count, max_source_tokens)
    return runtime_talk_from_canonical(document, split_manifest=split, observed_through_token_index=count - 1)


def _selected_talks(split: Mapping[str, Any], split_name: str, talk_id: str | None, max_talks: int | None) -> list[str]:
    expected = list(split["splits"][split_name])
    if talk_id is not None:
        if talk_id not in expected:
            raise ValueError(f"talk {talk_id} does not belong to {split_name}")
        expected = [talk_id]
    if max_talks is not None:
        expected = expected[:max_talks]
    return expected


class Provider:
    def __init__(self, translator: EnViT5Translator, identity: Any, batch_size: int) -> None:
        self.translator, self.identity, self.batch_size = translator, identity, batch_size
        self.calls = self.hits = 0

    def __call__(self, talk: RuntimeTalk, start: int, end: int):
        return self.batch(talk, start, [end])[0]

    def batch(self, talk: RuntimeTalk, start: int, ends: Sequence[int]):
        requests = []
        for end in ends:
            observed = RuntimeTalk(talk.talk_id, talk.split, talk.tokens[: end + 1])
            requests.append(make_translation_request(observed, start, end, translator=self.identity))
        hypotheses = translate_requests(
            self.translator, requests, translator_identity=self.identity,
            batch_size=min(self.batch_size, len(requests)),
        )
        self.calls += len(hypotheses)
        self.hits += sum(hypothesis.cache_hit is True for hypothesis in hypotheses)
        return hypotheses


def _translator(batch_size: int) -> tuple[Any, Provider]:
    config = load_config(ROOT / "configs/translator/envit5.json")
    identity = translator_identity(config)
    translator = EnViT5Translator(config, cache=TranslationCache(ROOT / "outputs/translator/cache"))
    return identity, Provider(translator, identity, batch_size)


def _pseudo_dir(split_name: str, smoke: bool) -> Path:
    return PSEUDO / "smoke" / split_name if smoke else PSEUDO / split_name


def _mu_dir(split_name: str, smoke: bool) -> Path:
    return MU_SUPERVISION / "smoke" / split_name if smoke else MU_SUPERVISION / split_name


def _read_jsonl_directory(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return rows


def _validate_supervision_talk_file(
    path: Path, talk_id: str, split_name: str, oracle_field: str,
) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid resumable pseudo-label file {path}: {error}") from error
    if not rows or any(row.get("talk_id") != talk_id or row.get("split") != split_name for row in rows):
        raise RuntimeError(f"pseudo-label file has wrong or empty talk/split identity: {path}")
    for row in rows:
        if row.get("label") not in {"LISTEN", "COMMIT"} or "causal" not in row or oracle_field not in row:
            raise RuntimeError(f"pseudo-label row has invalid supervision fields: {path}")
        pending = [row["causal"]]
        while pending:
            value = pending.pop()
            if isinstance(value, Mapping):
                if any(term in str(key).lower() for key in value for term in ("oracle", "future", "reference", "gold")):
                    raise RuntimeError(f"oracle/reference data leaked into causal features: {path}")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    return rows


def _validate_pseudo_talk_file(path: Path, talk_id: str, split_name: str) -> list[dict[str, Any]]:
    rows = _validate_supervision_talk_file(path, talk_id, split_name, "oracle_training_only")
    if any("mu_oracle_training_only" in row for row in rows):
        raise RuntimeError(f"TimelyMT pseudo-label file contains MU supervision: {path}")
    return rows


def _validate_mu_talk_file(path: Path, talk_id: str, split_name: str) -> list[dict[str, Any]]:
    rows = _validate_supervision_talk_file(path, talk_id, split_name, "mu_oracle_training_only")
    for row in rows:
        if "mu_oracle_training_only" not in row or "oracle_training_only" in row:
            raise RuntimeError(f"MU row has invalid or TimelyMT pseudo-label supervision fields: {path}")
        if set(row["causal"]) != {*MU_TEXT_FEATURES, "numeric"}:
            raise RuntimeError(f"MU causal text features do not match the frozen baseline: {path}")
        if set(row["causal"]["numeric"]) != set(MU_NUMERIC_FEATURES):
            raise RuntimeError(f"MU causal numeric features do not match the frozen baseline: {path}")
    return rows


def build_pseudo_manifest(
    split_name: str, directory: Path, expected_talks: Sequence[str], identity: Any,
    *, smoke: bool, limited: bool,
) -> dict[str, Any]:
    rows = _read_jsonl_directory(directory)
    talks = sorted({row["talk_id"] for row in rows})
    complete = set(talks) == set(expected_talks) and not limited
    status = "smoke" if smoke else "full" if complete else "partial"
    return {
        "artifact_status": status, "publishable": status == "full", "split": split_name,
        "config": config_document(), "config_checksum": stable_fingerprint(config_document()),
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator": asdict(identity), "state_count": len(rows),
        "LISTEN": sum(row["label"] == "LISTEN" for row in rows),
        "COMMIT": sum(row["label"] == "COMMIT" for row in rows),
        "talk_ids": talks, "expected_talk_ids": list(expected_talks),
        "training_only_note": "Future hypotheses construct train/dev supervision only and are unavailable to runtime policy.",
    }


def pseudo(
    split_name: str, batch_size: int, *, talk_id: str | None = None,
    max_talks: int | None = None, max_states: int | None = None, smoke: bool = False,
) -> None:
    if split_name not in {"train", "dev"}:
        raise ValueError("pseudo labels may only be generated for train or dev")
    limited = talk_id is not None or max_talks is not None or max_states is not None
    if limited and not smoke:
        raise ValueError("limited pseudo generation requires --smoke and writes to a separate non-experimental path")
    _, split = _manifests()
    identity, provider = _translator(batch_size)
    expected_all = list(split["splits"][split_name])
    selected = _selected_talks(split, split_name, talk_id, max_talks)
    output_dir = _pseudo_dir(split_name, smoke)
    started = time.perf_counter()
    for index, selected_id in enumerate(selected, start=1):
        path = output_dir / f"{selected_id}.jsonl"
        if path.exists():
            _validate_pseudo_talk_file(path, selected_id, split_name)
            print(f"talk {index}/{len(selected)} {selected_id}: resume hit output={path}")
            continue
        runtime_talk = _runtime_talk(selected_id, split)
        rows = generate_pseudo_labels(runtime_talk, provider, max_states=max_states)
        _atomic_jsonl(path, rows)
        misses = provider.calls - provider.hits
        print(f"talk {index}/{len(selected)} {selected_id}: states={len(rows)} cache_hits={provider.hits} cache_misses={misses} elapsed={time.perf_counter()-started:.1f}s output={path}")
    manifest = build_pseudo_manifest(split_name, output_dir, expected_all, identity, smoke=smoke, limited=limited)
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"artifact_status={manifest['artifact_status']} states={manifest['state_count']} output={output_dir / 'manifest.json'}")


def validate_pseudo(split_name: str, *, smoke: bool = False) -> None:
    if split_name not in {"train", "dev"}:
        raise ValueError("pseudo validation is only defined for train/dev")
    _, split = _manifests()
    identity, _ = _translator(1)
    directory = _pseudo_dir(split_name, smoke)
    rows = _read_jsonl_directory(directory)
    expected = list(split["splits"][split_name])
    actual = {row["talk_id"] for row in rows}
    if any(row["split"] != split_name for row in rows):
        raise RuntimeError("pseudo-label split contamination detected")
    limited = actual != set(expected)
    manifest = build_pseudo_manifest(split_name, directory, expected, identity, smoke=smoke, limited=limited)
    _atomic_json(directory / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("artifact_status", "state_count", "LISTEN", "COMMIT", "talk_ids", "expected_talk_ids")}, indent=2))


def build_mu_manifest(
    split_name: str, directory: Path, expected_talks: Sequence[str], identity: Any,
    *, smoke: bool, limited: bool,
) -> dict[str, Any]:
    rows = _read_jsonl_directory(directory)
    talks = sorted({row["talk_id"] for row in rows})
    complete = set(talks) == set(expected_talks) and not limited
    status = "smoke" if smoke else "full" if complete else "partial"
    return {
        "artifact_type": "mu_zhang2020_supervision",
        "artifact_status": status, "publishable": status == "full", "split": split_name,
        "config": mu_config_document(), "config_checksum": stable_fingerprint(mu_config_document()),
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator": asdict(identity), "state_count": len(rows),
        "LISTEN": sum(row["label"] == "LISTEN" for row in rows),
        "COMMIT": sum(row["label"] == "COMMIT" for row in rows),
        "talk_ids": talks, "expected_talk_ids": list(expected_talks),
        "training_only_note": "Full admissible remaining-unit translation (at most 48 lexical source tokens) constructs MU train/dev supervision only; MU runtime remains causal.",
    }


def mu_supervision(
    split_name: str, batch_size: int, *, talk_id: str | None = None,
    max_talks: int | None = None, max_states: int | None = None, smoke: bool = False,
) -> None:
    if split_name not in {"train", "dev"}:
        raise ValueError("MU supervision may only be generated for train or dev")
    limited = talk_id is not None or max_talks is not None or max_states is not None
    if limited and not smoke:
        raise ValueError("limited MU supervision requires --smoke and a non-experimental path")
    _, split = _manifests()
    identity, provider = _translator(batch_size)
    expected_all = list(split["splits"][split_name])
    selected = _selected_talks(split, split_name, talk_id, max_talks)
    output_dir = _mu_dir(split_name, smoke)
    for index, selected_id in enumerate(selected, start=1):
        path = output_dir / f"{selected_id}.jsonl"
        if path.exists():
            _validate_mu_talk_file(path, selected_id, split_name)
            print(f"MU talk {index}/{len(selected)} {selected_id}: resume hit output={path}")
            continue
        rows = generate_mu_supervision(_runtime_talk(selected_id, split), provider, max_states=max_states)
        _atomic_jsonl(path, rows)
        print(f"MU talk {index}/{len(selected)} {selected_id}: states={len(rows)} cache_hits={provider.hits} cache_misses={provider.calls-provider.hits} output={path}")
    manifest = build_mu_manifest(split_name, output_dir, expected_all, identity, smoke=smoke, limited=limited)
    _atomic_json(output_dir / "manifest.json", manifest)
    print(f"SMOKE / NON-EXPERIMENTAL" if smoke else f"artifact_status={manifest['artifact_status']}")
    print(f"states={manifest['state_count']} output={output_dir / 'manifest.json'}")


def validate_mu(split_name: str, *, smoke: bool = False) -> None:
    if split_name not in {"train", "dev"}:
        raise ValueError("MU validation is only defined for train/dev")
    _, split = _manifests()
    identity, _ = _translator(1)
    directory = _mu_dir(split_name, smoke)
    expected = list(split["splits"][split_name])
    for path in sorted(directory.glob("*.jsonl")):
        _validate_mu_talk_file(path, path.stem, split_name)
    rows = _read_jsonl_directory(directory)
    limited = {row["talk_id"] for row in rows} != set(expected)
    manifest = build_mu_manifest(split_name, directory, expected, identity, smoke=smoke, limited=limited)
    _atomic_json(directory / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("artifact_status", "state_count", "LISTEN", "COMMIT", "talk_ids")}, indent=2))


def train(manifest_path: Path, variant: str, *, allow_smoke: bool = False) -> None:
    import joblib
    import sklearn

    _manifests()
    manifest = _load_json(manifest_path)
    if manifest.get("split") != "train" or manifest.get("dataset_checksum") != DATASET_CHECKSUM or manifest.get("split_checksum") != SPLIT_CHECKSUM:
        raise RuntimeError("training requires checksum-valid TRAIN pseudo labels")
    status = manifest.get("artifact_status")
    if status != "full" and not allow_smoke:
        raise RuntimeError("final training refuses smoke/partial pseudo labels; use --allow-smoke only for integration testing")
    if status == "partial":
        raise RuntimeError("partial pseudo-label artifacts are never trainable; generate a bounded --smoke artifact instead")
    rows = _read_jsonl_directory(manifest_path.parent)
    if not rows or any(row["split"] != "train" for row in rows):
        raise RuntimeError("training rows are empty or split-contaminated")
    for talk_id in {row["talk_id"] for row in rows}:
        _validate_pseudo_talk_file(manifest_path.parent / f"{talk_id}.jsonl", talk_id, "train")
    started = time.perf_counter()
    policy = train_policy(rows, variant)
    output_dir = CHECKPOINTS / "smoke" if status == "smoke" else CHECKPOINTS
    path = output_dir / f"{variant}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, path)
    reloaded = joblib.load(path)
    probability = reloaded.predict_commit_probability(rows[0]["causal"])
    feature_dimension = int(policy.pipeline.named_steps["features"].transform([flatten_state(rows[0]["causal"], variant)]).shape[1])
    metadata = {
        "artifact_status": "smoke" if status == "smoke" else "full", "publishable": status == "full",
        "variant": variant, "training_states": len(rows),
        "LISTEN": sum(row["label"] == "LISTEN" for row in rows),
        "COMMIT": sum(row["label"] == "COMMIT" for row in rows),
        "talk_coverage": len({row["talk_id"] for row in rows}), "feature_dimension": feature_dimension,
        "training_seconds": time.perf_counter() - started, "checkpoint": path.as_posix(),
        "checkpoint_sha256": _sha256(path), "sklearn_version": sklearn.__version__,
        "class_weight": "balanced", "random_state": 20260809,
        "pseudo_manifest": manifest_path.as_posix(), "reload_predict_proba_smoke": probability,
    }
    _atomic_json(output_dir / f"{variant}.metadata.json", metadata)
    print(f"trained {variant}: status={metadata['artifact_status']} states={len(rows)} dim={feature_dimension} output={path}")


def train_mu(manifest_path: Path, *, allow_smoke: bool = False) -> None:
    import joblib
    import sklearn

    _manifests()
    manifest = _load_json(manifest_path)
    if manifest.get("artifact_type") != "mu_zhang2020_supervision" or manifest.get("split") != "train":
        raise RuntimeError("MU training requires MU TRAIN supervision")
    if manifest.get("dataset_checksum") != DATASET_CHECKSUM or manifest.get("split_checksum") != SPLIT_CHECKSUM:
        raise RuntimeError("MU training requires checksum-valid frozen data")
    status = manifest.get("artifact_status")
    if status != "full" and not allow_smoke:
        raise RuntimeError("final MU training refuses smoke/partial supervision")
    if status == "partial":
        raise RuntimeError("partial MU supervision is never trainable")
    rows = _read_jsonl_directory(manifest_path.parent)
    if not rows or any(row["split"] != "train" for row in rows):
        raise RuntimeError("MU training rows are empty or split-contaminated")
    for talk_id in {row["talk_id"] for row in rows}:
        _validate_mu_talk_file(manifest_path.parent / f"{talk_id}.jsonl", talk_id, "train")
    policy = train_mu_policy(rows)
    output_dir = CHECKPOINTS / "smoke" if status == "smoke" else CHECKPOINTS
    path = output_dir / "mu_zhang2020.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(policy, path)
    restored = joblib.load(path)
    probability = restored.predict_commit_probability(rows[0]["causal"])
    feature_dimension = int(policy.pipeline.named_steps["features"].transform([flatten_mu_state(rows[0]["causal"])]).shape[1])
    metadata = {
        "artifact_status": "smoke" if status == "smoke" else "full", "publishable": status == "full",
        "strategy": "mu_zhang2020", "training_states": len(rows),
        "LISTEN": sum(row["label"] == "LISTEN" for row in rows),
        "COMMIT": sum(row["label"] == "COMMIT" for row in rows),
        "feature_dimension": feature_dimension, "features": {"numeric": MU_NUMERIC_FEATURES, "text": MU_TEXT_FEATURES},
        "checkpoint": path.as_posix(), "checkpoint_sha256": _sha256(path),
        "sklearn_version": sklearn.__version__, "class_weight": "balanced", "random_state": 20260809,
        "supervision_manifest": manifest_path.as_posix(), "reload_predict_proba_smoke": probability,
    }
    _atomic_json(output_dir / "mu_zhang2020.metadata.json", metadata)
    print(f"trained MU: status={metadata['artifact_status']} states={len(rows)} dim={feature_dimension} output={path}")
    if status == "smoke":
        print("SMOKE / NON-EXPERIMENTAL")


def _prediction_root(smoke: bool) -> Path:
    return EXPERIMENT / "smoke" / "predictions" if smoke else EXPERIMENT / "predictions"


def _prediction_path(split_name: str, strategy: str, talk_id: str, smoke: bool = False) -> Path:
    return _prediction_root(smoke) / split_name / strategy / f"{talk_id}.json"


def rollout(
    split_name: str, strategies: Sequence[str], batch_size: int, *, talk_id: str | None = None,
    max_talks: int | None = None, max_source_tokens: int | None = None, smoke: bool = False,
) -> None:
    import joblib

    limited = talk_id is not None or max_talks is not None or max_source_tokens is not None
    if limited and not smoke:
        raise ValueError("limited rollout requires --smoke and writes to a separate non-experimental path")
    if split_name == "test" and smoke:
        raise RuntimeError("TEST smoke rollouts are forbidden")
    if split_name == "test" and not (EXPERIMENT / "frozen-eval-config.json").exists():
        raise RuntimeError("TEST predictions require frozen-eval-config.json")
    _, split = _manifests()
    selected = _selected_talks(split, split_name, talk_id, max_talks)
    _, provider = _translator(batch_size)
    learned: dict[str, LearnedPolicy] = {}
    mu_policy: MeaningfulUnitPolicy | None = None
    for strategy in strategies:
        if strategy.startswith("learned_"):
            variant = strategy.split("_")[1]
            checkpoint_dir = CHECKPOINTS / "smoke" if smoke else CHECKPOINTS
            metadata = _load_json(checkpoint_dir / f"{variant}.metadata.json")
            if smoke != (metadata["artifact_status"] == "smoke"):
                raise RuntimeError("checkpoint status does not match rollout status")
            learned[variant] = joblib.load(checkpoint_dir / f"{variant}.joblib")
        elif strategy == "mu_zhang2020":
            checkpoint_dir = CHECKPOINTS / "smoke" if smoke else CHECKPOINTS
            metadata = _load_json(checkpoint_dir / "mu_zhang2020.metadata.json")
            if smoke != (metadata["artifact_status"] == "smoke"):
                raise RuntimeError("MU checkpoint status does not match rollout status")
            mu_policy = joblib.load(checkpoint_dir / "mu_zhang2020.joblib")
    for strategy in strategies:
        for index, selected_id in enumerate(selected, start=1):
            path = _prediction_path(split_name, strategy, selected_id, smoke)
            if path.exists():
                print(f"{strategy} talk {index}/{len(selected)}: resume hit output={path}")
                continue
            runtime_talk = _runtime_talk(selected_id, split, max_source_tokens)
            if strategy in BASELINES:
                kind, parameter = BASELINES[strategy]
                commits = fixed_n(runtime_talk, provider, parameter) if kind == "fixed_n" else fixed_time(runtime_talk, provider, parameter) if kind == "fixed_time" else local_agreement_la2(runtime_talk, provider) if kind == "la2" else local_agreement_style(runtime_talk, provider, parameter)
            elif strategy == "mu_zhang2020":
                if mu_policy is None:
                    raise RuntimeError("MU policy was not loaded")
                commits = mu_rollout(runtime_talk, provider, mu_policy)
            else:
                _, variant, threshold_text = strategy.split("_")
                threshold = float(threshold_text)
                if threshold not in THRESHOLDS:
                    raise ValueError("learned rollout threshold is not preregistered")
                commits = learned_rollout(runtime_talk, provider, learned[variant], threshold)
            record = prediction_record(strategy, runtime_talk, commits)
            record.update({"artifact_status": "smoke" if smoke else "full", "publishable": not smoke})
            _atomic_json(path, record)
            print(f"{strategy} talk {index}/{len(selected)}: commits={len(commits)} cache_hits={provider.hits} cache_misses={provider.calls-provider.hits} output={path}")


def rollout_selected(split_name: str, batch_size: int) -> None:
    if split_name != "test":
        raise ValueError("rollout-selected is reserved for the frozen TEST stage")
    selection = _load_json(EXPERIMENT / "dev-selection.json")
    rollout(split_name, [selection["selected_strategy"]], batch_size)


def evaluate(split_name: str, strategies: Sequence[str], *, smoke: bool = False, include_selected: bool = False) -> None:
    if split_name == "test" and smoke:
        raise RuntimeError("TEST smoke evaluation is forbidden")
    if split_name == "test" and not (EXPERIMENT / "frozen-eval-config.json").exists():
        raise RuntimeError("TEST evaluation requires frozen-eval-config.json")
    if include_selected:
        selected = _load_json(EXPERIMENT / "dev-selection.json")["selected_strategy"]
        strategies = [*strategies, selected]
    split = _load_json(ROOT / "data/splits/experimental.json")
    root = _prediction_root(smoke)
    all_metrics = {}
    for strategy in strategies:
        paths = sorted((root / split_name / strategy).glob("*.json"))
        if not paths:
            raise RuntimeError(f"no prediction artifacts for {strategy}")
        records = [_load_json(path) for path in paths]
        if not smoke and {row["talk_id"] for row in records} != set(split["splits"][split_name]):
            raise RuntimeError("final evaluation requires complete split prediction coverage")
        references = {}
        for record in records:
            document = load_canonical_talk(_talk_paths()[record["talk_id"]])
            references[record["talk_id"]] = " ".join(segment["text"] for segment in document["target_reference"]["segments"])
        metrics = {**quality_metrics(records, references), **latency_metrics(records, references), "artifact_status": "smoke" if smoke else "full", "publishable": not smoke}
        all_metrics[strategy] = metrics
        output = (EXPERIMENT / "smoke" if smoke else EXPERIMENT) / "metrics" / split_name / f"{strategy}.json"
        _atomic_json(output, metrics)
        print(f"{strategy}: status={metrics['artifact_status']} BLEU={metrics['BLEU']:.3f} chrF2={metrics['chrF2']:.3f} AL={metrics['token_level_average_lagging']:.3f} LAAL={metrics['token_level_length_adaptive_average_lagging']:.3f}")
    output = (EXPERIMENT / "smoke" if smoke else EXPERIMENT) / "metrics" / split_name / "all.json"
    _atomic_json(output, all_metrics)


def select() -> None:
    metrics = _load_json(EXPERIMENT / "metrics/dev/all.json")
    if any(row.get("artifact_status") != "full" for row in metrics.values()):
        raise RuntimeError("DEV selection refuses smoke or partial metrics")
    result = select_dev_configuration(metrics)
    _atomic_json(EXPERIMENT / "dev-selection.json", result)
    print(json.dumps(result, indent=2))


def freeze() -> None:
    identity, _ = _translator(1)
    train_manifest = _load_json(PSEUDO / "train/manifest.json")
    dev_manifest = _load_json(PSEUDO / "dev/manifest.json")
    if train_manifest.get("artifact_status") != "full" or dev_manifest.get("artifact_status") != "full":
        raise RuntimeError("freeze requires full TRAIN and DEV pseudo-label manifests")
    mu_train_manifest = _load_json(MU_SUPERVISION / "train/manifest.json")
    mu_dev_manifest = _load_json(MU_SUPERVISION / "dev/manifest.json")
    if mu_train_manifest.get("artifact_status") != "full" or mu_dev_manifest.get("artifact_status") != "full":
        raise RuntimeError("freeze requires full TRAIN and DEV MU supervision manifests")
    selection = _load_json(EXPERIMENT / "dev-selection.json")
    checkpoints = {}
    for variant in VARIANTS:
        metadata = _load_json(CHECKPOINTS / f"{variant}.metadata.json")
        if metadata.get("artifact_status") != "full":
            raise RuntimeError("freeze refuses smoke policy checkpoints")
        checkpoints[variant] = _sha256(CHECKPOINTS / f"{variant}.joblib")
    mu_metadata = _load_json(CHECKPOINTS / "mu_zhang2020.metadata.json")
    if mu_metadata.get("artifact_status") != "full":
        raise RuntimeError("freeze refuses smoke MU checkpoint")
    checkpoints["mu_zhang2020"] = _sha256(CHECKPOINTS / "mu_zhang2020.joblib")
    document = {
        "experiment_run_version": "1.0.0", "dataset_checksum": DATASET_CHECKSUM,
        "split_checksum": SPLIT_CHECKSUM, "translator": asdict(identity),
        "pseudo_label_config": config_document(),
        "mu_supervision_config": mu_config_document(),
        "mu_feature_config": {"numeric": MU_NUMERIC_FEATURES, "text": MU_TEXT_FEATURES},
        "policy_feature_config": {"numeric": NUMERIC_FEATURES, "variants": {"P0": "local", "P1": "+previous source", "P2": "+previous source and system target"}, "tfidf": {"word_ngrams": [1, 2], "max_features_per_field": 10000, "min_df": 2}},
        "trained_checkpoint_hashes": checkpoints, "selected_learned_variant": selection["selected_variant"],
        "selected_learned_threshold": selection["selected_threshold"], "baseline_config": BASELINES,
        "evaluation_metric_config": {"quality": ["SacreBLEU corpus BLEU", "chrF2"], "hypothesis_tokenization": "translated_text.split()", "latency": ["standard token-level Average Lagging", "Length-Adaptive Average Lagging", "source-clock statistics"]},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    path = EXPERIMENT / "frozen-eval-config.json"
    if path.exists():
        raise RuntimeError("frozen evaluation config already exists and cannot be overwritten")
    _atomic_json(path, document)
    print(f"output={path} sha256={_sha256(path)}")


def report(split_name: str) -> None:
    metrics = _load_json(EXPERIMENT / f"metrics/{split_name}/all.json")
    output = EXPERIMENT / f"{split_name}-results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["strategy", "BLEU", "chrF2", "token_level_average_lagging", "token_level_length_adaptive_average_lagging", "mean_source_tokens_per_unit", "median_source_tokens_per_unit", "mean_simulated_source_clock_duration_ms", "mean_first_commit_source_tokens", "mean_first_commit_simulated_source_clock_latency_ms", "commits_per_100_source_tokens", "forced_commit_rate"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for strategy, row in metrics.items():
            writer.writerow({"strategy": strategy, **{name: row[name] for name in columns[1:]}})
    per_talk_output = EXPERIMENT / f"{split_name}-per-talk-results.csv"
    with per_talk_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "talk_id", "BLEU", "chrF2"])
        writer.writeheader()
        for strategy, row in metrics.items():
            for talk in row["per_talk"]:
                writer.writerow({"strategy": strategy, **talk})
    print(f"output={output} per_talk_output={per_talk_output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pseudo", "validate-pseudo", "mu-supervision", "validate-mu", "train", "train-mu", "rollout", "rollout-selected", "evaluate", "select", "freeze", "report"))
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--talk-id")
    parser.add_argument("--max-talks", type=int)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--max-source-tokens", type=int)
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pseudo-labels", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--include-selected", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "pseudo":
        pseudo(args.split or "train", args.batch_size, talk_id=args.talk_id, max_talks=args.max_talks, max_states=args.max_states, smoke=args.smoke)
    elif args.stage == "validate-pseudo":
        validate_pseudo(args.split or "train", smoke=args.smoke)
    elif args.stage == "mu-supervision":
        mu_supervision(args.split or "train", args.batch_size, talk_id=args.talk_id, max_talks=args.max_talks, max_states=args.max_states, smoke=args.smoke)
    elif args.stage == "validate-mu":
        validate_mu(args.split or "train", smoke=args.smoke)
    elif args.stage == "train":
        if args.pseudo_labels is None or args.variant is None:
            parser.error("train requires --pseudo-labels MANIFEST and --variant P0|P1|P2")
        train(args.pseudo_labels, args.variant, allow_smoke=args.allow_smoke)
    elif args.stage == "train-mu":
        if args.pseudo_labels is None:
            parser.error("train-mu requires --pseudo-labels MU_MANIFEST")
        train_mu(args.pseudo_labels, allow_smoke=args.allow_smoke)
    elif args.stage == "rollout":
        rollout(args.split or "dev", args.strategies or list(BASELINES), args.batch_size, talk_id=args.talk_id, max_talks=args.max_talks, max_source_tokens=args.max_source_tokens, smoke=args.smoke)
    elif args.stage == "rollout-selected":
        rollout_selected(args.split or "test", args.batch_size)
    elif args.stage == "evaluate":
        evaluate(args.split or "dev", args.strategies or list(BASELINES), smoke=args.smoke, include_selected=args.include_selected)
    elif args.stage == "select":
        select()
    elif args.stage == "freeze":
        freeze()
    else:
        report(args.split or "test")


if __name__ == "__main__":
    main()
