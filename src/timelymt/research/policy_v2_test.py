"""Fail-closed execution gate for the locked Policy V2 TEST protocol."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from timelymt.data.canonical.core import load_canonical_talk
from timelymt.data.translation_artifacts import stable_fingerprint, translator_identity
from timelymt.translator.envit5 import load_config
from .evaluation import latency_metrics, quality_metrics
from .policy_v2 import (
    DATASET_CHECKSUM, ENCODER_REVISION, EXPERIMENT_LABEL, EXPERIMENT_STATUS, LOCAL_RUNTIME,
    SPLIT_CHECKSUM, TRANSLATOR_FINGERPRINT, V1_SOURCE_COMMIT, atomic_json, load_v2_checkpoint,
    sha256_file, validate_prediction_record, validate_v1_checkpoint_metadata, v1_identity_document,
)
from .streaming import fixed_n, fixed_time, learned_rollout, local_agreement_la2, prediction_record


ROOT = Path(__file__).parents[3]
EXPERIMENT = ROOT / "outputs/experiments/policy-v2"
PLAN_PATH = EXPERIMENT / "test-plan.json"
V1_SOURCE = EXPERIMENT / "v1-source"
V2_FROZEN = EXPERIMENT / "v2-frozen-config.json"
V2_SELECTION = EXPERIMENT / "dev-selection.json"
V2_CHECKPOINTS = ROOT / "checkpoints/policy_v2"
V1_CHECKPOINTS = ROOT / "docs/archive/timelymt-checkpoint/checkpoints/policy"
PSEUDO_TEST = ROOT / "data/policy/pseudo_labels/test"

PRIMARY_STRATEGY = "v2_P2_0.50"
HISTORY_ABLATION = ("v2_P0_0.50", "v2_P1_0.50", "v2_P2_0.50")
REFERENCE_SYSTEMS = ("learned_P1_0.60", "local_agreement_la2", "fixed_n_8", "fixed_time_3200")
TEST_STRATEGIES = (*REFERENCE_SYSTEMS, *HISTORY_ABLATION)
FROZEN_V2_HASHES = {
    "P0": "f7f0c58d7ab4d3ec662aebc697a385c9747ab42ad0df77676aab7c962e03d299",
    "P1": "e1102f6c5245949a46335d61f9a054c1c9dbbdfc235f6946de1a5f3228413fd3",
    "P2": "4d531caf165175a4c8b5ef00b54ad09ef7effb3b5f453f0d3f28e1480263fbe7",
}
FORBIDDEN_OPERATIONS = (
    "training", "fine_tuning", "pseudo_label_generation", "threshold_search", "variant_selection",
    "dev_reselection", "test_based_selection", "hyperparameter_changes", "feature_changes",
    "checkpoint_replacement",
)
EXPECTED_PLAN = {
    "schema_version": "1.0.0", "experiment_label": EXPERIMENT_LABEL,
    "experiment_status": EXPERIMENT_STATUS, "primary_strategy": PRIMARY_STRATEGY,
    "history_ablation": list(HISTORY_ABLATION), "reference_systems": list(REFERENCE_SYSTEMS),
    "pipeline": ["frozen_prediction", "frozen_evaluation", "reporting"],
    "forbidden_operations": list(FORBIDDEN_OPERATIONS),
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"required frozen artifact is missing or invalid: {path}") from error


def _validate_test_plan(path: Path = PLAN_PATH) -> Mapping[str, Any]:
    plan = _load_json(path)
    if plan != EXPECTED_PLAN:
        raise RuntimeError("TEST plan differs from the locked protocol")
    if tuple(dict.fromkeys((*plan["reference_systems"], *plan["history_ablation"]))) != TEST_STRATEGIES:
        raise RuntimeError("TEST plan contains an unexpected, missing, or duplicate strategy")
    return plan


def _validate_v1() -> Mapping[str, Any]:
    metadata = _load_json(V1_SOURCE / "checkpoint-metadata.json")
    validate_v1_checkpoint_metadata(metadata)
    identity = _load_json(V1_SOURCE / "identity.json")
    if any(identity.get(key) != value for key, value in v1_identity_document().items()):
        raise RuntimeError("frozen V1 source identity mismatch")
    selection = _load_json(V1_SOURCE / "dev-selection.json")
    frozen = _load_json(V1_SOURCE / "frozen-eval-config.json")
    if selection.get("selected_strategy") != "learned_P1_0.60":
        raise RuntimeError("frozen V1 selected strategy mismatch")
    if frozen.get("selected_learned_variant") != "P1" or frozen.get("selected_learned_threshold") != 0.6:
        raise RuntimeError("frozen V1 selected policy identity mismatch")
    checkpoint_hashes = frozen.get("trained_checkpoint_hashes", {})
    if set(checkpoint_hashes) != {"P0", "P1", "P2", "mu_zhang2020"}:
        raise RuntimeError("frozen V1 checkpoint hash set mismatch")
    for variant, expected_hash in checkpoint_hashes.items():
        path = V1_CHECKPOINTS / (f"{variant}.joblib" if variant != "mu_zhang2020" else "mu_zhang2020.joblib")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"frozen V1 checkpoint hash mismatch: {variant}")
    return frozen


def _validate_v2() -> Mapping[str, Any]:
    frozen = _load_json(V2_FROZEN)
    selection = _load_json(V2_SELECTION)
    expected = {
        "artifact_status": "v2-dev-frozen-complete", "experiment_status": EXPERIMENT_STATUS,
        "selected_strategy": PRIMARY_STRATEGY, "dataset_checksum": DATASET_CHECKSUM,
        "split_checksum": SPLIT_CHECKSUM, "translator_fingerprint": TRANSLATOR_FINGERPRINT,
        "test_status": "UNTOUCHED", "checkpoint_hashes": FROZEN_V2_HASHES,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise RuntimeError("frozen V2 DEV config identity mismatch")
    if (
        selection.get("selected_strategy") != PRIMARY_STRATEGY
        or selection.get("selected_variant") != "P2"
        or selection.get("selected_threshold") != 0.5
        or selection.get("experiment_status") != EXPERIMENT_STATUS
        or frozen.get("selection_sha256") != sha256_file(V2_SELECTION)
        or frozen.get("metrics_sha256") != sha256_file(EXPERIMENT / "metrics/dev/all.json")
        or frozen.get("v1_metrics_sha256") != sha256_file(V1_SOURCE / "all.json")
    ):
        raise RuntimeError("frozen V2 DEV selection identity mismatch")
    if frozen.get("v1_source_identity") != v1_identity_document():
        raise RuntimeError("frozen V2 config does not bind the frozen V1 identity")
    return frozen


def _validate_checkpoints() -> None:
    from .policy_v2_runner import _valid_checkpoint

    for variant, expected_hash in FROZEN_V2_HASHES.items():
        path = V2_CHECKPOINTS / f"V2{variant}.pt"
        metadata = _load_json(V2_CHECKPOINTS / f"V2{variant}.metadata.json")
        if not _valid_checkpoint(variant) or sha256_file(path) != expected_hash or metadata.get("checkpoint_sha256") != expected_hash:
            raise RuntimeError(f"frozen V2 checkpoint hash or payload mismatch: {variant}")


def _validate_data_and_translator(v1_frozen: Mapping[str, Any]) -> None:
    dataset = _load_json(ROOT / "data/manifests/timelymt-streaming-dataset-v1.json")
    split = _load_json(ROOT / "data/splits/experimental.json")
    if dataset.get("manifest_checksum") != DATASET_CHECKSUM or stable_fingerprint(split) != SPLIT_CHECKSUM:
        raise RuntimeError("frozen dataset or split checksum mismatch")
    identity = asdict(translator_identity(load_config(ROOT / "configs/translator/envit5.json")))
    if identity != v1_frozen.get("translator") or identity.get("config_fingerprint") != TRANSLATOR_FINGERPRINT:
        raise RuntimeError("frozen translator model/revision/config identity mismatch")


def validate_test_gate() -> None:
    print("[1] Validate frozen V1 ............ ", end="")
    v1_frozen = _validate_v1()
    print("OK")
    print("[2] Validate frozen V2 ............ ", end="")
    _validate_v2()
    print("OK")
    print("[3] Validate TEST plan ............ ", end="")
    _validate_test_plan()
    print("OK")
    print("[4] Validate checkpoints .......... ", end="")
    _validate_checkpoints()
    _validate_data_and_translator(v1_frozen)
    if PSEUDO_TEST.exists():
        raise RuntimeError("TEST pseudo-label artifacts are forbidden")
    print("OK")


def _model_hash(strategy: str, v1_frozen: Mapping[str, Any]) -> str:
    if strategy == "learned_P1_0.60":
        return v1_frozen["trained_checkpoint_hashes"]["P1"]
    if strategy.startswith("v2_"):
        return FROZEN_V2_HASHES[strategy.split("_")[1]]
    return "not_applicable"


def _prediction_path(strategy: str, talk_id: str) -> Path:
    return EXPERIMENT / "predictions/test" / strategy / f"{talk_id}.json"


def _validate_test_prediction(record: Mapping[str, Any], strategy: str, talk_id: str, model_hash: str) -> None:
    validate_prediction_record(record, strategy=strategy, talk_id=talk_id, model_hash=model_hash, split_name="test")
    expected = {
        "split_checksum": SPLIT_CHECKSUM, "translator_fingerprint": TRANSLATOR_FINGERPRINT,
        "test_plan_sha256": sha256_file(PLAN_PATH), "primary_strategy": PRIMARY_STRATEGY,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("invalid resumable V2 TEST prediction protocol identity")


def _resume_prediction(path: Path, strategy: str, talk_id: str, model_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        _validate_test_prediction(_load_json(path), strategy, talk_id, model_hash)
        return True
    except RuntimeError:
        path.unlink()
        return False


def _write_prediction(path: Path, strategy: str, talk: Any, commits: Sequence[Any], model_hash: str) -> None:
    record = prediction_record(strategy, talk, commits)
    record.update({
        "artifact_status": "full", "publishable": False, "experiment_status": EXPERIMENT_STATUS,
        "experiment_label": EXPERIMENT_LABEL, "model_sha256": model_hash,
        "dataset_checksum": DATASET_CHECKSUM, "split_checksum": SPLIT_CHECKSUM,
        "translator_fingerprint": TRANSLATOR_FINGERPRINT, "encoder_revision": ENCODER_REVISION,
        "runtime": dict(LOCAL_RUNTIME), "test_plan_sha256": sha256_file(PLAN_PATH),
        "primary_strategy": PRIMARY_STRATEGY,
    })
    atomic_json(path, record)
    _validate_test_prediction(record, strategy, talk.talk_id, model_hash)


def _rollout_references() -> None:
    import joblib
    from .cli import _manifests, _runtime_talk, _translator

    _, split = _manifests()
    talk_ids = list(split["splits"]["test"])
    v1_frozen = _load_json(V1_SOURCE / "frozen-eval-config.json")
    learned = joblib.load(V1_CHECKPOINTS / "P1.joblib")
    _, provider = _translator(1, device=LOCAL_RUNTIME["translator_device"])
    for strategy in REFERENCE_SYSTEMS:
        model_hash = _model_hash(strategy, v1_frozen)
        for talk_id in talk_ids:
            path = _prediction_path(strategy, talk_id)
            if _resume_prediction(path, strategy, talk_id, model_hash):
                continue
            talk = _runtime_talk(talk_id, split)
            if strategy == "learned_P1_0.60":
                commits = learned_rollout(talk, provider, learned, 0.60)
            elif strategy == "local_agreement_la2":
                commits = local_agreement_la2(talk, provider)
            elif strategy == "fixed_n_8":
                commits = fixed_n(talk, provider, 8)
            else:
                commits = fixed_time(talk, provider, 3200)
            _write_prediction(path, strategy, talk, commits, model_hash)


def _rollout_v2(strategy: str) -> None:
    from .cli import _manifests, _runtime_talk, _translator
    from .policy_v2_runner import _encoder_cache

    variant = strategy.split("_")[1]
    metadata_path = V2_CHECKPOINTS / f"V2{variant}.metadata.json"
    _, cache = _encoder_cache(device="cpu", dtype="float32")
    policy = load_v2_checkpoint(V2_CHECKPOINTS / f"V2{variant}.pt", metadata_path, cache, device="cpu")
    _, provider = _translator(1, device="cuda")
    _, split = _manifests()
    model_hash = FROZEN_V2_HASHES[variant]
    for talk_id in split["splits"]["test"]:
        path = _prediction_path(strategy, talk_id)
        if _resume_prediction(path, strategy, talk_id, model_hash):
            continue
        talk = _runtime_talk(talk_id, split)
        _write_prediction(path, strategy, talk, learned_rollout(talk, provider, policy, 0.50), model_hash)


def _prediction_records(strategy: str, expected_talks: set[str], model_hash: str) -> list[Mapping[str, Any]]:
    paths = sorted((EXPERIMENT / "predictions/test" / strategy).glob("*.json"))
    records = [_load_json(path) for path in paths]
    if {record.get("talk_id") for record in records} != expected_talks or len(records) != len(expected_talks):
        raise RuntimeError(f"TEST evaluation requires full exact prediction coverage: {strategy}")
    for record in records:
        _validate_test_prediction(record, strategy, record["talk_id"], model_hash)
    return records


def _resumable_metrics(expected_talks: set[str], v1_frozen: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]] | None:
    path = EXPERIMENT / "metrics/test/all.json"
    if not path.is_file():
        return None
    try:
        metrics = _load_json(path)
        if set(metrics) != set(TEST_STRATEGIES):
            return None
        required = {
            "BLEU", "chrF2", "token_level_average_lagging",
            "token_level_length_adaptive_average_lagging", "number_of_commits",
            "commits_per_100_source_tokens", "per_talk",
        }
        for strategy, row in metrics.items():
            expected = {
                "artifact_status": "full", "experiment_status": EXPERIMENT_STATUS,
                "experiment_label": EXPERIMENT_LABEL, "dataset_checksum": DATASET_CHECKSUM,
                "split_checksum": SPLIT_CHECKSUM, "primary_strategy": PRIMARY_STRATEGY,
                "model_sha256": _model_hash(strategy, v1_frozen),
                "test_plan_sha256": sha256_file(PLAN_PATH),
            }
            if (
                not required.issubset(row)
                or any(row.get(key) != value for key, value in expected.items())
                or {item.get("talk_id") for item in row["per_talk"]} != expected_talks
            ):
                return None
        return metrics
    except (RuntimeError, TypeError, KeyError):
        return None


def _evaluate_test() -> Mapping[str, Mapping[str, Any]]:
    from .cli import _talk_paths

    split = _load_json(ROOT / "data/splits/experimental.json")
    expected = set(split["splits"]["test"])
    v1_frozen = _load_json(V1_SOURCE / "frozen-eval-config.json")
    resumed = _resumable_metrics(expected, v1_frozen)
    if resumed is not None:
        return resumed
    records_by_strategy = {
        strategy: _prediction_records(strategy, expected, _model_hash(strategy, v1_frozen))
        for strategy in TEST_STRATEGIES
    }
    references = {}
    talk_paths = _talk_paths()
    for talk_id in expected:
        document = load_canonical_talk(talk_paths[talk_id])
        references[talk_id] = " ".join(segment["text"] for segment in document["target_reference"]["segments"])
    metrics = {}
    for strategy, records in records_by_strategy.items():
        row = {**quality_metrics(records, references), **latency_metrics(records, references)}
        row.update({
            "artifact_status": "full", "publishable": False, "experiment_status": EXPERIMENT_STATUS,
            "experiment_label": EXPERIMENT_LABEL, "dataset_checksum": DATASET_CHECKSUM,
            "split_checksum": SPLIT_CHECKSUM, "primary_strategy": PRIMARY_STRATEGY,
            "model_sha256": _model_hash(strategy, v1_frozen), "test_plan_sha256": sha256_file(PLAN_PATH),
        })
        metrics[strategy] = row
        atomic_json(EXPERIMENT / f"metrics/test/{strategy}.json", row)
    atomic_json(EXPERIMENT / "metrics/test/all.json", metrics)
    return metrics


def _atomic_csv(path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    columns = [
        "strategy", "primary_strategy", "BLEU", "chrF2", "token_level_average_lagging",
        "token_level_length_adaptive_average_lagging", "number_of_commits",
        "commits_per_100_source_tokens", "per_talk_BLEU", "per_talk_chrF2",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False, suffix=".tmp") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for strategy, row in metrics.items():
            writer.writerow({
                "strategy": strategy, "primary_strategy": strategy == PRIMARY_STRATEGY,
                "BLEU": row["BLEU"], "chrF2": row["chrF2"],
                "token_level_average_lagging": row["token_level_average_lagging"],
                "token_level_length_adaptive_average_lagging": row["token_level_length_adaptive_average_lagging"],
                "number_of_commits": row["number_of_commits"],
                "commits_per_100_source_tokens": row["commits_per_100_source_tokens"],
                "per_talk_BLEU": json.dumps({item["talk_id"]: item["BLEU"] for item in row["per_talk"]}, sort_keys=True),
                "per_talk_chrF2": json.dumps({item["talk_id"]: item["chrF2"] for item in row["per_talk"]}, sort_keys=True),
            })
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_summary(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    summary = {
        "schema_version": "1.0.0", "experiment_status": EXPERIMENT_STATUS,
        "experiment_label": EXPERIMENT_LABEL, "primary_strategy": PRIMARY_STRATEGY,
        "selection_performed": False, "test_plan_sha256": sha256_file(PLAN_PATH),
        "strategies": metrics,
    }
    atomic_json(EXPERIMENT / "test-results.json", summary)
    _atomic_csv(EXPERIMENT / "test-results.csv", metrics)


def run_test() -> None:
    validate_test_gate()
    print("\nTEST GATE OPEN\n")
    print("[5] Reference TEST predictions .... RUN/SKIP")
    _rollout_references()
    for index, strategy in enumerate(HISTORY_ABLATION, start=6):
        print(f"[{index}] {strategy} ................ RUN/SKIP")
        _rollout_v2(strategy)
    print("[9] Evaluate TEST ................. RUN/SKIP")
    metrics = _evaluate_test()
    print("[10] Write TEST summary ........... RUN/SKIP")
    _write_summary(metrics)
    print("\nTEST COMPLETE")


def main() -> None:
    run_test()


if __name__ == "__main__":
    main()
