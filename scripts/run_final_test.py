"""Execute the pre-registered, one-shot TimelyMT held-out TEST evaluation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import sacrebleu
import sklearn
import torch
import transformers

from timelymt.data.canonical.core import load_canonical_talk
from timelymt.data.translation_artifacts import translator_identity
from timelymt.translator.cache import TranslationCache
from timelymt.translator.envit5 import EnViT5Translator, load_config
from timelymt.research.cli import Provider, _manifests, _runtime_talk, _talk_paths
from timelymt.research.evaluation import latency_metrics, quality_metrics
from timelymt.research.policy_p3_global import PreparedGlobalEmbedding, load_p3_checkpoint
from timelymt.research.policy_v2 import ENCODER_REVISION, EmbeddingCache, FrozenMiniLMEncoder, load_v2_checkpoint
from timelymt.research.policy_v2_runner import _valid_checkpoint
from timelymt.research.streaming import learned_rollout, prediction_record


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "outputs/final_test"
PRETEST = OUTPUT / "pretest-freeze.json"
P3_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
EXPECTED = {
    "p3": "ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9",
    "prepared": "d9b910afd1941873826065bcf6e343be28cd850d339b356457daadbde60ad2eb",
    "v1": "6e1d273bf2008178265ade654729bb9710e7acfa67b8caa70a8a6114cf8508c4",
    "v2": "4d531caf165175a4c8b5ef00b54ad09ef7effb3b5f453f0d3f28e1480263fbe7",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_freeze() -> tuple[Mapping[str, Any], list[str]]:
    if not PRETEST.is_file():
        raise RuntimeError("pretest-freeze.json must exist before TEST access")
    existing = [path for path in OUTPUT.rglob("*") if path.is_file() and path != PRETEST]
    if existing:
        raise RuntimeError(f"final TEST area is not pristine: {existing}")
    paths = {
        "p3": ROOT / "checkpoints/policy_p3_global/P3_GLOBAL.pt",
        "prepared": ROOT / "data/prepared_context/manifest.json",
        "v1": ROOT / "docs/archive/timelymt-checkpoint/checkpoints/policy/P1.joblib",
        "v2": ROOT / "checkpoints/policy_v2/V2P2.pt",
    }
    for name, path in paths.items():
        if sha256(path) != EXPECTED[name]:
            raise RuntimeError(f"protected {name} fingerprint mismatch")
    if not _valid_checkpoint("P2"):
        raise RuntimeError("frozen V2 P2 checkpoint payload is invalid")
    manifest = json.loads(paths["prepared"].read_text(encoding="utf-8"))
    if any(entry["split"] == "test" for entry in manifest["pools"]):
        raise RuntimeError("prepared-context manifest unexpectedly contains TEST")
    v1_frozen = json.loads((ROOT / "outputs/experiments/policy-v2/v1-source/frozen-eval-config.json").read_text(encoding="utf-8"))
    v2_frozen = json.loads((ROOT / "outputs/experiments/policy-v2/v2-frozen-config.json").read_text(encoding="utf-8"))
    if (v1_frozen["selected_learned_variant"], v1_frozen["selected_learned_threshold"]) != ("P1", 0.6):
        raise RuntimeError("V1 frozen selection mismatch")
    if v2_frozen["selected_strategy"] != "v2_P2_0.50":
        raise RuntimeError("V2 frozen selection mismatch")
    _, split = _manifests()
    return split, list(split["splits"]["test"])


def translator_provider() -> Provider:
    config = load_config(ROOT / "configs/translator/envit5.json")
    identity = translator_identity(config)
    translator = EnViT5Translator(
        config,
        device="cuda",
        cache=TranslationCache(OUTPUT / "cache/translator"),
    )
    return Provider(translator, identity, 1)


def write_prediction(group: str, strategy: str, talk: Any, commits: Sequence[Any], extra: Mapping[str, Any]) -> None:
    record = prediction_record(strategy, talk, commits)
    record.update(extra)
    atomic_json(OUTPUT / group / "predictions" / strategy / f"{talk.talk_id}.json", record)


def prediction_records(group: str, strategy: str, expected_talks: Sequence[str]) -> list[Mapping[str, Any]]:
    paths = sorted((OUTPUT / group / "predictions" / strategy).glob("*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(records) != len(expected_talks) or {row["talk_id"] for row in records} != set(expected_talks):
        raise RuntimeError(f"incomplete TEST prediction coverage for {strategy}")
    return records


def metric_row(records: Sequence[Mapping[str, Any]], references: Mapping[str, str]) -> dict[str, Any]:
    return {**quality_metrics(records, references), **latency_metrics(records, references)}


def format_value(value: float) -> str:
    return f"{value:.4f}"


def dev_row(path: Path, strategy: str) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))[strategy]


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    split, talk_ids = validate_freeze()
    commands = [
        "Get-FileHash -Algorithm SHA256 checkpoints/policy_p3_global/P3_GLOBAL.pt",
        "Get-FileHash -Algorithm SHA256 data/prepared_context/manifest.json",
        "git status --porcelain=v1 -uall",
        "C:/ProgramData/miniconda3/envs/timelymt-v2/python.exe scripts/run_final_test.py",
        "python -m pytest tests/research/test_research_mvp.py tests/research/test_policy_v2.py tests/research/test_policy_p3_global.py tests/research/test_policy_v2_test.py tests/data/test_prepared_context_artifacts.py",
    ]
    provider = translator_provider()

    # Fixed execution order: complete all predictions before loading references.
    v1_strategy = "learned_P1_0.60"
    v1_policy = joblib.load(ROOT / "docs/archive/timelymt-checkpoint/checkpoints/policy/P1.joblib")
    for talk_id in talk_ids:
        talk = _runtime_talk(talk_id, split)
        write_prediction("v1", v1_strategy, talk, learned_rollout(talk, provider, v1_policy, 0.60), {
            "model_sha256": EXPECTED["v1"], "threshold": 0.60,
        })

    v2_strategy = "v2_P2_0.50"
    v2_encoder = FrozenMiniLMEncoder(device="cpu", dtype="float32", batch_size=256)
    v2_cache = EmbeddingCache(OUTPUT / "cache/minilm", v2_encoder)
    v2_policy = load_v2_checkpoint(
        ROOT / "checkpoints/policy_v2/V2P2.pt",
        ROOT / "checkpoints/policy_v2/V2P2.metadata.json",
        v2_cache,
        device="cpu",
    )
    for talk_id in talk_ids:
        talk = _runtime_talk(talk_id, split)
        write_prediction("v2", v2_strategy, talk, learned_rollout(talk, provider, v2_policy, 0.50), {
            "encoder_revision": ENCODER_REVISION, "model_sha256": EXPECTED["v2"], "threshold": 0.50,
        })

    p3_encoder = FrozenMiniLMEncoder(device="cuda", dtype="float32", batch_size=256, allow_cuda_float32=True)
    p3_cache = EmbeddingCache(OUTPUT / "cache/minilm", p3_encoder)
    p3_metadata = ROOT / "checkpoints/policy_p3_global/P3_GLOBAL.metadata.json"
    for talk_id in talk_ids:
        talk = _runtime_talk(talk_id, split)
        prepared = PreparedGlobalEmbedding(
            talk_id=talk_id,
            split="test",
            eligible_source_ids=(),
            eligible_source_checksums=(),
            embedding=np.zeros(384, dtype=np.float32),
        )
        if prepared.embedding.dtype != np.float32 or prepared.embedding.shape != (384,) or np.any(prepared.embedding):
            raise RuntimeError("P3 TEST prepared input is not exact float32 zeros(384)")
        p3_policy = load_p3_checkpoint(
            ROOT / "checkpoints/policy_p3_global/P3_GLOBAL.pt",
            p3_metadata,
            p3_cache,
            prepared,
            manifest_path=ROOT / "data/prepared_context/manifest.json",
            device="cpu",
            prepared_context_mode="zero",
        )
        for threshold in P3_THRESHOLDS:
            strategy = f"p3_global_zeroctx_{threshold:.2f}"
            write_prediction("p3", strategy, talk, learned_rollout(talk, provider, p3_policy, threshold), {
                "model_sha256": EXPECTED["p3"],
                "prepared_context": prepared.provenance("zero"),
                "prepared_context_manifest_fingerprint": EXPECTED["prepared"],
                "prepared_context_mode": "zero",
                "prepared_context_vector": "exact float32 zeros(384)",
                "threshold": threshold,
            })

    talk_paths = _talk_paths()
    references = {
        talk_id: " ".join(
            segment["text"]
            for segment in load_canonical_talk(talk_paths[talk_id])["target_reference"]["segments"]
        )
        for talk_id in talk_ids
    }
    metrics: dict[str, Any] = {
        v1_strategy: metric_row(prediction_records("v1", v1_strategy, talk_ids), references),
        v2_strategy: metric_row(prediction_records("v2", v2_strategy, talk_ids), references),
    }
    for threshold in P3_THRESHOLDS:
        strategy = f"p3_global_zeroctx_{threshold:.2f}"
        metrics[strategy] = metric_row(prediction_records("p3", strategy, talk_ids), references)
    atomic_json(OUTPUT / "final-test-metrics.json", metrics)

    dev = {
        v1_strategy: dev_row(ROOT / "outputs/experiments/policy-v2/v1-source/all.json", v1_strategy),
        v2_strategy: dev_row(ROOT / "outputs/experiments/policy-v2/metrics/dev/all.json", v2_strategy),
        **{
            f"p3_global_zeroctx_{threshold:.2f}": dev_row(
                ROOT / "outputs/experiments/policy-p3-global/metrics/dev/all-zeroctx.json",
                f"p3_global_zeroctx_{threshold:.2f}",
            )
            for threshold in P3_THRESHOLDS
        },
    }
    rows = []
    comparison_rows = []
    for strategy, row in metrics.items():
        threshold = float(strategy.rsplit("_", 1)[1])
        rows.append(
            f"| {strategy} | {threshold:.2f} | {format_value(row['BLEU'])} | {format_value(row['chrF2'])} | "
            f"{format_value(row['token_level_average_lagging'])} | {format_value(row['token_level_length_adaptive_average_lagging'])} | {int(row['number_of_commits'])} |"
        )
        old = dev[strategy]
        comparison_rows.append(
            f"| {strategy} | {format_value(old['BLEU'])} | {format_value(row['BLEU'])} | "
            f"{format_value(old['chrF2'])} | {format_value(row['chrF2'])} | "
            f"{format_value(old['token_level_average_lagging'])} | {format_value(row['token_level_average_lagging'])} | "
            f"{format_value(old['token_level_length_adaptive_average_lagging'])} | {format_value(row['token_level_length_adaptive_average_lagging'])} |"
        )
    summary = "\n".join([
        "# TimelyMT Final Held-Out TEST Evaluation",
        "",
        "## Protocol",
        "",
        "TEST was opened only after the design and artifact freeze. No post-TEST tuning is permitted. No prepared TEST context was constructed; P3 used only the exact float32 zero vector of dimension 384.",
        "",
        "The causal prepared-context REAL/ZERO result remains a DEV controlled experiment. This TEST phase does not claim to independently validate prepared-context benefit on unseen context-bearing talks.",
        "",
        "## Evaluated Configurations",
        "",
        "- V1: `learned_P1_0.60`.",
        "- V2: `v2_P2_0.50`.",
        "- P3: no pre-existing selected threshold was found, so the pre-registered frozen grid `0.30, 0.40, 0.50, 0.60, 0.70` was evaluated with ZERO context. No TEST-based selection is made.",
        "",
        "## TEST Results",
        "",
        "| model/configuration | threshold | BLEU | chrF2 | AL | LAAL | commits |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## DEV vs TEST Descriptive Comparison",
        "",
        "These are descriptive frozen-split comparisons only. They do not define a new operating point.",
        "",
        "| configuration | DEV BLEU | TEST BLEU | DEV chrF2 | TEST chrF2 | DEV AL | TEST AL | DEV LAAL | TEST LAAL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *comparison_rows,
        "",
        "## Interpretation",
        "",
        "The table reports held-out behavior as observed. Relative differences and DEV-to-TEST shifts are descriptive only. No P3 threshold is designated best or selected from TEST. Because every P3 TEST run used ZERO context, these results neither support nor refute prepared-context benefit on TEST.",
        "",
        "## Post-TEST Freeze",
        "",
        "No model, threshold, feature, training procedure, prepared-context representation, or metric will be changed based on these TEST results.",
        "",
    ])
    (OUTPUT / "final-test-summary.md").write_text(summary, encoding="utf-8")

    produced = sorted(path for path in OUTPUT.rglob("*") if path.is_file())
    provenance = {
        "commands": commands,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_paths": [
            "configs/experiments/research-mvp.json",
            "configs/experiments/policy-v2.json",
            "configs/experiments/policy-p3-global.json",
            "configs/translator/envit5.json",
        ],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_after": subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True).splitlines(),
        "hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in produced},
        "models": {
            "p3_checkpoint_sha256": EXPECTED["p3"],
            "prepared_context_manifest_sha256": EXPECTED["prepared"],
            "v1_checkpoint_sha256": EXPECTED["v1"],
            "v2_checkpoint_sha256": EXPECTED["v2"],
        },
        "runtime": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "sacrebleu": sacrebleu.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "started_at_utc": started,
        "test_talk_ids": talk_ids,
    }
    atomic_json(OUTPUT / "provenance.json", provenance)
    print("FINAL TEST COMPLETE; experimentation is frozen")


if __name__ == "__main__":
    main()
