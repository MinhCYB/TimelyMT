"""Read-only P3 real-versus-zero prepared-context DEV report.

Reads only existing P3 DEV predictions and metrics. It never imports model,
translator, rollout, training, or evaluation code, and never reads TEST paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "outputs/experiments/policy-p3-global"
REPORTS = ROOT / "reports"
TALKS = (
    "ted-jeff-dean-ai-smart",
    "ted-luis-von-ahn-crowdsourcing",
    "ted-sims-witherspoon-ai-climate",
)
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def strategy(mode: str, threshold: float) -> str:
    return f"p3_global_{threshold:.2f}" if mode == "real" else f"p3_global_zeroctx_{threshold:.2f}"


def commit_stats(record: dict[str, Any]) -> dict[str, Any]:
    commits = record.get("commits")
    if not isinstance(commits, list) or not commits:
        raise ValueError("Prediction artifact lacks commits")
    lengths = [item["source_token_count"] for item in commits]
    return {
        "commit_count": len(commits),
        "mean_source_tokens_per_commit": mean(lengths),
        "four_token_fraction": lengths.count(4) / len(lengths),
        "four_to_five_token_fraction": sum(4 <= length <= 5 for length in lengths) / len(lengths),
        "long_commit_fraction": sum(length >= 8 for length in lengths) / len(lengths),
        "forced_max_length_commits": sum(item["reason"] == "max_length" for item in commits),
    }


def compare_records(real: dict[str, Any], zero: dict[str, Any]) -> dict[str, Any]:
    real_context, zero_context = real.get("prepared_context"), zero.get("prepared_context")
    if not isinstance(real_context, dict) or not isinstance(zero_context, dict):
        raise ValueError("Prepared-context provenance is absent")
    # Canonical P3 real artifacts were frozen before the ablation field existed.
    # Their absent mode is unambiguously the original real-context behavior.
    if real_context.get("prepared_context_mode", "real") != "real" or zero_context.get("prepared_context_mode") != "zero":
        raise ValueError("Artifacts do not have expected real/zero context modes")
    if real_context.get("eligible_source_ids") != zero_context.get("eligible_source_ids"):
        raise ValueError("Eligible source provenance differs between conditions")
    empty = not real_context.get("has_eligible_context")
    invariance = None
    if empty:
        # Exact JSON equality except condition/provenance fields is the relevant
        # deterministic rollout acceptance check for naturally zero contexts.
        stripped_real, stripped_zero = dict(real), dict(zero)
        for record in (stripped_real, stripped_zero):
            record.pop("strategy", None)
            record.pop("prepared_context_mode", None)
            record.pop("prepared_context_effective_embedding_norm", None)
            context = dict(record["prepared_context"])
            context.pop("prepared_context_mode", None)
            context.pop("prepared_context_effective_embedding_norm", None)
            record["prepared_context"] = context
        invariance = stripped_real == stripped_zero
    return {
        "has_eligible_context": bool(real_context["has_eligible_context"]),
        "eligible_source_ids": real_context["eligible_source_ids"],
        "real_embedding_norm": real_context["embedding_norm"],
        "zero_effective_embedding_norm": zero_context["prepared_context_effective_embedding_norm"],
        "empty_context_artifacts_identical_except_condition": invariance,
        "real_commit_behavior": commit_stats(real),
        "zero_commit_behavior": commit_stats(zero),
    }


def fmt(value: float | int) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def main() -> None:
    real_metrics = load(EXPERIMENT / "metrics/dev/all.json")
    zero_metrics = load(EXPERIMENT / "metrics/dev/all-zeroctx.json")
    output: dict[str, Any] = {"scope": "DEV only; read-only real-versus-zero P3 inference ablation", "thresholds": {}}
    markdown = ["# Controlled P3 Prepared-Context Ablation", "", "## Design", "", "Comparison is **P3_REAL - P3_ZERO_CONTEXT** using the same P3_GLOBAL checkpoint. It tests inference-time reliance on the prepared-global input by the trained policy; it does not establish that another training design could not use context differently.", "", "## Aggregate DEV", "", "| threshold | dBLEU | dchrF2 | dAL | dLAAL | d commits |", "|---:|---:|---:|---:|---:|---:|"]
    per_talk_lines = ["", "## Per-Talk and Commit Behavior", "", "| threshold | context | talk | dBLEU | dchrF2 | real/zero commits | real/zero mean span | empty artifacts identical |", "|---:|---|---|---:|---:|---:|---:|---|"]
    for threshold in THRESHOLDS:
        real_name, zero_name = strategy("real", threshold), strategy("zero", threshold)
        real_metric, zero_metric = real_metrics[real_name], zero_metrics[zero_name]
        aggregate = {name: real_metric[name] - zero_metric[name] for name in ("BLEU", "chrF2", "token_level_average_lagging", "token_level_length_adaptive_average_lagging", "number_of_commits")}
        markdown.append("| " + " | ".join([fmt(threshold), *(fmt(aggregate[name]) for name in aggregate)]) + " |")
        talks: dict[str, Any] = {}
        for talk in TALKS:
            real = load(EXPERIMENT / "predictions/dev" / real_name / f"{talk}.json")
            zero = load(EXPERIMENT / "predictions/dev" / zero_name / f"{talk}.json")
            comparison = compare_records(real, zero)
            real_talk = next(item for item in real_metric["per_talk"] if item["talk_id"] == talk)
            zero_talk = next(item for item in zero_metric["per_talk"] if item["talk_id"] == talk)
            comparison["quality_delta_real_minus_zero"] = {"BLEU": real_talk["BLEU"] - zero_talk["BLEU"], "chrF2": real_talk["chrF2"] - zero_talk["chrF2"]}
            talks[talk] = comparison
            real_stats, zero_stats = comparison["real_commit_behavior"], comparison["zero_commit_behavior"]
            per_talk_lines.append("| " + " | ".join([fmt(threshold), "bearing" if comparison["has_eligible_context"] else "empty", talk, fmt(comparison["quality_delta_real_minus_zero"]["BLEU"]), fmt(comparison["quality_delta_real_minus_zero"]["chrF2"]), f"{real_stats['commit_count']}/{zero_stats['commit_count']}", f"{fmt(real_stats['mean_source_tokens_per_commit'])}/{fmt(zero_stats['mean_source_tokens_per_commit'])}", str(comparison["empty_context_artifacts_identical_except_condition"])]) + " |")
        output["thresholds"][f"{threshold:.2f}"] = {"aggregate_delta_real_minus_zero": aggregate, "per_talk": talks}
    markdown.extend(per_talk_lines)
    markdown.extend(["", "## Interpretation", "", "Positive quality deltas favor real context; negative AL/LAAL deltas favor real context on latency. For naturally empty-context talks, the artifact equality column must be `True` under deterministic inference. A `False` value is an implementation/runtime anomaly to investigate before interpreting context-bearing effects.", ""])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "p3_prepared_context_ablation.md").write_text("\n".join(markdown), encoding="utf-8")
    (REPORTS / "p3_prepared_context_ablation.json").write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
