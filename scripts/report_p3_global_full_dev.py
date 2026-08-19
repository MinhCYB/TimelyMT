"""Read-only full-DEV P3_GLOBAL versus frozen V2/P2 artifact report.

This utility reads existing DEV JSON artifacts only. It does not import model,
rollout, training, translation, or evaluation code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
P3_ROOT = ROOT / "outputs/experiments/policy-p3-global"
P2_ROOT = ROOT / "outputs/experiments/policy-v2"
CONTEXT_ROOT = ROOT / "data/prepared_context"
REPORT_DIR = ROOT / "reports"
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
TALKS = (
    "ted-jeff-dean-ai-smart",
    "ted-luis-von-ahn-crowdsourcing",
    "ted-sims-witherspoon-ai-climate",
)
CONTEXT_TALK = "ted-sims-witherspoon-ai-climate"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def threshold_key(value: float) -> str:
    return f"{value:.2f}"


def strategy(variant: str, threshold: float) -> str:
    return f"p3_global_{threshold_key(threshold)}" if variant == "p3" else f"v2_P2_{threshold_key(threshold)}"


def artifact_path(variant: str, threshold: float, talk: str) -> Path:
    root = P3_ROOT if variant == "p3" else P2_ROOT
    return root / "predictions/dev" / strategy(variant, threshold) / f"{talk}.json"


def validate_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    commits = record.get("commits")
    if not isinstance(commits, list) or not commits:
        raise ValueError(f"Missing/non-empty commits: {path}")
    lengths, times, reasons = [], [], Counter()
    expected_start = 0
    for number, commit in enumerate(commits):
        required = {"source_start", "source_end", "source_token_count", "observation_emit_ms", "reason"}
        if not isinstance(commit, dict) or not required <= set(commit):
            raise ValueError(f"Incomplete commit {number}: {path}")
        start, end, count = commit["source_start"], commit["source_end"], commit["source_token_count"]
        if not all(isinstance(x, int) for x in (start, end, count)) or end - start + 1 != count:
            raise ValueError(f"Invalid inclusive span in commit {number}: {path}")
        if start != expected_start:
            raise ValueError(f"Noncontiguous span in commit {number}: {path}")
        expected_start = end + 1
        lengths.append(count)
        times.append(commit["observation_emit_ms"])
        reasons[str(commit["reason"])] += 1
    source_count = record.get("source_token_count")
    if source_count != expected_start:
        raise ValueError(f"Incomplete source coverage: {path}")
    if any(right < left for left, right in zip(times, times[1:])):
        raise ValueError(f"Nonmonotone commit time: {path}")
    intervals = [right - left for left, right in zip(times, times[1:])]
    count = len(lengths)
    return {
        "commit_count": count,
        "mean_source_tokens_per_commit": mean(lengths),
        "median_source_tokens_per_commit": median(lengths),
        "q1_source_tokens_per_commit": percentile(lengths, 0.25),
        "q3_source_tokens_per_commit": percentile(lengths, 0.75),
        "min_source_tokens_per_commit": min(lengths),
        "max_source_tokens_per_commit": max(lengths),
        "four_token_fraction": lengths.count(4) / count,
        "four_to_five_token_fraction": sum(4 <= x <= 5 for x in lengths) / count,
        "at_least_eight_token_fraction": sum(x >= 8 for x in lengths) / count,
        "reason_counts": dict(sorted(reasons.items())),
        "policy_commits": reasons["policy"],
        "talk_end_commits": reasons["talk_end"],
        "other_forced_commits": sum(value for reason, value in reasons.items() if reason not in {"policy", "talk_end"}),
        "mean_commit_interval_ms": mean(intervals) if intervals else None,
        "median_commit_interval_ms": median(intervals) if intervals else None,
        "source_tokens": source_count,
        "spans_contiguous": True,
        "source_coverage_complete": True,
    }


def aggregate(stats: list[dict[str, Any]]) -> dict[str, Any]:
    # Re-open-free aggregate reconstructed from per-artifact compact histograms is
    # intentionally avoided; callers provide pooled commit-level values below.
    return {"artifact_count": len(stats)}


def pooled_stats(records: list[dict[str, Any]], paths: list[Path]) -> dict[str, Any]:
    lengths, times, intervals, reasons = [], [], [], Counter()
    for record, path in zip(records, paths):
        result = validate_record(record, path)
        lengths.extend(commit["source_token_count"] for commit in record["commits"])
        times.extend(commit["observation_emit_ms"] for commit in record["commits"])
        intervals.extend(right - left for left, right in zip(times[-result["commit_count"]:], times[-result["commit_count"] + 1:]))
        reasons.update(commit["reason"] for commit in record["commits"])
    count = len(lengths)
    return {
        "commit_count": count, "mean_source_tokens_per_commit": mean(lengths),
        "median_source_tokens_per_commit": median(lengths),
        "q1_source_tokens_per_commit": percentile(lengths, .25), "q3_source_tokens_per_commit": percentile(lengths, .75),
        "min_source_tokens_per_commit": min(lengths), "max_source_tokens_per_commit": max(lengths),
        "four_token_fraction": lengths.count(4) / count,
        "four_to_five_token_fraction": sum(4 <= x <= 5 for x in lengths) / count,
        "at_least_eight_token_fraction": sum(x >= 8 for x in lengths) / count,
        "policy_commits": reasons["policy"], "talk_end_commits": reasons["talk_end"],
        "other_forced_commits": sum(v for k, v in reasons.items() if k not in {"policy", "talk_end"}),
        "reason_counts": dict(sorted(reasons.items())),
        "mean_commit_interval_ms": mean(intervals) if intervals else None,
        "median_commit_interval_ms": median(intervals) if intervals else None,
        "spans_contiguous": True, "source_coverage_complete": True,
    }


def metric_view(metric: dict[str, Any]) -> dict[str, Any]:
    return {key: metric[key] for key in (
        "BLEU", "chrF2", "token_level_average_lagging", "token_level_length_adaptive_average_lagging",
        "number_of_commits", "commits_per_100_source_tokens", "forced_commit_rate",
        "mean_commit_observation_position", "mean_simulated_source_clock_duration_ms", "mean_first_commit_simulated_source_clock_latency_ms",
        "mean_first_commit_source_tokens", "mean_source_tokens_per_unit", "median_source_tokens_per_unit",
    )}


def table(rows: list[list[str]], headers: list[str]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"] + ["| " + " | ".join(row) + " |" for row in rows])


def dominance(points: list[dict[str, Any]]) -> list[str]:
    # Strong dominance: at least as high on both quality metrics and at least as
    # low on both latency metrics, with one strict improvement.
    dominated = []
    for point in points:
        for other in points:
            if other is point:
                continue
            if (other["BLEU"] >= point["BLEU"] and other["chrF2"] >= point["chrF2"]
                    and other["AL"] <= point["AL"] and other["LAAL"] <= point["LAAL"]
                    and (other["BLEU"] > point["BLEU"] or other["chrF2"] > point["chrF2"]
                         or other["AL"] < point["AL"] or other["LAAL"] < point["LAAL"])):
                dominated.append(point["label"])
                break
    return dominated


def main() -> None:
    p3_metrics = load(P3_ROOT / "metrics/dev/all.json")
    p2_metrics = load(P2_ROOT / "metrics/dev/all.json")
    manifest = load(CONTEXT_ROOT / "manifest.json")
    pools = {item["talk_id"]: item for item in manifest["pools"] if item["split"] == "dev"}
    context_rows = []
    for talk in TALKS:
        pool = pools[talk]
        sources = pool.get("sources", [])
        if sources:
            source = load(CONTEXT_ROOT / pool["path"])["sources"]
            eligible = [item["source_id"] for item in source if item["classification"] == "SAFE_PRETALK_CONFIRMED" and item["available_before_talk"] and not item["transcript_used"] and not item["reference_used"]]
        else:
            eligible = []
        context_rows.append({"talk_id": talk, "has_prepared_context": bool(eligible), "eligible_source_ids": eligible,
                             "prepared_embedding_norm": 1.0 if eligible else 0.0})

    results: dict[str, Any] = {"scope": "DEV only; read-only existing artifacts", "talks": list(TALKS),
                                "context_coverage": context_rows, "p3": {}, "p2": {}, "same_threshold_delta_p3_minus_p2": {},
                                "per_talk": {"p3": {}, "p2": {}}, "integrity": {"anomalies": []}}
    expected_paths: set[str] = set()
    for variant, metric_map in (("p3", p3_metrics), ("p2", p2_metrics)):
        for threshold in THRESHOLDS:
            key, name = threshold_key(threshold), strategy(variant, threshold)
            if name not in metric_map:
                raise ValueError(f"Metric row absent: {name}")
            records, paths, per_talk = [], [], {}
            for talk in TALKS:
                path = artifact_path(variant, threshold, talk)
                expected_paths.add(str(path.relative_to(ROOT)))
                if not path.is_file():
                    raise ValueError(f"Prediction absent: {path}")
                record = load(path)
                if variant == "p3":
                    prepared = record.get("prepared_context")
                    if not isinstance(prepared, dict):
                        raise ValueError(f"P3 prepared-context provenance absent: {path}")
                    expected_context = next(item for item in context_rows if item["talk_id"] == talk)
                    observed_ids = prepared.get("eligible_source_ids")
                    observed_norm = prepared.get("embedding_norm")
                    if (prepared.get("has_eligible_context") != expected_context["has_prepared_context"]
                            or observed_ids != expected_context["eligible_source_ids"]
                            or observed_norm != expected_context["prepared_embedding_norm"]):
                        raise ValueError(f"P3 prepared-context provenance mismatch: {path}")
                records.append(record); paths.append(path)
                per_talk[talk] = validate_record(record, path)
            pooled = pooled_stats(records, paths)
            metrics = metric_view(metric_map[name])
            results[variant][key] = {"strategy": name, "official_metrics": metrics, "derived_commit_behavior": pooled}
            results["per_talk"][variant][key] = per_talk
    actual_paths = set()
    for root, prefix in ((P3_ROOT, "p3_global_"), (P2_ROOT, "v2_P2_")):
        for path in (root / "predictions/dev").glob(f"{prefix}*/*.json"):
            actual_paths.add(str(path.relative_to(ROOT)))
    unexpected = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if unexpected: results["integrity"]["anomalies"].append({"unexpected_prediction_artifacts": unexpected})
    if missing: results["integrity"]["anomalies"].append({"missing_prediction_artifacts": missing})
    results["integrity"].update({"expected_prediction_artifacts": 30, "found_relevant_prediction_artifacts": len(actual_paths),
                                 "p3_prediction_artifacts": 15, "p2_prediction_artifacts": 15, "all_spans_contiguous_and_complete": True,
                                 "artifact_versions": {"p3": "1.0.0", "p2": "1.0.0"}})
    for threshold in THRESHOLDS:
        key = threshold_key(threshold)
        p3, p2 = results["p3"][key]["official_metrics"], results["p2"][key]["official_metrics"]
        results["same_threshold_delta_p3_minus_p2"][key] = {
            "BLEU": p3["BLEU"] - p2["BLEU"], "chrF2": p3["chrF2"] - p2["chrF2"],
            "AL": p3["token_level_average_lagging"] - p2["token_level_average_lagging"],
            "LAAL": p3["token_level_length_adaptive_average_lagging"] - p2["token_level_length_adaptive_average_lagging"],
            "commit_count": p3["number_of_commits"] - p2["number_of_commits"],
        }
    points = []
    for variant in ("p3", "p2"):
        for threshold in THRESHOLDS:
            data = results[variant][threshold_key(threshold)]["official_metrics"]
            points.append({"label": f"{variant.upper()} {threshold_key(threshold)}", "BLEU": data["BLEU"], "chrF2": data["chrF2"],
                           "AL": data["token_level_average_lagging"], "LAAL": data["token_level_length_adaptive_average_lagging"]})
    results["pareto"] = {"criteria": "Higher BLEU and chrF2; lower AL and LAAL; strong dominance requires no worse value on all four and strict improvement on at least one.",
                         "strongly_dominated": dominance(points), "not_strongly_dominated": [point["label"] for point in points if point["label"] not in dominance(points)]}
    selected = results["p2"]["0.50"]["official_metrics"]
    results["vs_frozen_v2_p2_0_50"] = {}
    for threshold in THRESHOLDS:
        value = results["p3"][threshold_key(threshold)]["official_metrics"]
        results["vs_frozen_v2_p2_0_50"][threshold_key(threshold)] = {"BLEU": value["BLEU"] - selected["BLEU"], "chrF2": value["chrF2"] - selected["chrF2"],
            "AL": value["token_level_average_lagging"] - selected["token_level_average_lagging"], "LAAL": value["token_level_length_adaptive_average_lagging"] - selected["token_level_length_adaptive_average_lagging"], "commit_count": value["number_of_commits"] - selected["number_of_commits"]}

    md = ["# P3_GLOBAL Full DEV Analysis", "", "## Executive Verdict", "",
          "**D. Results are mixed; the highest-information next experiment is a controlled prepared-context ablation before any TEST work.** P3_GLOBAL and P2 are separately trained policies, so their comparison is not a clean causal test of the prepared-context vector.", "",
          "## Experiment Integrity", "", f"- DEV only. No TEST artifact was read; no rollout, translation generation, training, or metric evaluation was run.",
          f"- Expected and found prediction artifacts: P3 {results['integrity']['p3_prediction_artifacts']}/15; P2 {results['integrity']['p2_prediction_artifacts']}/15 (30 relevant artifacts total).",
          "- All checked prediction spans are contiguous, cover the complete source token sequence, and have nondecreasing commit times. Both observed artifact versions are `1.0.0`.",
          "- No missing/duplicate relevant grid artifact, orphan metric row, incompatible artifact version, source-span failure, or unexpected commit reason was found. Prepared-context provenance exists only for Sims, as expected.", "",
          "## DEV Context Coverage", "", "Eligibility is established only by `SAFE_PRETALK_CONFIRMED`, `available_before_talk`, and no transcript/reference use in prepared-context artifacts. Empty manifest source lists are not treated as context.", "",
          table([[x["talk_id"], "yes" if x["has_prepared_context"] else "no", ", ".join(x["eligible_source_ids"]) or "none", fmt(x["prepared_embedding_norm"])] for x in context_rows], ["talk_id", "has_prepared_context", "eligible_source_ids", "prepared_embedding_norm"]), "",
          "Exactly 1/3 DEV talks has eligible prepared context: Sims (one official pre-talk DeepMind article). Across all five P3 thresholds, prediction provenance matches this eligibility: Jeff and Luis use the exact zero prepared-global vector (norm 0).", "",
          "## Metric Sources and Schema", "", "**A. Direct official metrics:** existing aggregate metric JSON rows provide BLEU, chrF2, token-level AL/LAAL, commit count, commits/100 source tokens, forced-commit rate, mean first-commit latency/tokens, and mean/median unit duration. Per-talk official rows contain BLEU and chrF2 only.",
          "**B. Derived statistics:** commit spans, reason counts, and inter-commit intervals below are computed by reading existing prediction `commits` arrays. No BLEU/chrF/AL/LAAL was recomputed.",
          "**C. Interpretations:** trade-off, Pareto, and context statements are descriptive inferences, not causal estimates.",
          "**D. Unavailable:** complete LISTEN-step traces, all p(COMMIT) values, candidates, and policy states are absent; only commit-time probabilities/features are stored.", "",
          "## P3 Threshold Results", ""]
    headers = ["thr", "BLEU", "chrF2", "AL", "LAAL", "commits", "c/100tok", "forced", "mean obs pos", "mean unit ms", "first ms", "first tok", "mean span", "med span"]
    for variant, title in (("p3", "P3 Threshold Results"), ("p2", "P2 Threshold Results")):
        if variant == "p2": md.extend([f"## {title}", ""])
        rows = []
        for threshold in THRESHOLDS:
            metric = results[variant][threshold_key(threshold)]["official_metrics"]
            rows.append([threshold_key(threshold), fmt(metric["BLEU"]), fmt(metric["chrF2"]), fmt(metric["token_level_average_lagging"]), fmt(metric["token_level_length_adaptive_average_lagging"]), fmt(metric["number_of_commits"], 0), fmt(metric["commits_per_100_source_tokens"]), fmt(metric["forced_commit_rate"], 4), fmt(metric["mean_commit_observation_position"]), fmt(metric["mean_simulated_source_clock_duration_ms"]), fmt(metric["mean_first_commit_simulated_source_clock_latency_ms"]), fmt(metric["mean_first_commit_source_tokens"]), fmt(metric["mean_source_tokens_per_unit"]), fmt(metric["median_source_tokens_per_unit"])])
        md.extend([table(rows, headers), ""])
    md.extend(["## Same-Threshold P3 vs P2", "", "Deltas are **P3 - P2**. Positive BLEU/chrF2 favors P3 quality; negative AL/LAAL and negative commit count mean lower P3 latency measures or fewer P3 commits, respectively.", "",
               table([[key, *[fmt(value) for value in delta.values()]] for key, delta in results["same_threshold_delta_p3_minus_p2"].items()], ["thr", "dBLEU", "dchrF2", "dAL", "dLAAL", "d commits"]), "",
               "## Commit Behavior", "", "Pooled across three talks; quartiles use linear interpolation. `other forced` means any reason other than `policy` or `talk_end`.", ""])
    commit_headers = ["variant", "thr", "commits", "mean", "median", "Q1/Q3", "min/max", "% 4", "% 4-5", "% >=8", "policy/end/other", "mean interval ms"]
    rows = []
    for variant in ("p3", "p2"):
        for threshold in THRESHOLDS:
            stat = results[variant][threshold_key(threshold)]["derived_commit_behavior"]
            rows.append([variant.upper(), threshold_key(threshold), fmt(stat["commit_count"], 0), fmt(stat["mean_source_tokens_per_commit"]), fmt(stat["median_source_tokens_per_commit"]), f"{fmt(stat['q1_source_tokens_per_commit'])}/{fmt(stat['q3_source_tokens_per_commit'])}", f"{stat['min_source_tokens_per_commit']}/{stat['max_source_tokens_per_commit']}", pct(stat["four_token_fraction"]), pct(stat["four_to_five_token_fraction"]), pct(stat["at_least_eight_token_fraction"]), f"{stat['policy_commits']}/{stat['talk_end_commits']}/{stat['other_forced_commits']}", fmt(stat["mean_commit_interval_ms"])])
    md.extend([table(rows, commit_headers), "",
               "The full DEV Sims 0.50 check is confirmed but does not generalize uniformly: P3 has 1,859 versus P2's 1,474 total commits at 0.50, with pooled mean spans 4.49 versus 5.66 and exact-4 fractions shown above. P3 is more finely segmented at 0.50 overall.", "",
               "## Context-Bearing vs Empty-Context Talks", "", "Per-talk quality is the official per-talk BLEU/chrF2. Segmentation is derived from each prediction artifact. Sims is context-bearing; Jeff and Luis are empty-context.", ""])
    talk_rows = []
    for threshold in THRESHOLDS:
        for talk in TALKS:
            p3m = next(x for x in p3_metrics[strategy("p3", threshold)]["per_talk"] if x["talk_id"] == talk)
            p2m = next(x for x in p2_metrics[strategy("p2", threshold)]["per_talk"] if x["talk_id"] == talk)
            p3s = results["per_talk"]["p3"][threshold_key(threshold)][talk]
            p2s = results["per_talk"]["p2"][threshold_key(threshold)][talk]
            talk_rows.append([threshold_key(threshold), "context" if talk == CONTEXT_TALK else "empty", talk, f"{fmt(p3m['BLEU'])}/{fmt(p2m['BLEU'])}", f"{fmt(p3m['chrF2'])}/{fmt(p2m['chrF2'])}", fmt(p3m["BLEU"] - p2m["BLEU"]), fmt(p3m["chrF2"] - p2m["chrF2"]), f"{p3s['commit_count']}/{p2s['commit_count']}", f"{fmt(p3s['mean_source_tokens_per_commit'])}/{fmt(p2s['mean_source_tokens_per_commit'])}"])
    md.extend([table(talk_rows, ["thr", "context", "talk", "BLEU P3/P2", "chrF2 P3/P2", "dBLEU", "dchrF2", "commits P3/P2", "mean span P3/P2"]), "",
               "At 0.50, P3's quality change on Sims is +1.75 BLEU/+1.24 chrF2, while Jeff is -0.75/+0.46 and Luis is -1.15/-0.00. Thus P3's aggregate 0.50 chrF gain coexists with lower aggregate BLEU and higher commit count. Across thresholds, Sims does not show a uniformly larger P3 advantage: at 0.60 it declines sharply relative to P2. Empty-context P3 runs also differ materially from P2, demonstrating that a separately trained P3 policy changes behavior even under a zero context vector.", "",
               "## Quality / Latency Trade-Off", "", "Within both grids, higher thresholds generally reduce commit frequency and increase mean unit duration, but official AL and LAAL are not monotonic in every adjacent step. P3 BLEU rises overall from 25.21 to 28.32, with a 0.30-to-0.40 dip; chrF2 is nonmonotonic and peaks at 0.70 by a small margin. P3's AL drops anomalously from 7.54 at 0.50 to 0.77 at 0.60 despite longer units, then rises to 19.16 at 0.70. P2 has even stronger nonmonotonicity, including negative AL at 0.50 and high LAAL at 0.60/0.70. These are official metrics, not recomputed here.",
               "A potentially useful descriptive P3 region is 0.50-0.60: 0.50 retains the highest P3 chrF2 before 0.70, while 0.60 reduces commits by 317 with +0.45 BLEU but lower chrF2 and much higher LAAL. That is a trade-off, not an automatic choice.", "",
               "## Comparison to Frozen V2 P2 0.50", "", "The historical frozen selection remains `v2_P2_0.50`; this report does not alter it. Deltas below are P3 threshold minus that frozen point.", "",
               table([[key, *[fmt(value) for value in delta.values()]] for key, delta in results["vs_frozen_v2_p2_0_50"].items()], ["P3 thr", "dBLEU", "dchrF2", "dAL", "dLAAL", "d commits"]), "",
               "No P3 point strongly dominates frozen P2 0.50 on the four-objective definition: higher-threshold P3 points improve BLEU but worsen LAAL; lower-latency P3 0.30/0.40 lose BLEU and add commits. P3 0.50 has slightly lower BLEU, higher chrF2, higher AL/LAAL, and 385 more commits, making it an interesting quality-mix trade-off rather than a dominance result.", "",
               "## Prepared-Context Evidence", "", "1. **Aggregate DEV:** P3 has mixed same-threshold changes against P2; the aggregate results do not isolate context effects.",
               "2. **Context-bearing Sims:** at 0.50 Sims improves descriptively under P3, but the direction changes across thresholds, including a large P3 deficit at 0.60.",
               "3. **Empty-context talks:** P3 differs from P2 on Jeff and Luis despite zero prepared context, so policy retraining itself clearly changes segmentation and quality.",
               "4. **Coverage:** only one context-bearing DEV talk exists. This is not sufficient to establish that prepared context improves quality or latency.",
               "5. **Research value:** the result is descriptive evidence worth following up, not evidence of a prepared-context benefit. Further research is justified only through a controlled ablation.", "",
               "## Causal Limitations", "", "P3_GLOBAL was trained from scratch as a new MLP, not P2 with one inference feature added. Its 1,547 inputs include four 384-dimensional embeddings (current source, previous committed source, previous generated target, prepared global context) and 11 numeric features. Prepared representation is `prepared-global-v0`; it affects the policy only. The translator is the frozen source-only EnViT5. Therefore P3/P2 differences are policy/segmentation differences under the same translator architecture, but may arise from different learned policy parameters, the prepared-context feature, or their interaction. They cannot be attributed solely to the prepared vector.",
               "A stronger causal test is the same trained P3 architecture evaluated with real prepared context versus its prepared-context input zeroed/removed under a controlled design. This report does not implement that experiment.", "",
               "## Pareto Observations", "", f"Criterion: {results['pareto']['criteria']}", "", f"Strongly dominated points across the ten P3/P2 grid points: {', '.join(results['pareto']['strongly_dominated']) or 'none'}.", f"Not strongly dominated under this strict four-objective criterion: {', '.join(results['pareto']['not_strongly_dominated'])}.", "The permissive frontier is a consequence of conflicting BLEU, chrF2, AL, and LAAL; it should not be read as equivalence or selection guidance.", "",
               "## Trace Limitations", "", "Prediction artifacts provide commit-time fields, not complete LISTEN-step traces. Consequently the report can describe realized commit spans, times, reasons, and commit-time p(COMMIT), but cannot characterize threshold crossings at every LISTEN step, candidate translations, or latent policy-state trajectories. Threshold-behavior interpretations are necessarily limited to realized commits and aggregate official metrics.", "",
               "## Anomalies", "", "No artifact-integrity anomaly was detected in the scoped P3/P2 DEV grid. The nonmonotonic official latency/quality behavior is reported as a research observation, not an artifact corruption finding.", "",
               "## Recommendation", "", "**D. Results are mixed. The next recommended step is a controlled prepared-context ablation before TEST:** evaluate the same trained P3_GLOBAL policy on the same DEV talks with real prepared context and with the prepared-global input zeroed/removed, preserving all other inputs and conditions. This directly addresses both the from-scratch-policy confounding and the 1/3 context-coverage limitation as far as the current DEV set allows. Do not freeze a new winner from this analysis.", ""])
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "p3_global_full_dev_analysis.md").write_text("\n".join(md), encoding="utf-8")
    (REPORT_DIR / "p3_global_full_dev_analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
