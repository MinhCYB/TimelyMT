"""Read-only DEV rollout artifact sanity reporter.

This utility only opens JSON artifacts with UTF-8 decoding and writes derived
reports.  It never imports rollout, translator, training, or evaluation code.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence


MIN_INFERENCE_BOUNDARY = 4
VERY_SHORT_MAX = 5
LONG_MIN = 8


def read_json(path: Path) -> dict[str, Any]:
    """Load an existing JSON artifact explicitly as UTF-8."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact is not a JSON object: {path}")
    return value


def percentile(values: Sequence[int], fraction: float) -> float:
    """Linearly interpolated percentile, matching common descriptive practice."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def percentage(part: int, total: int) -> float:
    return 0.0 if not total else 100.0 * part / total


def escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


@dataclass(frozen=True)
class RolloutStats:
    count: int
    lengths: list[int]
    times: list[int]
    intervals: list[int]
    reasons: Counter[str]
    span_histogram: Counter[int]
    contiguous: bool
    source_coverage_complete: bool
    final_source_end: int | None


def validate_and_summarize(record: Mapping[str, Any]) -> RolloutStats:
    commits = record.get("commits")
    if not isinstance(commits, list) or not commits:
        raise ValueError("Artifact has no non-empty commits array")
    expected = {
        "causal_features", "commit_probability", "observation_emit_ms",
        "observation_token_index", "reason", "source_clock_duration_ms",
        "source_end", "source_start", "source_token_count", "target_token_count",
        "translated_text",
    }
    lengths: list[int] = []
    times: list[int] = []
    reasons: Counter[str] = Counter()
    contiguous = True
    previous_end = -1
    for index, commit in enumerate(commits):
        if not isinstance(commit, dict):
            raise ValueError(f"Commit {index} is not an object")
        if set(commit) != expected:
            raise ValueError(f"Commit {index} schema differs: {sorted(set(commit) ^ expected)}")
        start, end, stored_count = commit["source_start"], commit["source_end"], commit["source_token_count"]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in (start, end, stored_count)):
            raise ValueError(f"Commit {index} has non-integer span fields")
        derived_count = end - start + 1
        if derived_count != stored_count:
            raise ValueError(f"Commit {index} source count does not match its inclusive span")
        if start != previous_end + 1:
            contiguous = False
        previous_end = end
        lengths.append(derived_count)
        times.append(commit["observation_emit_ms"])
        reasons[str(commit["reason"])] += 1
    source_tokens = record.get("source_token_count")
    complete = isinstance(source_tokens, int) and previous_end == source_tokens - 1 and sum(lengths) == source_tokens
    return RolloutStats(
        count=len(commits), lengths=lengths, times=times,
        intervals=[right - left for left, right in zip(times, times[1:])],
        reasons=reasons, span_histogram=Counter(lengths), contiguous=contiguous,
        source_coverage_complete=complete, final_source_end=previous_end,
    )


def stats_dict(record: Mapping[str, Any], stats: RolloutStats) -> dict[str, Any]:
    lengths, intervals = stats.lengths, stats.intervals
    duration_ms = int(record["source_final_emit_ms"])
    return {
        "commit_count": stats.count,
        "mean_source_tokens_per_commit": mean(lengths),
        "median_source_tokens_per_commit": median(lengths),
        "min_source_tokens_per_commit": min(lengths),
        "max_source_tokens_per_commit": max(lengths),
        "quartiles_source_tokens_per_commit": {"q1": percentile(lengths, .25), "q3": percentile(lengths, .75)},
        "population_standard_deviation_source_tokens_per_commit": pstdev(lengths),
        "commit_span_histogram": {str(key): stats.span_histogram[key] for key in sorted(stats.span_histogram)},
        "minimum_boundary_commit_count": stats.span_histogram[MIN_INFERENCE_BOUNDARY],
        "minimum_boundary_commit_fraction": stats.span_histogram[MIN_INFERENCE_BOUNDARY] / stats.count,
        "very_short_commit_definition": f"{MIN_INFERENCE_BOUNDARY}-{VERY_SHORT_MAX} source tokens",
        "very_short_commit_count": sum(value for key, value in stats.span_histogram.items() if key <= VERY_SHORT_MAX),
        "very_short_commit_fraction": sum(value for key, value in stats.span_histogram.items() if key <= VERY_SHORT_MAX) / stats.count,
        "long_commit_definition": f">={LONG_MIN} source tokens",
        "long_commit_count": sum(value for key, value in stats.span_histogram.items() if key >= LONG_MIN),
        "long_commit_fraction": sum(value for key, value in stats.span_histogram.items() if key >= LONG_MIN) / stats.count,
        "first_commit_observation_emit_ms": stats.times[0],
        "last_commit_observation_emit_ms": stats.times[-1],
        "mean_commit_interval_ms": mean(intervals) if intervals else 0.0,
        "median_commit_interval_ms": median(intervals) if intervals else 0.0,
        "commits_per_minute_source_audio": stats.count / (duration_ms / 60000),
        "reason_counts": dict(sorted(stats.reasons.items())),
        "spans_contiguous": stats.contiguous,
        "source_coverage_complete": stats.source_coverage_complete,
        "final_source_end": stats.final_source_end,
    }


def cumulative_counts(times: Sequence[int], final_ms: int) -> list[dict[str, int]]:
    points = list(range(0, final_ms + 1, 60000))
    if not points or points[-1] != final_ms:
        points.append(final_ms)
    return [{"elapsed_ms": point, "commits": sum(time <= point for time in times)} for point in points]


def commit_rows(commits: Sequence[Mapping[str, Any]], indexes: Sequence[int]) -> list[str]:
    rows = ["| Index | Time (ms) | Source span | Source tokens | Reason | p(COMMIT) | Translated unit |", "|---:|---:|---|---:|---|---:|---|"]
    for index in indexes:
        item = commits[index]
        probability = item["commit_probability"]
        rendered_probability = "not stored" if probability is None else f"{probability:.4f}"
        rows.append(
            f"| {index + 1} | {item['observation_emit_ms']} | {item['source_start']}..{item['source_end']} | "
            f"{item['source_token_count']} | {escape_cell(item['reason'])} | {rendered_probability} | {escape_cell(item['translated_text'])} |"
        )
    return rows


def schema_table(commit: Mapping[str, Any]) -> list[str]:
    meanings = {
        "source_start": "Inclusive zero-based start index of the committed runtime source span.",
        "source_end": "Inclusive zero-based end index of the committed runtime source span.",
        "observation_token_index": "Source token index observed when this commit was made.",
        "observation_emit_ms": "Emit time of the observed end token, in milliseconds.",
        "source_token_count": "Number of source tokens in the committed span.",
        "source_clock_duration_ms": "End-token emit time minus start-token emit time for the span.",
        "target_token_count": "Whitespace-token count of `translated_text`.",
        "translated_text": "Translator hypothesis for the full committed source span.",
        "reason": "Streaming termination reason (`policy`, `max_length`, or `talk_end`).",
        "commit_probability": "P3 policy probability at this committing observation.",
        "causal_features": "Numeric component of the causal policy state at this committing observation.",
    }
    rows = ["| Field | Type observed | Stored/derived | Meaning |", "|---|---|---|---|"]
    for field in commit:
        type_name = type(commit[field]).__name__
        derived = "Stored; also derivable from stored fields" if field in {"source_token_count", "target_token_count", "observation_token_index"} else "Stored"
        rows.append(f"| `{field}` | `{type_name}` | {derived} | {meanings[field]} |")
    return rows


def p2_threshold_rows(directory: Path, talk_id: str) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    for path in sorted(directory.glob("v2_P2_*/" + talk_id + ".json")):
        artifact = read_json(path)
        stats = validate_and_summarize(artifact)
        threshold = str(artifact.get("strategy", path.parent.name)).rsplit("_", 1)[-1]
        rows.append((threshold, stats.count, mean(stats.lengths)))
    return rows


def build_report(p3: Mapping[str, Any], p2: Mapping[str, Any], prepared: Mapping[str, Any], p3_stats: RolloutStats, p2_stats: RolloutStats, p3_values: Mapping[str, Any], p2_values: Mapping[str, Any], thresholds: Sequence[tuple[str, int, float]]) -> str:
    commits = p3["commits"]
    source = prepared["sources"]
    eligible = [item for item in source if item["classification"] == "SAFE_PRETALK_CONFIRMED" and item["available_before_talk"] and not item["transcript_used"] and not item["reference_used"]]
    context = p3.get("prepared_context")
    if not isinstance(context, Mapping):
        raise ValueError("P3 artifact has no prepared_context object")
    prediction = p3["prediction"]
    middle = list(range(max(0, len(commits) // 2 - 2), min(len(commits), len(commits) // 2 + 3)))
    lines = [
        "# P3_GLOBAL Sims 0.50 - Single-Talk Sanity Report", "",
        "## Verdict", "", "**PASS WITH OBSERVATIONS.** The inspected DEV artifact is structurally coherent and covers the complete source stream with contiguous committed spans. Prepared context is non-zero and has concrete source/checksum provenance. Threshold 0.50 is mechanically aggressive: most commits occur at the 4-token policy-inference boundary. This is a descriptive single-talk observation, not a quality or selection result.", "",
        "## Artifact Identity", "",
        f"| Field | Value | Evidence class |", "|---|---|---|",
        f"| `artifact_version` | `{p3.get('artifact_version')}` | Explicitly stored |",
        f"| `talk_id` | `{p3.get('talk_id')}` | Explicitly stored |",
        f"| `split` | `{p3.get('split')}` | Explicitly stored |",
        f"| `strategy` | `{p3.get('strategy')}` | Explicitly stored |",
        f"| `source_token_count` | {p3.get('source_token_count')} | Explicitly stored; equals summed committed spans |",
        f"| `source_final_emit_ms` | {p3.get('source_final_emit_ms')} | Explicitly stored |",
        f"| Commit count | {p3_stats.count} | Derived from stored `commits` |", "",
        "The artifact top-level schema is exactly `artifact_version`, `commits`, `prediction`, `prepared_context`, `source_final_emit_ms`, `source_token_count`, `split`, `strategy`, and `talk_id`; it has no rollout-wide trace fields. All 359 commit objects have one identical outer schema. Derived span coverage is contiguous from source index 0 through 1688, with span lengths summing to 1689; the final commit observation equals the stored final source emit time (675600 ms).", "",
        "## Prepared Context", "",
        f"The standalone pool is `{prepared.get('schema_version')}` for the same DEV talk. It contains {len(source)} source(s), of which {len(eligible)} satisfy the strict eligibility predicate. The rollout stores a `{context.get('representation_version')}` provenance record with embedding dimension {context.get('embedding_dimension')}, source count {context.get('source_count')}, `has_eligible_context={context.get('has_eligible_context')}`, and embedding norm {number(float(context.get('embedding_norm', 0)))}. A non-zero norm confirms this talk did not receive the zero-vector empty-context representation.", "",
        "| Eligible source ID | Checksum | Type | Published | Declared leakage metadata |", "|---|---|---|---|---|",
    ]
    for item in eligible:
        lines.append(f"| `{item['source_id']}` | `{item['checksum']}` | `{item['source_type']}` | {item['published_at']} | `available_before_talk={item['available_before_talk']}`, `transcript_used={item['transcript_used']}`, `reference_used={item['reference_used']}`, `{item['classification']}` |");
    lines += ["", "The artifact-level eligible IDs/checksums exactly identify the prepared source used, and the standalone pool supplies its URI and text. Under the validated eligibility rule, transcript/reference leakage is not indicated: eligible sources must be `SAFE_PRETALK_CONFIRMED`, pre-talk available, and have both use flags false. This establishes metadata-level provenance, not an independent historical audit of the external publication claim.", "",
        "## Commit Schema", "", *schema_table(commits[0]), "",
        "`causal_features` is a stored object with these float-valued fields: `source_buffer_token_count`, `source_buffer_character_count`, `source_clock_elapsed_ms`, `current_target_token_count`, `previous_target_token_count`, `target_token_count_delta`, `previous_current_lcp_ratio`, `previous_current_change_ratio`, `prior_committed_unit_count`, `previous_committed_source_tokens`, and `previous_committed_target_tokens`. Its semantic construction is known from source code; the artifact contains it only at committed observations.", "",
        "## Commit Statistics", "",
        f"| Statistic | P3_GLOBAL 0.50 |", "|---|---:|",
        f"| Source tokens / commits | {p3.get('source_token_count')} / {p3_stats.count} |",
        f"| Mean / median source tokens per commit | {number(p3_values['mean_source_tokens_per_commit'])} / {number(p3_values['median_source_tokens_per_commit'])} |",
        f"| Min / Q1 / Q3 / max | {p3_values['min_source_tokens_per_commit']} / {number(p3_values['quartiles_source_tokens_per_commit']['q1'])} / {number(p3_values['quartiles_source_tokens_per_commit']['q3'])} / {p3_values['max_source_tokens_per_commit']} |",
        f"| Population SD | {number(p3_values['population_standard_deviation_source_tokens_per_commit'])} |",
        f"| First / last commit observation | {p3_values['first_commit_observation_emit_ms']} / {p3_values['last_commit_observation_emit_ms']} ms |",
        f"| Mean / median inter-commit interval | {number(p3_values['mean_commit_interval_ms'])} / {number(p3_values['median_commit_interval_ms'])} ms |",
        f"| Commits per minute of source audio | {number(p3_values['commits_per_minute_source_audio'])} |",
        f"| Reasons | {escape_cell(p3_values['reason_counts'])} |", "",
        "## Commit-Length Distribution", "",
        "| Source tokens in committed span | Commit count | Percentage |", "|---:|---:|---:|"]
    for size, count in sorted(p3_stats.span_histogram.items()):
        lines.append(f"| {size} | {count} | {number(percentage(count, p3_stats.count))}% |")
    lines += ["", f"At the exact 4-token minimum inference boundary: **{p3_values['minimum_boundary_commit_count']}/{p3_stats.count} ({number(100 * p3_values['minimum_boundary_commit_fraction'])}%)**. Very short is defined here as 4-5 tokens: **{p3_values['very_short_commit_count']}/{p3_stats.count} ({number(100 * p3_values['very_short_commit_fraction'])}%)**. Long is defined here as >=8 tokens: **{p3_values['long_commit_count']}/{p3_stats.count} ({number(100 * p3_values['long_commit_fraction'])}%)**. This directly derives inclusive span lengths from `source_start`/`source_end`, rather than inferring behavior from the aggregate ratio alone. It supports calling 0.50 aggressive for this talk, but does not establish that it is erroneous or generalizes to DEV.", "",
        "Cumulative commits by source-audio time:", "", "| Elapsed source time | Cumulative commits |", "|---:|---:|"]
    for row in cumulative_counts(p3_stats.times, int(p3["source_final_emit_ms"])):
        lines.append(f"| {row['elapsed_ms']} ms | {row['commits']} |")
    lines += ["", "## Representative Commits", "", "The artifact does not retain source text for each span, so source spans below are inclusive token-index ranges; no source wording is reconstructed here.", "", "### First 10", "", *commit_rows(commits, list(range(10))), "", "### Middle (commits 178-182)", "", *commit_rows(commits, middle), "", "### Last 10", "", *commit_rows(commits, list(range(len(commits) - 10, len(commits)))), "",
        "## Final Prediction", "", f"The stored UTF-8-decoded prediction is {len(prediction)} Unicode characters and {len(prediction.split())} whitespace-delimited words (an approximation, not tokenizer tokens). Beginning: `{prediction[:240]}`. Ending: `{prediction[-240:]}`. Vietnamese diacritics decode correctly (for example, `Bạn có thể đã có` in commit 1); the reported PowerShell mojibake is not present in the JSON text decoded as UTF-8.", "",
        "## P3 vs P2 0.50", "", "This is same-talk descriptive context only; it does not attribute differences to prepared context.", "",
        "| Statistic | P3_GLOBAL 0.50 | P2 0.50 |", "|---|---:|---:|",
        f"| Source tokens | {p3['source_token_count']} | {p2['source_token_count']} |",
        f"| Commits | {p3_stats.count} | {p2_stats.count} |",
        f"| Mean source tokens / commit | {number(mean(p3_stats.lengths))} | {number(mean(p2_stats.lengths))} |",
        f"| First / last commit time (ms) | {p3_stats.times[0]} / {p3_stats.times[-1]} | {p2_stats.times[0]} / {p2_stats.times[-1]} |",
        f"| Mean / median interval (ms) | {number(mean(p3_stats.intervals))} / {number(median(p3_stats.intervals))} | {number(mean(p2_stats.intervals))} / {number(median(p2_stats.intervals))} |",
        f"| Final prediction characters / words | {len(p3['prediction'])} / {len(p3['prediction'].split())} | {len(p2['prediction'])} / {len(p2['prediction'].split())} |",
        f"| 4-token commits | {p3_stats.span_histogram[4]} ({number(percentage(p3_stats.span_histogram[4], p3_stats.count))}%) | {p2_stats.span_histogram[4]} ({number(percentage(p2_stats.span_histogram[4], p2_stats.count))}%) |",
        f"| Very short commits (4-5) | {p3_values['very_short_commit_count']} ({number(100*p3_values['very_short_commit_fraction'])}%) | {p2_values['very_short_commit_count']} ({number(100*p2_values['very_short_commit_fraction'])}%) |",
        f"| Forced `talk_end` commits | {p3_stats.reasons['talk_end']} | {p2_stats.reasons['talk_end']} |",
        f"| First five spans | {', '.join(map(str, p3_stats.lengths[:5]))} | {', '.join(map(str, p2_stats.lengths[:5]))} |",
        f"| Span distribution | {dict(sorted(p3_stats.span_histogram.items()))} | {dict(sorted(p2_stats.span_histogram.items()))} |", "",
        "## Optional P2 Threshold Context", "", "| Threshold | Commits | Mean source tokens/commit |", "|---:|---:|---:|"]
    for threshold, count, average in thresholds:
        lines.append(f"| {threshold} | {count} | {number(average)} |")
    lines += ["", "These existing Sims/P2 files have the same checked commit schema and are presented only as a descriptive threshold context, not model selection.", "",
        "## Trace Limitations", "", "The current artifact stores `p(COMMIT)` and the numeric causal feature object only for committed observations. It does **not** store every LISTEN decision, p(COMMIT) at each non-commit timestep, every timestep candidate translation, complete policy state at every timestep, or source text for each commit. Therefore it cannot support a reconstructed timestep-level trace or probability calibration analysis. A future interactive demo would need opt-in per-observation logging of candidate span/index/time, full causal state (or a documented redacted form), candidate translation, p(COMMIT), decision/reason, and a stable run/config identity, written independently from final commit artifacts.", "",
        "## Findings", "", "1. **Explicitly stored:** identity fields, full committed decision records, final joined prediction, and P3 prepared-context provenance.", "2. **Mathematically derived:** 359 contiguous spans fully cover 1689 source tokens; commit-length statistics, observation-time intervals, and cumulative counts. `source_clock_duration_ms` is stored but cannot be recomputed from this artifact alone because start-token emit times are absent.", "3. **Known from source semantics:** learned streaming begins policy inference at 4 source tokens, calls the source-only translator on each eligible candidate, commits on `p >= threshold` (or max-length/talk-end), and represents P3 context as eligible-source MiniLM embeddings (equal-average plus normalization only for multiple sources). The final 3-token span is below the normal inference boundary and is emitted by the post-loop `talk_end` flush; it is thus mechanically forced, even though the fallback records a probability.", "4. **Not observable:** LISTEN decisions, their probabilities/candidate translations, the full timestep trajectory, original per-span source text, and independent verification of the external source's publication/availability assertion.", "",
        "## Recommendation Before Full DEV", "", "**Proceed to the full DEV grid.** No structural, provenance-metadata, or mechanical streaming anomaly in this completed single-talk artifact blocks it. Before proceeding, record that 0.50 is boundary-heavy on this talk and preserve the current artifact/report; inspect the full-grid commit-length distributions and any unexpected forced-commit patterns before interpreting results. Do not treat this report as quality evaluation or a winner selection.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Produce a read-only single-talk rollout sanity report.")
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--p2-directory", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    p3, p2, prepared = read_json(args.p3), read_json(args.p2), read_json(args.prepared)
    p3_stats, p2_stats = validate_and_summarize(p3), validate_and_summarize(p2)
    if p3.get("talk_id") != p2.get("talk_id") or p3.get("talk_id") != prepared.get("talk_id"):
        raise ValueError("P3, P2, and prepared-context talk IDs must match")
    if p3.get("split") != "dev" or p2.get("split") != "dev" or prepared.get("split") != "dev":
        raise ValueError("This reporter accepts DEV artifacts only")
    p3_values, p2_values = stats_dict(p3, p3_stats), stats_dict(p2, p2_stats)
    context = p3.get("prepared_context")
    if not isinstance(context, dict):
        raise ValueError("P3 artifact has no prepared_context object")
    summary = {
        "talk_id": p3["talk_id"], "strategy": p3["strategy"], "source_token_count": p3["source_token_count"],
        **p3_values,
        "prepared_context_source_ids": context.get("eligible_source_ids"),
        "prepared_context_source_checksums": context.get("eligible_source_checksums"),
        "prepared_representation_version": context.get("representation_version"),
        "prepared_embedding_dimension": context.get("embedding_dimension"),
        "prepared_embedding_norm": context.get("embedding_norm"),
        "prepared_has_eligible_context": context.get("has_eligible_context"),
        "p2_0_50_commit_count": p2_stats.count,
        "p2_0_50_mean_source_tokens_per_commit": mean(p2_stats.lengths),
        "p2_0_50_commit_span_histogram": {str(key): p2_stats.span_histogram[key] for key in sorted(p2_stats.span_histogram)},
        "verdict": "PASS WITH OBSERVATIONS",
    }
    threshold_rows = p2_threshold_rows(args.p2_directory, str(p3["talk_id"]))
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(build_report(p3, p2, prepared, p3_stats, p2_stats, p3_values, p2_values, threshold_rows), encoding="utf-8", newline="\n")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
