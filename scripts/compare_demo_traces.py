"""Read-only REAL-versus-ZERO comparison for validated demo policy traces.

This utility only reads JSON trace artifacts.  It never imports model code,
loads checkpoints, runs rollouts, or accesses TEST data.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SYNC_KEYS = ("talk_id", "threshold", "checkpoint_sha256", "source_token_count", "source_final_emit_ms")


def read_trace(path: Path) -> dict[str, Any]:
    """Read one UTF-8 demo trace without invoking any inference code."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_synchronized(real: dict[str, Any], zero: dict[str, Any]) -> None:
    """Reject comparison unless both artifacts share an exact source timeline."""
    for key in SYNC_KEYS:
        if real.get(key) != zero.get(key):
            raise ValueError(f"trace mismatch for {key}: {real.get(key)!r} != {zero.get(key)!r}")
    real_events, zero_events = real.get("events"), zero.get("events")
    if not isinstance(real_events, list) or not isinstance(zero_events, list):
        raise ValueError("both traces must contain event lists")
    if len(real_events) != len(zero_events):
        raise ValueError(f"event count mismatch: {len(real_events)} != {len(zero_events)}")
    for index, (real_event, zero_event) in enumerate(zip(real_events, zero_events)):
        for key in ("event_index", "source_token_end", "observation_ms"):
            if real_event.get(key) != zero_event.get(key):
                raise ValueError(
                    f"source timeline mismatch at pair index {index} for {key}: "
                    f"{real_event.get(key)!r} != {zero_event.get(key)!r}"
                )


def _same_candidate(real: dict[str, Any], zero: dict[str, Any]) -> bool:
    return all(real.get(key) == zero.get(key) for key in (
        "candidate_source_start", "candidate_source_end", "candidate_translation",
    ))


def _same_commit(real: dict[str, Any], zero: dict[str, Any]) -> bool:
    return _same_candidate(real, zero) and all(real.get(key) == zero.get(key) for key in (
        "committed_source_text", "committed_target_text", "committed_unit_index",
    ))


def classify(real: dict[str, Any], zero: dict[str, Any]) -> str:
    """Classify a synchronized event without treating WAIT nulls as scores."""
    real_decision, zero_decision = real["decision"], zero["decision"]
    if real_decision == zero_decision == "WAIT" and _same_candidate(real, zero):
        return "SAME_WAIT"
    if real_decision == zero_decision == "LISTEN" and _same_candidate(real, zero):
        return "SAME_LISTEN"
    if real_decision == zero_decision == "COMMIT" and _same_commit(real, zero):
        return "SAME_COMMIT"
    if real_decision == "COMMIT" and zero_decision == "LISTEN":
        return "REAL_COMMIT_ZERO_LISTEN"
    if real_decision == "LISTEN" and zero_decision == "COMMIT":
        return "REAL_LISTEN_ZERO_COMMIT"
    if real_decision == zero_decision == "COMMIT":
        return "DIFFERENT_COMMIT_BOUNDARY"
    return "OTHER_DIFFERENCE"


def _regions(indices: list[int]) -> list[dict[str, int]]:
    if not indices:
        return []
    regions: list[dict[str, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            regions.append({"start_event_index": start, "end_event_index": previous, "length": previous - start + 1})
            start = index
        previous = index
    regions.append({"start_event_index": start, "end_event_index": previous, "length": previous - start + 1})
    return regions


def _event_record(real: dict[str, Any], zero: dict[str, Any], relation: str) -> dict[str, Any]:
    delta = None if real.get("p_commit") is None or zero.get("p_commit") is None else real["p_commit"] - zero["p_commit"]
    return {
        "event_index": real["event_index"], "observation_ms": real["observation_ms"],
        "source_token_end": real["source_token_end"], "relation": relation,
        "delta_p_commit": delta, "absolute_delta_p_commit": None if delta is None else abs(delta),
        "real": {key: real.get(key) for key in (
            "decision", "decision_reason", "p_commit", "candidate_source_start", "candidate_source_end",
            "candidate_source_text", "candidate_translation", "committed_target_text",
        )},
        "zero": {key: zero.get(key) for key in (
            "decision", "decision_reason", "p_commit", "candidate_source_start", "candidate_source_end",
            "candidate_source_text", "candidate_translation", "committed_target_text",
        )},
    }


def _commit_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    commits = [event for event in events if event["decision"] == "COMMIT"]
    times = [event["observation_ms"] for event in commits]
    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    return {
        "total_commits": len(commits), "first_commit_ms": times[0] if times else None,
        "last_commit_ms": times[-1] if times else None,
        "mean_commit_interval_ms": statistics.mean(intervals) if intervals else None,
    }


def _mmss(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _top_moments(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    choices = [record for record in records if record["real"]["decision"] != record["zero"]["decision"]]
    def score(record: dict[str, Any]) -> tuple[float, float, int]:
        real_probability, zero_probability = record["real"]["p_commit"], record["zero"]["p_commit"]
        straddles = real_probability is not None and zero_probability is not None and (real_probability >= threshold) != (zero_probability >= threshold)
        readable = min(len(record["real"]["candidate_translation"] or ""), len(record["zero"]["candidate_translation"] or ""))
        return (2.0 if straddles else 0.0, record["absolute_delta_p_commit"] or 0.0, readable)
    return sorted(choices, key=score, reverse=True)[:8]


def _bookmark(record: dict[str, Any], identifier: str, label: str, description: str) -> dict[str, Any]:
    return {"id": identifier, "event_index": record["event_index"], "observation_ms": record["observation_ms"], "label": label, "description": description}


def analyze(real: dict[str, Any], zero: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    require_synchronized(real, zero)
    records = [_event_record(real_event, zero_event, classify(real_event, zero_event)) for real_event, zero_event in zip(real["events"], zero["events"])]
    decision_differences = [record for record in records if record["real"]["decision"] != record["zero"]["decision"]]
    probability_records = [record for record in records if record["absolute_delta_p_commit"] is not None]
    absolute_deltas = [record["absolute_delta_p_commit"] for record in probability_records]
    regions = _regions([record["event_index"] for record in decision_differences])
    max_probability_record = max(probability_records, key=lambda record: record["absolute_delta_p_commit"], default=None)
    max_divergence_record = max(decision_differences, key=lambda record: record["absolute_delta_p_commit"] or -1, default=None)
    first = decision_differences[0] if decision_differences else None
    last = decision_differences[-1] if decision_differences else None
    same_span_policy_divergences = [record for record in decision_differences if (
        record["real"]["candidate_source_start"] == record["zero"]["candidate_source_start"]
        and record["real"]["candidate_source_end"] == record["zero"]["candidate_source_end"]
    )]
    boundary_divergences = [record for record in records if (
        record["real"]["candidate_source_start"] != record["zero"]["candidate_source_start"]
    )]
    translation_boundary_divergences = [record for record in boundary_divergences if (
        record["real"]["candidate_translation"] != record["zero"]["candidate_translation"]
    )]
    relation_counts = Counter(record["relation"] for record in records)
    first_real_commit = next((record for record in records if record["relation"] == "REAL_COMMIT_ZERO_LISTEN"), None)
    first_zero_commit = next((record for record in records if record["relation"] == "REAL_LISTEN_ZERO_COMMIT"), None)
    first_boundary = next((record for record in records if record["relation"] == "DIFFERENT_COMMIT_BOUNDARY"), None)
    resync = next((record for record in records[(first["event_index"] + 1 if first else 0):] if record["relation"] in {"SAME_COMMIT", "SAME_LISTEN"}), None)
    moments = _top_moments(records, real["threshold"])
    report = {
        "talk_id": real["talk_id"], "threshold": real["threshold"], "pair_integrity": {
            "synchronized": True, **{key: real[key] for key in SYNC_KEYS}, "event_count": len(records),
            "source_timeline_fields": ["event_index", "source_token_end", "observation_ms"],
        },
        "summary": {"identical_decisions": len(records) - len(decision_differences), "differing_decisions": len(decision_differences)},
        "divergence_statistics": {
            "relation_counts": dict(sorted(relation_counts.items())), "real_commit_zero_listen": relation_counts["REAL_COMMIT_ZERO_LISTEN"],
            "real_listen_zero_commit": relation_counts["REAL_LISTEN_ZERO_COMMIT"], "first_divergence": first,
            "last_divergence": last, "divergence_regions": regions,
            "longest_divergence_region": max(regions, key=lambda region: region["length"], default=None),
        },
        "probability_differences": {
            "paired_probability_events": len(probability_records), "mean_absolute_delta": statistics.mean(absolute_deltas) if absolute_deltas else None,
            "median_absolute_delta": statistics.median(absolute_deltas) if absolute_deltas else None,
            "max_absolute_delta": max(absolute_deltas) if absolute_deltas else None,
            "max_absolute_delta_event": max_probability_record,
            "differing_decision_events": [{key: record[key] for key in ("event_index", "relation", "delta_p_commit", "absolute_delta_p_commit")} for record in decision_differences],
            "max_differing_decision_delta_event": max_divergence_record,
        },
        "commit_timeline": {"real": _commit_summary(real["events"]), "zero": _commit_summary(zero["events"]),
            "first_real_commit_zero_listen": first_real_commit, "first_real_listen_zero_commit": first_zero_commit,
            "first_different_commit_boundary": first_boundary, "first_later_same_non_wait_event": resync},
        "candidate_translation_cascades": {
            "same_span_policy_divergence_count": len(same_span_policy_divergences), "first_same_span_policy_divergence": same_span_policy_divergences[0] if same_span_policy_divergences else None,
            "different_candidate_boundary_count": len(boundary_divergences), "first_different_candidate_boundary": boundary_divergences[0] if boundary_divergences else None,
            "different_translation_after_boundary_count": len(translation_boundary_divergences),
            "first_translation_difference_after_boundary": translation_boundary_divergences[0] if translation_boundary_divergences else None,
        },
        "top_demo_moments": moments,
        "interpretation_guardrails": [
            "REAL and ZERO use the same P3 checkpoint.", "Prepared context affects the policy only.",
            "A different commit can alter future candidate spans, causing later candidate translations to differ indirectly.",
            "Candidate-translation differences after boundary divergence are downstream consequences, not direct context injection into EnViT5.",
            "This trace is an illustrative controlled DEV case, not general evidence of superiority.",
        ],
    }
    bookmark_records = [(first, "first-divergence", "First policy divergence", "First synchronized event with different REAL and ZERO policy decisions."),
                        (max_divergence_record, "max-probability-gap", "Largest decision-divergence probability gap", "Differing decision with the largest absolute REAL-minus-ZERO commit probability gap."),
                        (first_real_commit, "real-commit-zero-listen", "REAL commits while ZERO listens", "Clear threshold-straddling policy difference."),
                        (first_zero_commit, "real-listen-zero-commit", "ZERO commits while REAL listens", "Clear threshold-straddling policy difference."),
                        (first_boundary, "commit-boundary-cascade", "Downstream commit-boundary cascade", "Both policies commit but their earlier choices yield different candidate boundaries.")]
    bookmarks = []
    for record, identifier, label, description in bookmark_records:
        if record is not None:
            bookmarks.append(_bookmark(record, identifier, label, description))
    late = next((record for record in reversed(decision_differences) if not real["events"][record["event_index"]].get("is_forced")), None)
    if late is not None:
        bookmarks.append(_bookmark(late, "late-talk-divergence", "Late-talk policy divergence", "Late non-forced policy difference, useful for showing the effect persists beyond the opening."))
    return report, {"talk_id": real["talk_id"], "threshold": real["threshold"], "bookmarks": bookmarks}


def markdown(report: dict[str, Any], bookmarks: dict[str, Any]) -> str:
    integrity, stats, probabilities = report["pair_integrity"], report["divergence_statistics"], report["probability_differences"]
    total, same = integrity["event_count"], report["summary"]["identical_decisions"]
    percent = lambda value: f"{100 * value / total:.2f}%"
    record_lines = lambda record: "None" if record is None else f"event {record['event_index']} at {_mmss(record['observation_ms'])} ({record['observation_ms']} ms)"
    lines = ["# Sims 0.60 REAL vs ZERO Trace Analysis", "", "## Trace Pair Integrity", "",
        f"Synchronized: **yes**. DEV talk `{integrity['talk_id']}`; threshold `{integrity['threshold']:.2f}`; checkpoint SHA `{integrity['checkpoint_sha256']}`; source tokens `{integrity['source_token_count']}`; final source emission `{integrity['source_final_emit_ms']}` ms; events `{total}`.",
        "Exact equality was required for event count and every `event_index`, `source_token_end`, and `observation_ms`; no heuristic alignment was used.", "", "## Summary", "",
        f"Identical decisions: **{same}/{total} ({percent(same)})**. Differing decisions: **{total - same}/{total} ({percent(total - same)})**.", "", "## Divergence Statistics", "",
        f"REAL COMMIT / ZERO LISTEN: **{stats['real_commit_zero_listen']}**. REAL LISTEN / ZERO COMMIT: **{stats['real_listen_zero_commit']}**.",
        f"First divergence: {record_lines(stats['first_divergence'])}. Last divergence: {record_lines(stats['last_divergence'])}.",
        f"Divergence regions: **{len(stats['divergence_regions'])}**. Longest: `{stats['longest_divergence_region']}`.", "", "## First Divergence", ""]
    first = stats["first_divergence"]
    if first:
        lines += [f"Event `{first['event_index']}` at `{first['observation_ms']}` ms ({_mmss(first['observation_ms'])}), source token end `{first['source_token_end']}`, threshold `{report['threshold']:.2f}`, delta p_commit `{first['delta_p_commit']:+.6f}`.",
            f"REAL: span `{first['real']['candidate_source_start']}..{first['real']['candidate_source_end']}`: “{first['real']['candidate_source_text']}”; translation: “{first['real']['candidate_translation']}”; p_commit `{first['real']['p_commit']:.6f}`; `{first['real']['decision']}` because `{first['real']['decision_reason']}`.",
            f"ZERO: span `{first['zero']['candidate_source_start']}..{first['zero']['candidate_source_end']}`: “{first['zero']['candidate_source_text']}”; translation: “{first['zero']['candidate_translation']}”; p_commit `{first['zero']['p_commit']:.6f}`; `{first['zero']['decision']}` because `{first['zero']['decision_reason']}`.",
            "This is a useful demo moment because both policies observe the same source clock and candidate span, while their commit probabilities fall on opposite sides of the policy threshold. It isolates a policy timing difference; it does not establish translation improvement.", ""]
    lines += ["## Top Demo Moments", ""]
    for rank, record in enumerate(report["top_demo_moments"], 1):
        lines.append(f"{rank}. Event `{record['event_index']}` at `{_mmss(record['observation_ms'])}`; “{record['real']['candidate_source_text']}”; REAL `{record['real']['p_commit']:.6f}` / `{record['real']['decision']}`, ZERO `{record['zero']['p_commit']:.6f}` / `{record['zero']['decision']}`, delta `{record['delta_p_commit']:+.6f}`. `{record['relation']}` with a threshold-straddling, readable candidate.")
    timeline = report["commit_timeline"]
    lines += ["", "## Commit Timeline", "", f"REAL: `{timeline['real']}`.", f"ZERO: `{timeline['zero']}`.", f"First REAL-first point: {record_lines(timeline['first_real_commit_zero_listen'])}. First ZERO-first point: {record_lines(timeline['first_real_listen_zero_commit'])}. First different commit boundary: {record_lines(timeline['first_different_commit_boundary'])}. Later same non-WAIT event: {record_lines(timeline['first_later_same_non_wait_event'])}.", "", "## Probability Differences", "", f"Paired probability events: `{probabilities['paired_probability_events']}`. Mean absolute delta: `{probabilities['mean_absolute_delta']:.6f}`. Median: `{probabilities['median_absolute_delta']:.6f}`. Maximum: `{probabilities['max_absolute_delta']:.6f}` at {record_lines(probabilities['max_absolute_delta_event'])}. WAIT events remain null and are excluded.", "", "## Candidate Translation Cascades", ""]
    cascade = report["candidate_translation_cascades"]
    lines += [f"A. Same-span probability/decision divergences: `{cascade['same_span_policy_divergence_count']}`; first: {record_lines(cascade['first_same_span_policy_divergence'])}.", f"B. Different candidate boundaries after earlier commits: `{cascade['different_candidate_boundary_count']}` events; translations also differ at `{cascade['different_translation_after_boundary_count']}` of them; first translation difference: {record_lines(cascade['first_translation_difference_after_boundary'])}.", "These later translation differences are downstream effects of altered source spans, not direct prepared-context input to EnViT5.", "", "## Demo Bookmarks", ""]
    lines += [f"- `{bookmark['id']}`: event `{bookmark['event_index']}` at `{_mmss(bookmark['observation_ms'])}`. {bookmark['description']}" for bookmark in bookmarks["bookmarks"]]
    lines += ["", "## Interpretation Guardrails", ""] + [f"- {guardrail}" for guardrail in report["interpretation_guardrails"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_trace", type=Path); parser.add_argument("zero_trace", type=Path)
    parser.add_argument("--report-json", type=Path, required=True); parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--bookmarks", type=Path, required=True)
    args = parser.parse_args()
    report, bookmarks = analyze(read_trace(args.real_trace), read_trace(args.zero_trace))
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.bookmarks.write_text(json.dumps(bookmarks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report_md.write_text(markdown(report, bookmarks), encoding="utf-8")


if __name__ == "__main__":
    main()
