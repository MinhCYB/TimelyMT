"""Reference-owning evaluation, separate from prediction generation."""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Mapping, Sequence


def average_lagging(source_length: int, target_emissions: Sequence[int]) -> float:
    """Standard token-level AL with g(t) equal to consumed lexical source tokens."""
    if source_length <= 0 or not target_emissions:
        raise ValueError("Average Lagging requires non-empty source and target")
    target_length = len(target_emissions)
    ratio = target_length / source_length
    try:
        tau = next(index for index, consumed in enumerate(target_emissions, start=1) if consumed >= source_length)
    except StopIteration:
        tau = target_length
    return sum(target_emissions[t - 1] - (t - 1) / ratio for t in range(1, tau + 1)) / tau


def latency_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    commits = [commit for record in records for commit in record["commits"]]
    source_total = sum(record["source_token_count"] for record in records)
    unit_tokens = [commit["source_token_count"] for commit in commits]
    durations = [commit["source_clock_duration_ms"] for commit in commits]
    first = [record["commits"][0] for record in records]
    al_weighted_sum = 0.0
    target_token_total = 0
    for record in records:
        emissions = [commit["source_end"] + 1 for commit in record["commits"] for _ in commit["translated_text"].split()]
        talk_al = average_lagging(record["source_token_count"], emissions)
        al_weighted_sum += talk_al * len(emissions)
        target_token_total += len(emissions)
    return {
        "number_of_commits": float(len(commits)),
        "commits_per_100_source_tokens": 100 * len(commits) / source_total,
        "mean_source_tokens_per_unit": mean(unit_tokens),
        "median_source_tokens_per_unit": median(unit_tokens),
        "mean_simulated_source_clock_duration_ms": mean(durations),
        "mean_first_commit_source_tokens": mean(item["source_token_count"] for item in first),
        "mean_first_commit_simulated_source_clock_latency_ms": mean(item["observation_emit_ms"] for item in first),
        "mean_commit_observation_position": mean(item["observation_token_index"] + 1 for item in commits),
        "forced_commit_rate": sum(item["reason"] in {"max_length", "talk_end"} for item in commits) / len(commits),
        "token_level_average_lagging": al_weighted_sum / target_token_total,
    }


def quality_metrics(records: Sequence[Mapping[str, Any]], references: Mapping[str, str]) -> dict[str, Any]:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF

    predictions = [record["prediction"] for record in records]
    refs = [references[record["talk_id"]] for record in records]
    bleu_metric, chrf_metric = BLEU(), CHRF(beta=2)
    bleu = bleu_metric.corpus_score(predictions, [refs])
    chrf = chrf_metric.corpus_score(predictions, [refs])
    per_talk = []
    for record, reference in zip(records, refs, strict=True):
        per_talk.append({
            "talk_id": record["talk_id"],
            "BLEU": sacrebleu.sentence_bleu(record["prediction"], [reference]).score,
            "chrF2": sacrebleu.sentence_chrf(record["prediction"], [reference], beta=2).score,
        })
    return {
        "BLEU": bleu.score,
        "chrF2": chrf.score,
        "sacrebleu_version": sacrebleu.__version__,
        "BLEU_signature": str(bleu_metric.get_signature()),
        "chrF2_signature": str(chrf_metric.get_signature()),
        "per_talk": per_talk,
    }
