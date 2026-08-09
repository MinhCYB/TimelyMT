"""Causal streaming strategies and shared prediction artifact semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping, Protocol, Sequence

from timelymt.data.translation_artifacts import RuntimeTalk, TranslationHypothesis


MIN_SOURCE_TOKENS = 4
MAX_SOURCE_TOKENS = 48
HypothesisProvider = Callable[[RuntimeTalk, int, int], TranslationHypothesis]


class CommitPolicy(Protocol):
    def predict_commit_probability(self, state: Mapping[str, Any]) -> float: ...


@dataclass(frozen=True)
class Commit:
    source_start: int
    source_end: int
    observation_token_index: int
    observation_emit_ms: int
    source_token_count: int
    source_clock_duration_ms: int
    target_token_count: int
    translated_text: str
    reason: str
    commit_probability: float | None = None
    causal_features: Mapping[str, float] | None = None


def lcp_length(hypotheses: Sequence[str]) -> int:
    """Longest common whitespace-token prefix, preserving case and punctuation."""
    tokenized = [text.split() for text in hypotheses]
    if not tokenized:
        return 0
    length = 0
    for values in zip(*tokenized):
        if len(set(values)) != 1:
            break
        length += 1
    return length


def pair_lcp_ratio(current: str, previous: str) -> float:
    return lcp_length([current, previous]) / max(1, len(previous.split()))


def normalized_change_ratio(current: str, previous: str) -> float:
    current_tokens, previous_tokens = current.split(), previous.split()
    if not current_tokens and not previous_tokens:
        return 0.0
    return 1.0 - SequenceMatcher(a=previous_tokens, b=current_tokens, autojunk=False).ratio()


def causal_state(
    talk: RuntimeTalk,
    start: int,
    end: int,
    current_hypothesis: str,
    previous_hypothesis: str,
    commits: Sequence[Commit],
) -> dict[str, Any]:
    """Build the only mapping accepted by learned policy inference."""
    tokens = talk.tokens[start : end + 1]
    previous = commits[-1] if commits else None
    current_count, previous_count = len(current_hypothesis.split()), len(previous_hypothesis.split())
    numeric = {
        "source_buffer_token_count": float(len(tokens)),
        "source_buffer_character_count": float(len(" ".join(token.text for token in tokens))),
        "source_clock_elapsed_ms": float(tokens[-1].emit_ms - tokens[0].emit_ms),
        "current_target_token_count": float(current_count),
        "previous_target_token_count": float(previous_count),
        "target_token_count_delta": float(current_count - previous_count),
        "previous_current_lcp_ratio": pair_lcp_ratio(current_hypothesis, previous_hypothesis),
        "previous_current_change_ratio": normalized_change_ratio(current_hypothesis, previous_hypothesis),
        "prior_committed_unit_count": float(len(commits)),
        "previous_committed_source_tokens": float(previous.source_token_count if previous else 0),
        "previous_committed_target_tokens": float(previous.target_token_count if previous else 0),
    }
    return {
        "current_source_text": " ".join(token.text for token in tokens),
        "previous_committed_source_text": (
            " ".join(token.text for token in talk.tokens[previous.source_start : previous.source_end + 1])
            if previous else ""
        ),
        "previous_committed_target_text": previous.translated_text if previous else "",
        "numeric": numeric,
    }


def _commit(
    talk: RuntimeTalk,
    start: int,
    end: int,
    hypothesis: TranslationHypothesis,
    reason: str,
    probability: float | None = None,
    features: Mapping[str, float] | None = None,
) -> Commit:
    tokens = talk.tokens[start : end + 1]
    return Commit(
        source_start=start,
        source_end=end,
        observation_token_index=end,
        observation_emit_ms=tokens[-1].emit_ms,
        source_token_count=len(tokens),
        source_clock_duration_ms=tokens[-1].emit_ms - tokens[0].emit_ms,
        target_token_count=len(hypothesis.translated_text.split()),
        translated_text=hypothesis.translated_text,
        reason=reason,
        commit_probability=probability,
        causal_features=features,
    )


def _emission_commit(
    talk: RuntimeTalk,
    source_start: int,
    observation_end: int,
    target_tokens: Sequence[str],
    reason: str,
) -> Commit:
    """Record a target-prefix emission at its causal source observation."""
    source_tokens = talk.tokens[source_start : observation_end + 1]
    return Commit(
        source_start=source_start,
        source_end=observation_end,
        observation_token_index=observation_end,
        observation_emit_ms=talk.tokens[observation_end].emit_ms,
        source_token_count=len(source_tokens),
        source_clock_duration_ms=(
            source_tokens[-1].emit_ms - source_tokens[0].emit_ms if source_tokens else 0
        ),
        target_token_count=len(target_tokens),
        translated_text=" ".join(target_tokens),
        reason=reason,
    )


def fixed_n(talk: RuntimeTalk, provider: HypothesisProvider, n: int) -> list[Commit]:
    commits: list[Commit] = []
    for start in range(0, len(talk.tokens), n):
        end = min(start + n - 1, len(talk.tokens) - 1)
        reason = "fixed_n" if end - start + 1 == n else "talk_end"
        commits.append(_commit(talk, start, end, provider(talk, start, end), reason))
    return commits


def fixed_time(talk: RuntimeTalk, provider: HypothesisProvider, delta_ms: int) -> list[Commit]:
    commits: list[Commit] = []
    start = 0
    for end in range(len(talk.tokens)):
        count = end - start + 1
        elapsed = talk.tokens[end].emit_ms - talk.tokens[start].emit_ms
        reason = "max_length" if count >= MAX_SOURCE_TOKENS else "fixed_time" if count >= MIN_SOURCE_TOKENS and elapsed >= delta_ms else None
        if reason:
            commits.append(_commit(talk, start, end, provider(talk, start, end), reason))
            start = end + 1
    if start < len(talk.tokens):
        end = len(talk.tokens) - 1
        commits.append(_commit(talk, start, end, provider(talk, start, end), "talk_end"))
    return commits


def local_agreement_style(talk: RuntimeTalk, provider: HypothesisProvider, history_k: int) -> list[Commit]:
    commits: list[Commit] = []
    start = 0
    history: list[TranslationHypothesis] = []
    for end in range(len(talk.tokens)):
        count = end - start + 1
        if count < MIN_SOURCE_TOKENS:
            continue
        hypothesis = provider(talk, start, end)
        history.append(hypothesis)
        window = history[-history_k:]
        ratio = lcp_length([item.translated_text for item in window]) / max(1, len(window[0].translated_text.split()))
        reason = "max_length" if count >= MAX_SOURCE_TOKENS else "agreement" if len(window) == history_k and ratio >= 0.90 else None
        if end == len(talk.tokens) - 1 and reason is None:
            reason = "talk_end"
        if reason:
            commits.append(_commit(talk, start, end, hypothesis, reason))
            start, history = end + 1, []
    if start < len(talk.tokens):
        end = len(talk.tokens) - 1
        commits.append(_commit(talk, start, end, provider(talk, start, end), "talk_end"))
    return commits


def local_agreement_la2(talk: RuntimeTalk, provider: HypothesisProvider) -> list[Commit]:
    """Local Agreement LA-2 adaptation over source-unit token prefixes.

    Consecutive normalized hypotheses use exact ``str.split()`` tokens. Only
    newly agreed prefix tokens are emitted; the current suffix is flushed at
    the 48-token source-unit bound or talk end without revising emitted tokens.
    """
    commits: list[Commit] = []
    start = 0
    previous_tokens: list[str] | None = None
    emitted_count = 0
    last_emission_end = start - 1
    for end in range(len(talk.tokens)):
        count = end - start + 1
        if count < MIN_SOURCE_TOKENS:
            continue
        current_tokens = provider(talk, start, end).translated_text.split()
        termination = (
            "max_length" if count >= MAX_SOURCE_TOKENS
            else "talk_end" if end == len(talk.tokens) - 1
            else None
        )
        if termination:
            remaining = current_tokens[emitted_count:]
            if remaining:
                commits.append(_emission_commit(
                    talk, last_emission_end + 1, end, remaining, termination,
                ))
            start = end + 1
            previous_tokens = None
            emitted_count = 0
            last_emission_end = start - 1
        elif previous_tokens is not None:
            stable_count = lcp_length((" ".join(previous_tokens), " ".join(current_tokens)))
            stable_count = max(emitted_count, stable_count)
            newly_stable = current_tokens[emitted_count:stable_count]
            if newly_stable:
                commits.append(_emission_commit(talk, last_emission_end + 1, end, newly_stable, "agreement"))
                emitted_count = stable_count
                last_emission_end = end
            previous_tokens = current_tokens
        else:
            previous_tokens = current_tokens
    if start < len(talk.tokens):
        end = len(talk.tokens) - 1
        remaining = provider(talk, start, end).translated_text.split()
        if remaining:
            commits.append(_emission_commit(talk, start, end, remaining, "talk_end"))
    return commits


def learned_rollout(
    talk: RuntimeTalk,
    provider: HypothesisProvider,
    policy: CommitPolicy,
    threshold: float,
) -> list[Commit]:
    commits: list[Commit] = []
    start = 0
    previous_hypothesis = ""
    for end in range(len(talk.tokens)):
        count = end - start + 1
        if count < MIN_SOURCE_TOKENS:
            continue
        hypothesis = provider(talk, start, end)
        state = causal_state(talk, start, end, hypothesis.translated_text, previous_hypothesis, commits)
        probability = policy.predict_commit_probability(state)
        reason = "max_length" if count >= MAX_SOURCE_TOKENS else "policy" if probability >= threshold else None
        if end == len(talk.tokens) - 1 and reason is None:
            reason = "talk_end"
        if reason:
            commits.append(_commit(talk, start, end, hypothesis, reason, probability, state["numeric"]))
            start, previous_hypothesis = end + 1, ""
        else:
            previous_hypothesis = hypothesis.translated_text
    if start < len(talk.tokens):
        end = len(talk.tokens) - 1
        hypothesis = provider(talk, start, end)
        state = causal_state(talk, start, end, hypothesis.translated_text, previous_hypothesis, commits)
        probability = policy.predict_commit_probability(state)
        commits.append(_commit(talk, start, end, hypothesis, "talk_end", probability, state["numeric"]))
    return commits


def prediction_record(strategy: str, talk: RuntimeTalk, commits: Sequence[Commit]) -> dict[str, Any]:
    return {
        "artifact_version": "1.0.0",
        "strategy": strategy,
        "talk_id": talk.talk_id,
        "split": talk.split,
        "source_token_count": len(talk.tokens),
        "source_final_emit_ms": talk.tokens[-1].emit_ms,
        "commits": [asdict(commit) for commit in commits],
        "prediction": " ".join(commit.translated_text for commit in commits),
    }


def select_dev_configuration(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    """Apply the preregistered selection rule to already evaluated DEV metrics."""
    if "fixed_n_8" not in metrics:
        raise ValueError("DEV selection requires fixed_n_8 metrics")
    reference_al = metrics["fixed_n_8"]["token_level_average_lagging"]
    learned = [name for name in metrics if name.startswith("learned_P")]
    if not learned:
        raise ValueError("DEV selection requires learned policy metrics")
    eligible = [name for name in learned if metrics[name]["token_level_average_lagging"] <= reference_al]
    def quality_key(name: str) -> tuple[float, float, float, float]:
        row = metrics[name]
        return (row["chrF2"], row["BLEU"], -row["token_level_average_lagging"], float(name.rsplit("_", 1)[1]))
    if eligible:
        selected = max(eligible, key=quality_key)
        reason = "highest DEV chrF among learned configurations with AL no worse than fixed_n_8"
    else:
        smallest_al = min(metrics[name]["token_level_average_lagging"] for name in learned)
        nearest = [name for name in learned if metrics[name]["token_level_average_lagging"] == smallest_al]
        selected = max(nearest, key=quality_key)
        reason = "smallest DEV AL above fixed_n_8, then highest chrF"
    return {
        "selected_strategy": selected,
        "selected_variant": selected.split("_")[1],
        "selected_threshold": float(selected.rsplit("_", 1)[1]),
        "fixed_n_8_dev_AL": reference_al,
        "reason": reason,
        "selected_dev_metrics": dict(metrics[selected]),
    }
