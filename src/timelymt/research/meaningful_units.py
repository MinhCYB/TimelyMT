"""Zhang et al. (2020) Meaningful Unit semantic adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from timelymt.data.translation_artifacts import RuntimeTalk

from .streaming import (
    Commit,
    HypothesisProvider,
    MAX_SOURCE_TOKENS,
    MIN_SOURCE_TOKENS,
    _commit,
    lcp_length,
)


MU_NUMERIC_FEATURES = (
    "source_buffer_token_count",
    "source_buffer_character_count",
    "source_clock_elapsed_ms",
    "current_target_token_count",
    "source_target_length_ratio",
)
MU_TEXT_FEATURES = ("current_source_text", "current_hypothesis_text")


@dataclass(frozen=True)
class MeaningfulUnitConfig:
    supervision_version: str = "1.0.0"
    min_source_tokens: int = MIN_SOURCE_TOKENS
    max_source_tokens: int = MAX_SOURCE_TOKENS
    prefix_preservation_threshold: float = 0.90
    hypothesis_tokenization: str = "translated_text.split()"
    oracle_scope: str = "full admissible remaining source unit (up to 48 lexical tokens); training/dev supervision only"


def mu_causal_state(talk: RuntimeTalk, start: int, end: int, hypothesis: str) -> dict[str, Any]:
    tokens = talk.tokens[start : end + 1]
    source_text = " ".join(token.text for token in tokens)
    target_count = len(hypothesis.split())
    return {
        "current_source_text": source_text,
        "current_hypothesis_text": hypothesis,
        "numeric": {
            "source_buffer_token_count": float(len(tokens)),
            "source_buffer_character_count": float(len(source_text)),
            "source_clock_elapsed_ms": float(tokens[-1].emit_ms - tokens[0].emit_ms),
            "current_target_token_count": float(target_count),
            "source_target_length_ratio": float(len(tokens) / max(1, target_count)),
        },
    }


def _prefix_preservation_ratio(current: str, full: str) -> float:
    return lcp_length((current, full)) / max(1, len(current.split()))


def generate_mu_supervision(
    talk: RuntimeTalk,
    provider: HypothesisProvider,
    *,
    max_states: int | None = None,
) -> list[dict[str, Any]]:
    """Create independent MU labels using non-causal oracle translations."""
    if talk.split == "test":
        raise ValueError("MU supervision is forbidden for test")
    rows: list[dict[str, Any]] = []
    start = episode = 0
    while start < len(talk.tokens):
        committed = False
        for end in range(start, len(talk.tokens)):
            if max_states is not None and len(rows) >= max_states:
                return rows
            count = end - start + 1
            if count < MIN_SOURCE_TOKENS:
                continue
            oracle_end = min(start + MAX_SOURCE_TOKENS - 1, len(talk.tokens) - 1)
            ends = [end] if end == oracle_end else [end, oracle_end]
            if hasattr(provider, "batch"):
                hypotheses = provider.batch(talk, start, ends)  # type: ignore[attr-defined]
            else:
                hypotheses = [provider(talk, start, value) for value in ends]
            current = hypotheses[0]
            full = hypotheses[-1]
            ratio = _prefix_preservation_ratio(current.translated_text, full.translated_text)
            reason = (
                "talk_end" if end == len(talk.tokens) - 1
                else "max_length" if end == oracle_end
                else "meaningful_unit" if ratio >= MeaningfulUnitConfig().prefix_preservation_threshold
                else None
            )
            state = mu_causal_state(talk, start, end, current.translated_text)
            rows.append({
                "talk_id": talk.talk_id,
                "split": talk.split,
                "episode_id": episode,
                "state_source_start": start,
                "state_source_end": end,
                "observation_token_index": end,
                "observation_emit_ms": talk.tokens[end].emit_ms,
                "causal": state,
                "label": "COMMIT" if reason else "LISTEN",
                "label_reason": reason,
                "mu_oracle_training_only": {
                    "full_source_end": oracle_end,
                    "prefix_preservation_ratio": ratio,
                    "threshold": MeaningfulUnitConfig().prefix_preservation_threshold,
                },
            })
            if reason:
                start, episode, committed = end + 1, episode + 1, True
                break
        if not committed:
            break
    return rows


def flatten_mu_state(state: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {name: float(state["numeric"][name]) for name in MU_NUMERIC_FEATURES}
    row.update({name: str(state[name]) for name in MU_TEXT_FEATURES})
    return row


def _select_mu_numeric(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [[float(row[name]) for name in MU_NUMERIC_FEATURES] for row in rows]


def _select_mu_text(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[str]:
    return [str(row[key]) for row in rows]


@dataclass
class MeaningfulUnitPolicy:
    pipeline: Any

    def predict_commit_probability(self, state: Mapping[str, Any]) -> float:
        return float(self.pipeline.predict_proba([flatten_mu_state(state)])[0, 1])


def train_mu_policy(rows: Sequence[Mapping[str, Any]]) -> MeaningfulUnitPolicy:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    transformers: list[tuple[str, Any]] = [("numeric", Pipeline([
        ("select", FunctionTransformer(_select_mu_numeric, validate=False)),
        ("scale", StandardScaler()),
    ]))]
    for column in MU_TEXT_FEATURES:
        transformers.append((column, Pipeline([
            ("select", FunctionTransformer(_select_mu_text, kw_args={"key": column}, validate=False)),
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), max_features=10000, min_df=1,
                token_pattern=r"(?u)\b\w+\b",
            )),
        ])))
    pipeline = Pipeline([
        ("features", FeatureUnion(transformers)),
        ("classifier", LogisticRegression(
            class_weight="balanced", random_state=20260809, max_iter=1000, solver="liblinear",
        )),
    ])
    examples = [flatten_mu_state(row["causal"]) for row in rows]
    labels = [1 if row["label"] == "COMMIT" else 0 for row in rows]
    pipeline.fit(examples, labels)
    return MeaningfulUnitPolicy(pipeline)


def mu_rollout(
    talk: RuntimeTalk,
    provider: HypothesisProvider,
    policy: Any,
    threshold: float = 0.50,
) -> list[Commit]:
    """Sequential causal MU segmentation with no oracle or history at runtime."""
    commits: list[Commit] = []
    start = 0
    for end in range(len(talk.tokens)):
        count = end - start + 1
        if count < MIN_SOURCE_TOKENS:
            continue
        hypothesis = provider(talk, start, end)
        state = mu_causal_state(talk, start, end, hypothesis.translated_text)
        probability = policy.predict_commit_probability(state)
        reason = "max_length" if count >= MAX_SOURCE_TOKENS else "meaningful_unit" if probability >= threshold else None
        if end == len(talk.tokens) - 1 and reason is None:
            reason = "talk_end"
        if reason:
            commits.append(_commit(talk, start, end, hypothesis, reason, probability, state["numeric"]))
            start = end + 1
    if start < len(talk.tokens):
        end = len(talk.tokens) - 1
        hypothesis = provider(talk, start, end)
        state = mu_causal_state(talk, start, end, hypothesis.translated_text)
        probability = policy.predict_commit_probability(state)
        commits.append(_commit(talk, start, end, hypothesis, "talk_end", probability, state["numeric"]))
    return commits


def mu_config_document() -> dict[str, Any]:
    return asdict(MeaningfulUnitConfig())
