"""Lightweight causal LISTEN/COMMIT logistic-regression policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


NUMERIC_FEATURES = (
    "source_buffer_token_count", "source_buffer_character_count", "source_clock_elapsed_ms",
    "current_target_token_count", "previous_target_token_count", "target_token_count_delta",
    "previous_current_lcp_ratio", "previous_current_change_ratio", "prior_committed_unit_count",
    "previous_committed_source_tokens", "previous_committed_target_tokens",
)
VARIANTS = ("P0", "P1", "P2")


def _select_numeric(rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [[float(row[name]) for name in NUMERIC_FEATURES] for row in rows]


def _select_text(rows: Sequence[Mapping[str, Any]], *, key: str) -> list[str]:
    return [str(row[key]) for row in rows]


def flatten_state(state: Mapping[str, Any], variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown policy variant: {variant}")
    row = {name: float(state["numeric"][name]) for name in NUMERIC_FEATURES}
    row["current_source_text"] = state["current_source_text"]
    if variant in {"P1", "P2"}:
        row["previous_source_text"] = state["previous_committed_source_text"]
    if variant == "P2":
        row["previous_target_text"] = state["previous_committed_target_text"]
    return row


def build_pipeline(variant: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    text_columns = ["current_source_text"]
    if variant in {"P1", "P2"}:
        text_columns.append("previous_source_text")
    if variant == "P2":
        text_columns.append("previous_target_text")
    transformers: list[tuple[str, Any]] = [
        ("numeric", Pipeline([
            ("select", FunctionTransformer(_select_numeric, validate=False)),
            ("scale", StandardScaler()),
        ]))
    ]
    for column in text_columns:
        transformers.append((column, Pipeline([
            ("select", FunctionTransformer(_select_text, kw_args={"key": column}, validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2)),
        ])))
    return Pipeline([
        ("features", FeatureUnion(transformers)),
        ("classifier", LogisticRegression(class_weight="balanced", random_state=20260809, max_iter=1000, solver="liblinear")),
    ])


@dataclass
class LearnedPolicy:
    variant: str
    pipeline: Any

    def predict_commit_probability(self, state: Mapping[str, Any]) -> float:
        return float(self.pipeline.predict_proba([flatten_state(state, self.variant)])[0, 1])


def train_policy(rows: Sequence[Mapping[str, Any]], variant: str) -> LearnedPolicy:
    pipeline = build_pipeline(variant)
    examples = [flatten_state(row["causal"], variant) for row in rows]
    labels = [1 if row["label"] == "COMMIT" else 0 for row in rows]
    pipeline.fit(examples, labels)
    return LearnedPolicy(variant, pipeline)
