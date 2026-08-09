"""Future-stability supervision generation; oracle values never enter features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from timelymt.data.translation_artifacts import RuntimeTalk
from .streaming import Commit, HypothesisProvider, MAX_SOURCE_TOKENS, MIN_SOURCE_TOKENS, causal_state, lcp_length


@dataclass(frozen=True)
class PseudoLabelConfig:
    pseudo_label_version: str = "1.0.0"
    min_source_tokens: int = MIN_SOURCE_TOKENS
    max_source_tokens: int = MAX_SOURCE_TOKENS
    lookahead_updates: int = 2
    stability_threshold: float = 0.90
    hypothesis_tokenization: str = "whitespace split"


def generate_pseudo_labels(
    talk: RuntimeTalk,
    provider: HypothesisProvider,
    *,
    max_states: int | None = None,
) -> list[dict[str, Any]]:
    if talk.split == "test":
        raise ValueError("pseudo labels are forbidden for test")
    rows: list[dict[str, Any]] = []
    commits: list[Commit] = []
    start, episode, previous_hypothesis = 0, 0, ""
    for end in range(len(talk.tokens)):
        if max_states is not None and len(rows) >= max_states:
            break
        count = end - start + 1
        if count < MIN_SOURCE_TOKENS:
            continue
        lookahead_ends = [end]
        if end + 2 < len(talk.tokens) and count + 2 <= MAX_SOURCE_TOKENS:
            lookahead_ends.extend((end + 1, end + 2))
        if hasattr(provider, "batch"):
            hypotheses = provider.batch(talk, start, lookahead_ends)  # type: ignore[attr-defined]
        else:
            hypotheses = [provider(talk, start, index) for index in lookahead_ends]
        current = hypotheses[0]
        state = causal_state(talk, start, end, current.translated_text, previous_hypothesis, commits)
        reason, oracle = None, None
        if count >= MAX_SOURCE_TOKENS:
            reason = "max_length"
        elif end == len(talk.tokens) - 1:
            reason = "talk_end"
        elif len(hypotheses) == 3:
            future = hypotheses[1:]
            oracle = lcp_length([current.translated_text, *(item.translated_text for item in future)]) / max(1, len(current.translated_text.split()))
            if oracle >= 0.90:
                reason = "stability"
        row = {
            "talk_id": talk.talk_id,
            "split": talk.split,
            "episode_id": episode,
            "state_source_start": start,
            "state_source_end": end,
            "observation_token_index": end,
            "observation_emit_ms": talk.tokens[end].emit_ms,
            "causal": state,
            "current_normalized_hypothesis": current.translated_text,
            "past_normalized_hypothesis": previous_hypothesis,
            "label": "COMMIT" if reason else "LISTEN",
            "label_reason": reason,
            "oracle_training_only": {"future_stability_ratio": oracle, "lookahead_updates": 2},
        }
        rows.append(row)
        if reason:
            commits.append(Commit(start, end, end, talk.tokens[end].emit_ms, count, talk.tokens[end].emit_ms - talk.tokens[start].emit_ms, len(current.translated_text.split()), current.translated_text, reason))
            start, episode, previous_hypothesis = end + 1, episode + 1, ""
        else:
            previous_hypothesis = current.translated_text
    if start < len(talk.tokens) and (max_states is None or len(rows) < max_states):
        end = len(talk.tokens) - 1
        current = provider(talk, start, end)
        state = causal_state(talk, start, end, current.translated_text, previous_hypothesis, commits)
        rows.append({
            "talk_id": talk.talk_id, "split": talk.split, "episode_id": episode,
            "state_source_start": start, "state_source_end": end,
            "observation_token_index": end, "observation_emit_ms": talk.tokens[end].emit_ms,
            "causal": state, "current_normalized_hypothesis": current.translated_text,
            "past_normalized_hypothesis": previous_hypothesis, "label": "COMMIT",
            "label_reason": "talk_end",
            "oracle_training_only": {"future_stability_ratio": None, "lookahead_updates": 2},
        })
    return rows


def config_document() -> dict[str, Any]:
    return asdict(PseudoLabelConfig())
