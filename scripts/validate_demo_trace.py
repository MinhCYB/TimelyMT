"""Read-only validation for a P3 demo policy trace artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


NUMERIC_FEATURES = {
    "source_buffer_token_count",
    "source_buffer_character_count",
    "source_clock_elapsed_ms",
    "current_target_token_count",
    "previous_target_token_count",
    "target_token_count_delta",
    "previous_current_lcp_ratio",
    "previous_current_change_ratio",
    "prior_committed_unit_count",
    "previous_committed_source_tokens",
    "previous_committed_target_tokens",
}


def validate(trace: dict[str, Any]) -> None:
    if trace.get("artifact_version") != "demo-policy-trace-v1":
        raise ValueError("unsupported artifact_version")
    if trace.get("split") != "dev":
        raise ValueError("demo traces must be DEV only")
    if trace.get("prepared_context_mode") not in {"real", "zero"}:
        raise ValueError("invalid prepared_context_mode")
    context = trace.get("prepared_context")
    if not isinstance(context, dict):
        raise ValueError("missing prepared_context")
    if trace["prepared_context_mode"] == "zero" and context.get("prepared_context_effective_embedding_norm") != 0.0:
        raise ValueError("zero context must report effective norm 0.0")
    events = trace.get("events")
    if not isinstance(events, list) or len(events) != trace.get("source_token_count"):
        raise ValueError("event count must equal source_token_count")
    previous_time = -1
    for index, event in enumerate(events):
        if event.get("event_index") != index or event.get("source_token_end") != index:
            raise ValueError("event timeline is not contiguous")
        observation_ms = event.get("observation_ms")
        if not isinstance(observation_ms, int) or observation_ms < previous_time:
            raise ValueError("observation_ms is not monotonic")
        previous_time = observation_ms
        start, end = event.get("candidate_source_start"), event.get("candidate_source_end")
        if not isinstance(start, int) or start < 0 or end != index or start > end:
            raise ValueError("candidate source indices are not causal")
        decision = event.get("decision")
        if decision == "WAIT":
            if end - start + 1 >= 4 or any(event.get(name) is not None for name in ("candidate_translation", "previous_candidate_translation", "p_commit", "numeric_features")):
                raise ValueError("WAIT contains inference data")
        elif decision in {"LISTEN", "COMMIT"}:
            if event.get("candidate_translation") is None or event.get("p_commit") is None:
                raise ValueError(f"{decision} is missing candidate inference data")
            if set((event.get("numeric_features") or {}).keys()) != NUMERIC_FEATURES:
                raise ValueError(f"{decision} numeric features differ from the causal 11")
            if decision == "LISTEN" and event["p_commit"] >= event["threshold"]:
                raise ValueError("LISTEN reaches threshold")
            if decision == "COMMIT":
                if event.get("committed_unit_index") is None or event.get("committed_source_text") is None or event.get("committed_target_text") is None:
                    raise ValueError("COMMIT is missing committed fields")
        else:
            raise ValueError("invalid decision")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    validate(json.loads(args.trace.read_text(encoding="utf-8")))
    print(f"valid trace: {args.trace}")


if __name__ == "__main__":
    main()
