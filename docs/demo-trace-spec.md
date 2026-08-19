# TimelyMT Demo Trace Contract

## Purpose and Scope

This specification defines a new, append-only, optional trace artifact for a faithful streaming research demo. It does not reconstruct missing events from existing commit artifacts. Existing artifacts store only committed observations and therefore cannot provide every candidate translation, LISTEN decision, non-commit probability, or policy state.

Initial scope is P3_GLOBAL DEV-only traces for `ted-sims-witherspoon-ai-climate`, with side-by-side `real` and `zero` prepared-context modes at threshold 0.60. Threshold 0.50 may be traced later if explicitly requested. TEST is forbidden.

## Artifact Location and Identity

Write one JSON document per rollout condition:

```text
outputs/demo_traces/<run_id>.json
```

`run_id` should be deterministic and human-readable, for example:

```text
p3-global__ted-sims-witherspoon-ai-climate__dev__real__0.60__ccf829fd
```

The trace is separate from canonical prediction artifacts. It must never add fields to, rewrite, or change normal `outputs/experiments/policy-p3-global/predictions/...` records.

Recommended top-level schema:

```json
{
  "artifact_version": "demo-policy-trace-v1",
  "run_id": "p3-global__ted-sims-witherspoon-ai-climate__dev__real__0.60__ccf829fd",
  "talk_id": "ted-sims-witherspoon-ai-climate",
  "split": "dev",
  "strategy": "p3_global_0.60",
  "threshold": 0.60,
  "prepared_context_mode": "real",
  "checkpoint_sha256": "...",
  "source_stream": {
    "source_token_count": 1689,
    "source_final_emit_ms": 675600,
    "clock": "simulated_source_emit_ms",
    "observation_key": "source_token_end"
  },
  "prepared_context": {},
  "events": []
}
```

The generator should copy the P3 checkpoint SHA-256 from validated checkpoint metadata and should record the concrete strategy name. It may additionally record frozen translator/config and prepared-manifest fingerprints when readily available, but must not load new model inputs only to enrich metadata.

## Prepared Context Provenance

Store the existing prepared-context provenance, plus the mode and effective norm. For the first showcase, the panel must make these facts explicit:

| Field | REAL_CONTEXT | ZERO_CONTEXT |
|---|---|---|
| Eligible source | `deepmind-wind-energy-2019-lead` | Same source/provenance |
| Eligibility | `SAFE_PRETALK_CONFIRMED` | Same eligibility |
| Effective context norm | 1.0 | 0.0 |

The trace must state that the source exists in both conditions; only the policy embedding is disabled in ZERO mode. It must not imply that the document was removed or that EnViT5 received it.

Suggested provenance shape:

```json
{
  "representation_version": "prepared-global-v0",
  "eligible_source_ids": ["deepmind-wind-energy-2019-lead"],
  "eligible_source_checksums": ["sha256:..."],
  "has_eligible_context": true,
  "embedding_dimension": 384,
  "prepared_context_embedding_norm": 1.0,
  "prepared_context_effective_embedding_norm": 1.0,
  "mode_explanation": "Source provenance is retained; only the policy prepared-global slice is zeroed in zero mode."
}
```

Do not store the 384-dimensional embedding or the full 1,547-dimensional policy input by default.

## Event Timeline

`events` is one append-only sequence ordered by `event_index`. Every arrival creates exactly one event for its policy session. `event_index` is zero-based and equals the observed source token index for the current single-token source stream. `source_token_end` and `observation_ms` are the stable synchronization keys.

Each event contains only information available causally at that observation. All text must be semantic text already present in the stream or already returned by the existing translator call; the implementation must not generate a second candidate merely for tracing.

```json
{
  "event_index": 17,
  "observation_ms": 7616,
  "source_token_end": 17,
  "candidate_source_start": 16,
  "candidate_source_end": 17,
  "candidate_source_text": "like this assembling",
  "candidate_translation": null,
  "previous_candidate_translation": null,
  "p_commit": null,
  "threshold": 0.60,
  "decision": "WAIT",
  "decision_reason": "min_source_tokens",
  "is_forced": false,
  "committed_unit_index": null,
  "committed_source_text": null,
  "committed_target_text": null,
  "numeric_features": null
}
```

### Common Fields

| Field | Meaning |
|---|---|
| `event_index` | Monotonic observation index; one event per source-token arrival |
| `observation_ms` | Emit time of `source_token_end` in simulated source-clock milliseconds |
| `source_token_end` | Inclusive global source token index observed at this event |
| `candidate_source_start` | Inclusive first token of this session's uncommitted candidate |
| `candidate_source_end` | Inclusive final observed candidate token; equals `source_token_end` |
| `candidate_source_text` | Exact causal source reconstruction for the open candidate span |
| `candidate_translation` | Normalized EnViT5 candidate, or `null` for WAIT |
| `previous_candidate_translation` | Prior candidate translation for this open span, or `null` when absent/reset |
| `p_commit` | P3 probability, or `null` only for WAIT |
| `threshold` | Threshold used for this trace/run |
| `decision` | `WAIT`, `LISTEN`, or `COMMIT` |
| `decision_reason` | Explicit reason described below |
| `is_forced` | `true` only for `max_length` or `talk_end` COMMIT |
| `committed_unit_index` | Zero-based index for COMMIT, otherwise `null` |
| `committed_source_text` | Exact committed source span for COMMIT, otherwise `null` |
| `committed_target_text` | Candidate translation committed at that event, otherwise `null` |

For LISTEN and COMMIT events, `numeric_features` records the existing eleven causal numeric features without embeddings:

```json
{
  "source_buffer_token_count": 4.0,
  "source_buffer_character_count": 24.0,
  "source_clock_elapsed_ms": 1884.0,
  "current_target_token_count": 5.0,
  "previous_target_token_count": 0.0,
  "target_token_count_delta": 5.0,
  "previous_current_lcp_ratio": 0.0,
  "previous_current_change_ratio": 1.0,
  "prior_committed_unit_count": 3.0,
  "previous_committed_source_tokens": 4.0,
  "previous_committed_target_tokens": 5.0
}
```

These values are recorded after the existing `causal_state(...)` call. They are an observability representation of inputs already used by the policy, not a new feature path.

### Event Semantics

| Decision | Preconditions | `decision_reason` | Required fields |
|---|---|---|---|
| WAIT | Candidate has fewer than 4 source tokens | `min_source_tokens` | No translator/policy fields: `candidate_translation`, `p_commit`, `numeric_features` are `null` |
| LISTEN | Candidate has at least 4 tokens; `p_commit < threshold`; not final/forced | `below_threshold` | Candidate translation, previous candidate translation, `p_commit`, and numeric features present |
| COMMIT | Candidate is committed | `policy`, `max_length`, or `talk_end` | Candidate translation, `p_commit`, numeric features, and committed fields present |

Commit precedence must exactly match the current `learned_rollout` implementation:

1. `max_length` when candidate token count is at least 48.
2. `policy` when `p_commit >= threshold` and max length did not apply.
3. `talk_end` when the final observed token did not otherwise commit.

The post-loop talk-end fallback is also represented as a COMMIT event. It occurs when the residual candidate remains below four tokens and therefore did not enter the main inference branch. The current code nevertheless computes its existing candidate translation, causal state, and probability before committing it. In the trace, this is a `COMMIT` with `decision_reason="talk_end"` and `is_forced=true`; no synthetic preceding LISTEN events may be invented.

## REAL/ZERO Synchronization

The two policy sessions are independent after a commit. They may have different candidate starts, translations, probabilities, commits, and histories. They must never share commit/history state.

They are synchronized by the common source observation timeline:

- Both trace files must describe the same `talk_id`, split, source token count, and final emit time.
- Join events primarily on `source_token_end`, with `event_index` as the identical observation index.
- Validate matching `observation_ms` at each joined source token.
- Do not join by commit index or assume coincident commits.

For a side-by-side UI, source playback advances from the shared source timeline. At each source observation, show the one REAL event and one ZERO event keyed by that source token. A divergence is any joined event where `decision` differs; its probability delta is `p_real - p_zero` only when both probabilities are present. WAIT has no probability and must not be rendered as zero.

## Exact Instrumentation Point

Instrument `src/timelymt/research/streaming.py`, function `learned_rollout`, in the `for end in range(len(talk.tokens))` observation loop.

The trace event must be assembled from the already computed values at this exact point:

```text
hypothesis = provider(talk, start, end)
state = causal_state(...)
probability = policy.predict_commit_probability(state)
reason = max_length / policy threshold / talk_end
```

Emit a LISTEN or COMMIT trace event after `reason` has been determined and before the existing `start`/`previous_hypothesis` mutation. Emit WAIT events in the existing `count < MIN_SOURCE_TOKENS` branch before `continue`. Emit the residual post-loop talk-end COMMIT from the existing fallback block after its already computed hypothesis/state/probability. This captures full observation behavior while using exactly the translator output and policy probability the canonical rollout already computed.

The P3 call-site integration belongs in `src/timelymt/research/policy_p3_global_runner.py`, function `rollout_p3`, where a DEV-only trace sink can be created per `(talk_id, threshold, prepared_context_mode)` and passed into `learned_rollout`. The CLI plumbing belongs in `src/timelymt/research/cli.py` under `rollout-p3`.

Recommended interface:

```text
rollout-p3 ... --trace-output outputs/demo_traces/<run_id>.json
```

Require a single DEV talk and a single threshold whenever `--trace-output` is supplied. Reject `--split train`, `--split test`, multi-talk, and multi-threshold trace requests. Tracing is OFF by default.

Use a narrow optional callback/protocol argument on `learned_rollout`, for example `trace_event: Callable[[Mapping[str, Any]], None] | None = None`. The default `None` path must preserve existing behavior and canonical output. The runner owns JSON assembly and atomic trace writing after successful rollout; `learned_rollout` only emits in-memory causal event mappings. This prevents trace I/O from changing decision logic and avoids duplicate inference.

## Validation Plan

Future instrumentation must add focused offline tests proving:

1. Tracing OFF produces byte-identical canonical prediction artifacts, or at minimum semantically identical artifacts if unavoidable serialization plumbing changes are separately justified.
2. Tracing ON produces identical prediction text, commit records, commit reasons, probabilities, and spans to tracing OFF for the same deterministic fake provider/policy.
3. Every event contains only causal source up through `source_token_end`; no future token, reference, alignment, oracle, or gold field is present.
4. WAIT events contain no `p_commit`, candidate translation, previous candidate translation, or numeric policy features.
5. LISTEN and COMMIT events contain `p_commit`, candidate translation, and all eleven numeric features.
6. Every COMMIT event maps one-to-one, in order, to the canonical `Commit` record, including source span, observation time, reason, probability, and target text.
7. The final `talk_end` flush is represented, including the shorter-than-four-token fallback case.
8. REAL and ZERO traces for one talk have the same complete source timeline (`event_index`, `source_token_end`, and `observation_ms`), while allowing independent policy state and commits.
9. Trace CLI rejects TEST paths and permits only scoped DEV output.
10. Trace metadata records the requested P3 checkpoint SHA-256 and prepared-context mode; zero mode preserves source provenance while reporting effective norm zero.

No future implementation should test this by accessing TEST, regenerating translations outside the requested DEV trace run, training, evaluating, or selecting a winner.

## Local Demo Stack Recommendation

The repository has Python research code and no existing frontend package or web application. The smallest local-first stack is therefore:

```text
Python trace generation
+ static HTML/CSS/vanilla JavaScript viewer
+ python -m http.server for local serving on Windows
```

This adds no Node, bundler, database, authentication, API server, or framework dependency. The viewer loads two pre-generated JSON trace files, performs its joins in the browser, and has no inference capability. It is sufficient for desktop playback, state stepping, and divergence inspection. A framework should be considered only if the minimal static viewer proves inadequate.
