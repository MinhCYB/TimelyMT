# TimelyMT Research Demo UI Specification

## Goal

Build one desktop research-observability page for a faithful replay of streaming policy traces. The primary showcase is `ted-sims-witherspoon-ai-climate`, comparing the same P3 checkpoint under REAL_CONTEXT and ZERO_CONTEXT at threshold 0.60. This is the most informative descriptive controlled-ablation case, not a claim that 0.60 is the best threshold.

The page explains five facts visually:

1. Source tokens arrive over time.
2. The candidate translation can change while the policy LISTENs.
3. The policy commits only when its threshold/forced rule says so.
4. Prepared context can alter the policy decision.
5. Committed Vietnamese output is irreversible and changes each session's later state.

The UI is a research instrument, not a polished consumer translator. It must render only trace fields actually captured at the corresponding observation. It must never fabricate LISTEN decisions or probabilities from commit-only artifacts.

## Desktop Layout

```text
--------------------------------------------------------------------------
TimelyMT - Context-Aware Streaming Translation
Sims Witherspoon: Can AI help solve the climate crisis? | 00:00.000 | 0.60
[Play] [Pause] [Step] [Reset] [0.5x] [1x] [2x] [4x] [Threshold: 0.60] [Next difference]
--------------------------------------------------------------------------
Source Stream                         Policy Comparison                  Prepared Knowledge
Observed English text                 REAL CONTEXT | ZERO CONTEXT         source / eligibility / norms
New token highlight                   candidate     candidate             source ID and provenance
Candidate source range                p/threshold   p/threshold           short approved excerpt
                                      WAIT/LISTEN/COMMIT                  mode explanation
--------------------------------------------------------------------------
Committed Translation Timeline
REAL committed Vietnamese output      ZERO committed Vietnamese output
Compact shared source-clock markers, colored by REAL/ZERO commit and divergence
--------------------------------------------------------------------------
Optional Numeric Features (collapsed by default)
REAL state features                   ZERO state features
--------------------------------------------------------------------------
```

Use a wide three-column desktop grid. The source and policy comparison are the visual focus; prepared knowledge is a compact evidence panel, not a document reader. On smaller windows, stack Prepared Knowledge under the policy panel while preserving the shared source controls and two-condition comparison.

## Header and Controls

Header content:

- Product/research title: `TimelyMT - Context-Aware Streaming Translation`.
- Talk title and ID in a smaller subtitle.
- Shared simulated source-clock time, formatted `mm:ss.mmm`.
- Active threshold and trace identity/mode labels.

Controls:

| Control | Behavior |
|---|---|
| Play / Pause | Advance through the shared source observation timeline; no inference occurs in the UI |
| Step one source observation | Advance exactly one joined `source_token_end` event |
| Reset | Return both panes and committed timelines to observation zero |
| 0.5x / 1x / 2x / 4x | Playback multiplier for elapsed source-clock time; stepping stays one observation |
| Threshold selector | Show only thresholds for which both matching REAL/ZERO trace files exist; initially 0.60, optionally 0.50 |
| Show numeric features | Reveal/hide the existing eleven causal numeric features for each condition |
| Next differing REAL/ZERO decision | Jump to the next joined observation whose `decision` differs |

Controls operate on loaded trace files only. The threshold selector loads another pre-generated pair; it does not alter probabilities, recompute decisions, or re-run rollout.

## Source Stream Panel

Show a bounded scrolling source window rather than the entire talk at once:

- Earlier observed tokens: neutral text.
- Newly arrived token: strong highlight for the current `source_token_end`.
- Unobserved future source: hidden by default, not gray text. A research viewer should not visually leak the future stream.
- Display the two current candidate ranges separately when REAL and ZERO have diverged; the spans can differ after a commit.
- Display exact candidate source text from each trace event, not reconstructed sentence boundaries or canonical subtitle text.

The source panel uses the shared clock. It must not use a policy's commit index as playback time.

## Policy Comparison Panel

Render two equal cards, REAL_CONTEXT on the left and ZERO_CONTEXT on the right. At every joined observation, each card shows:

- Decision badge: WAIT, LISTEN, or COMMIT.
- Candidate source range and candidate source text.
- Candidate Vietnamese translation when present.
- `p(COMMIT)` and threshold when inference occurred.
- Decision reason: `min_source_tokens`, `below_threshold`, `policy`, `max_length`, or `talk_end`.
- Forced indicator only for `max_length` and `talk_end`.
- Prior committed-unit count and candidate buffer size from numeric features when the details toggle is enabled.

WAIT must explicitly state: `Fewer than 4 candidate source tokens. No translation or policy inference.` It must show `p(COMMIT): not evaluated`, not `0.00`.

For a policy COMMIT, add a small immutable marker such as `Committed unit N`. Once a unit appears in the bottom timeline, keep it visually fixed during later playback. Do not display a later candidate as a revision of a committed unit.

## Difference Highlighting

At each synchronized observation, compare `decision` values:

| Joined state | Treatment |
|---|---|
| Different decisions | Strong shared divergence band above both cards |
| Same WAIT or same LISTEN | Deemphasize comparison; normal state cards |
| Same COMMIT | Neutral shared commit marker |

For divergent decisions, prominently show:

```text
REAL: LISTEN, p = 0.57
ZERO: COMMIT, p = 0.63
threshold = 0.60
delta p (REAL - ZERO) = -0.06
```

The actual order depends on the trace. Use semantic colors with labels and text, not color alone: a distinct COMMIT color, a muted LISTEN color, and a neutral WAIT state. `Next differing REAL/ZERO decision` advances by joined source observation, skipping no source events in the trace itself.

If either side is WAIT, show probability delta as unavailable. Never treat a missing probability as zero.

## Prepared Knowledge Panel

Show provenance as scientific context rather than model explanation. For the Sims trace, display:

| Field | Display |
|---|---|
| Prepared source | `deepmind-wind-energy-2019-lead` |
| Type | Official article |
| Eligibility | `SAFE_PRETALK_CONFIRMED` |
| Leakage controls | Available before talk; transcript not used; reference not used |
| REAL effective context norm | 1.0 |
| ZERO effective context norm | 0.0 |

Show a short excerpt and source URL/provenance from the trace metadata. The panel must include this exact interpretation in plain language:

> The approved prepared source exists in both experimental conditions. ZERO_CONTEXT disables only its 384-dimensional policy embedding. The source is not supplied to EnViT5 in either condition.

Do not display an embedding heatmap, raw 384-vector, or inferred semantic attribution. The experiment establishes an intervention on the vector, not a human-readable explanation of individual probability changes.

## Committed Translation Timeline

The bottom region is a shared-clock timeline plus two independent irreversible output streams.

- Place compact REAL and ZERO commit markers at each event's `observation_ms` on the same horizontal clock.
- Mark a divergence-originating commit with a small outlined indicator.
- Below markers, show separate accumulated Vietnamese units in commit order for REAL and ZERO.
- Each unit includes commit number, source-clock time, reason, source index range, and target text.
- Scroll newly committed units into view during playback but do not delete prior units.

Avoid pretending that all commits align. The rows are independently ordered by each policy's `committed_unit_index`; shared horizontal placement is by source clock.

## Numeric Features Panel

This panel is collapsed by default. When expanded, display the eleven values available in the trace for each non-WAIT event in a compact two-column comparison table:

```text
source_buffer_token_count
source_buffer_character_count
source_clock_elapsed_ms
current_target_token_count
previous_target_token_count
target_token_count_delta
previous_current_lcp_ratio
previous_current_change_ratio
prior_committed_unit_count
previous_committed_source_tokens
previous_committed_target_tokens
```

For WAIT, display `not evaluated` for all policy-only values. Do not add embeddings, target references, future source, oracle labels, alignments, or any latent state not captured by the trace contract.

## Data Loading and Failure States

The UI takes a manifest or two explicitly configured trace paths. Before rendering, validate:

- Matching `artifact_version`, talk, split, source token count, final emit time, threshold, and checkpoint SHA-256.
- REAL trace has mode `real`; ZERO trace has mode `zero`.
- Both event arrays cover matching `event_index`, `source_token_end`, and `observation_ms` timelines.
- Every event decision and required field obeys `docs/demo-trace-spec.md`.

If validation fails, show a blocking research-data error and no comparison. Do not attempt approximate joins or fallback to existing rollout commit artifacts.

## Non-Goals

- No live model inference, training, rollout, or evaluation from the UI.
- No TEST data, path, selector, or reference display.
- No LLM translator, translator rebuild, or prepared-document conditioning of EnViT5.
- No polished consumer product features, accounts, backend database, or dashboard expansion.
- No claims that P3 is superior or that prepared context generally improves TimelyMT.
