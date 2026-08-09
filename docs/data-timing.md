# Source Timing And Streaming Tokens

## Purpose

M0.5 converts an English `ParsedTranscript` into the source-only timed intermediate defined by [`schemas/timed-source.schema.json`](../schemas/timed-source.schema.json). It creates causal lexical token availability for later streaming experiments without building the canonical streaming talk, running a translator, or using bilingual alignment.

The lifecycle is:

```text
data/streaming/raw/
data/streaming/parsed/
data/streaming/aligned/
data/streaming/timed/
data/streaming/processed/
```

The conventional output is `data/streaming/timed/<talk_id>/source.en.json`. This artifact contains only English source reference segments, English lexical tokens, source timing parameters, statistics, and provenance. The timing CLI reads only `data/streaming/parsed/<talk_id>/source.en.json`; Vietnamese text, target lengths, target timing, alignment costs, and alignment groups cannot affect source timing.

## Causal Semantics

All timestamps are non-negative integer milliseconds. `emit_ms` is the end boundary of a token: the moment when the complete source token has become available to the online system. At runtime, a token is visible exactly when `current_time >= emit_ms`. It is not a token start timestamp.

Emission times are globally monotonically non-decreasing. Equal times are valid, including at touching caption boundaries and within zero-duration captions.

## Lexical Tokens

Runtime tokens are deterministic, lightweight, human-readable lexical units independent of Marian, XLM-R, MiniLM, SentencePiece, BPE, or any neural tokenizer. Whitespace and standalone punctuation are not tokens. Leading, trailing, and subtitle punctuation are removed so prepared punctuation does not become a free online boundary cue. Apostrophes and hyphens inside words are retained, as are conservative forms for numbers, acronyms, and technical terms such as `we're`, `GPT-4`, `3.14`, `C++`, and `C#`.

Parsed segment text is never mutated. For example, reference text `AI, however, is changing quickly.` remains unchanged while its runtime token texts are `AI`, `however`, `is`, `changing`, `quickly`. A punctuation-only segment remains in the artifact with an empty token array.

## Recovered Caption Timing

When every parsed English segment has an original `start_ms`, mode auto-detection selects `recovered_from_caption_starts`. For every segment except the final one, the next caption start is the current end boundary:

```text
current.end_ms = next.start_ms
```

No millisecond is subtracted. Starts remain authoritative, input order is preserved, and decreasing starts reject generation rather than being silently sorted. The artifact records the original parsed `timing_source`, such as `wit3_seekvideo`.

The final caption first uses a non-negative integer `duration_ms` or `talk_duration_ms` from English parsed source metadata when it is at least the final start. Otherwise its duration is estimated from its lexical token count and the configured simulated speech rate. The selected strategy is recorded as `source_metadata_duration` or `speech_rate_estimate`; an estimated boundary is never represented as an original timestamp.

Duplicate adjacent starts produce a zero-duration caption. Its lexical tokens all complete at the shared boundary, retain deterministic global order, and may share `emit_ms`. A token-bearing caption's final token always emits at its segment end.

## Simulated TED Timing

When every English parsed segment has null `start_ms`, mode auto-detection selects `simulated`. This is the case for the current modern TED continuous-transcript pilots. Segment duration is computed from lexical token count and a configurable `words_per_second` simulation parameter, defaulting to the neutral experimental setting `2.5`. The default is not claimed to be a measured property of TED speech.

Each segment begins at the previous segment's end, so the synthetic source clock is continuous and has no artificial gaps. Rounded segment durations use integer milliseconds. The CLI is:

```console
make time-data ARGS="--talk ted-jeff-dean-ai-smart"
```

Options are `--words-per-second`, `--allocation {uniform,character_weighted}`, and `--force`.

Simulated modern TED timings are **not true word-level speech alignment**. Latency derived from these artifacts must be described as **simulated source-clock latency**, or equivalent wording.

## Token Allocation

Within each segment, token completion boundaries allocate the exact interval from `start_ms` through `end_ms`:

- `uniform`: every lexical token has equal weight.
- `character_weighted`: the default; each token weight is `max(1, number of alphanumeric characters)`.

Integer cumulative allocation is deterministic and monotonic. The final token receives an explicit rounding correction to equal `segment.end_ms`. Character weighting is only a lightweight approximation that gives longer written words more interval; it is not a phoneme, TTS, audio, or forced-alignment model.

## Statistics And Validation

Artifacts report segment and token counts, clock start and end, integer duration, mean tokens per segment, and effective tokens per second. Semantic validation enforces English identity; preserved non-empty parsed segments, IDs, text, and order; valid intervals; deterministic token reconstruction; unique deterministic token IDs; contiguous global and per-segment indices; resolving source-segment references; in-segment emissions; global monotonicity; final-token/end equality; and statistics consistency.

The schema disallows unknown top-level, segment, token, timing, statistics, and provenance fields. There is no target-reference field.

## Limitations

Recovered WIT3 intervals provide caption boundaries, not observed word boundaries. Within-caption token completion is allocated heuristically. The final WIT3 caption may have a simulated fallback. Duplicate starts provide ordering but no positive duration. Punctuation removal avoids prepared boundary leakage but also removes potentially spoken punctuation distinctions. Character length is not pronunciation duration. Simulated TED clocks support controlled comparisons, not claims about real audio latency.

## Alignment Calibration Debt

M0.4 alignment remains **usable but needs tuning**. M0.6 preserves this debt without changing alignment artifacts. Before M0.8 acquisition scaling, review skip and group penalties and establish a manually checked development subset. M0.5 does not tune alignment, consume alignment costs, or interpret alignment cost as confidence.
