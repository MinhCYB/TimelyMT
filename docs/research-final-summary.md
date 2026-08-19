# TimelyMT Research Summary

## 1. Problem

TimelyMT studies context-aware, low-latency English-to-Vietnamese streaming translation for technical talks. The frozen translator can produce a Vietnamese hypothesis from the English source observed so far; the research question is the timing policy:

```text
When should the system LISTEN for more source?
When should it COMMIT the current translation?
```

Committing too early can preserve an unstable prefix. Waiting longer can improve the available source context but increases delay. The objective is to characterize this quality/latency trade-off with causal streaming policies, not to retrain the translator.

## 2. Streaming System

The source stream consists of English lexical tokens arriving at their simulated source-clock emit times. A policy maintains an uncommitted candidate source span beginning immediately after its most recent commit.

- With fewer than four candidate source tokens, the system waits. It makes no translator request and no policy inference.
- At four or more candidate tokens, it first translates the current candidate source span with frozen EnViT5.
- It then constructs the causal policy state and predicts `p(COMMIT)`.
- Below the configured threshold, the decision is LISTEN; the candidate remains open and the next source token extends it.
- At or above threshold, the decision is COMMIT.
- A span at the 48-token maximum is forced to commit (`max_length`).
- At talk end, any remaining source span is flushed (`talk_end`), including a span shorter than four tokens.
- A committed Vietnamese unit is irreversible. Its source boundary and generated target become history for later policy decisions in that policy session.

The translator and timing policy remain separate. EnViT5 receives only the current source span, never prepared documents, target references, alignments, future source, or policy labels. The wrapper adds EnViT5's internal `en:` control prefix only at model preprocessing, uses pinned deterministic greedy decoding, and removes only an exact leading generated `vi:` control prefix plus at most one following ASCII space. The policy consumes the normalized candidate hypothesis and causal streaming state.

## 3. V1

V1 established the causal streaming research baseline: a frozen Dataset v1 and split, frozen source-only EnViT5, causal prefix translation artifacts, numeric streaming/stability features, and learned P0/P1/P2 LISTEN/COMMIT policies alongside fixed and local-agreement baselines. Vietnamese references are evaluation-only and never cross the translator or causal policy boundary.

V1's learned-policy selection was completed on DEV and its checkpoint stage is recorded as `dev-frozen-complete`. This is the immutable upstream supervision and experiment identity used by later V2/P3 work. V1 established the operational semantics and evaluation framework; it did not establish that a later contextual extension is superior.

## 4. V2

V2 is explicitly a post-hoc exploratory DEV extension built from the immutable V1 supervision. It replaced V1's sparse text representation with a frozen multilingual semantic encoder and a small MLP while retaining the same causal state schema and eleven numeric features.

| Variant | Semantic text inputs |
|---|---|
| P0 | Current uncommitted source + numeric features |
| P1 | P0 + previous committed source |
| P2 | P1 + previous generated target |

The frozen encoder is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e62509716f15c5fd03a6fd3156a4bc5e43f83f26`. It produces 384-dimensional embeddings using attention-mask mean pooling followed by L2 normalization. P2 therefore uses 1,152 semantic dimensions plus 11 numeric dimensions, for 1,163 inputs.

The previous selected configuration is `v2_P2_0.50`. That selection and its frozen V2 configuration remain historical facts; this summary does not alter them. Full DEV comparison did not establish V2 quality superiority over V1. V2 is a useful contextual/nonlinear extension, not a claim of an overall V1 improvement.

## 5. Why Prepared Context Was Added

Technical talks can have genuinely available information before a talk begins: project pages, articles, papers, announcements, slides, or other approved public material. The hypothesis was that such material could help the timing policy judge whether the currently observed source is stable enough to commit.

This is policy-side context only. It must be genuinely pre-talk and cannot be derived from the complete transcript, Vietnamese reference, alignments, reference-derived glossaries, TEST data, or summaries generated from the full transcript. Prepared context is immutable once a talk begins and must not change EnViT5's source-only input.

## 6. PreparedContext

Prepared context is stored as a talk-specific `PreparedContextPool`. A source is eligible only when all of the following hold:

- `classification=SAFE_PRETALK_CONFIRMED`
- `available_before_talk=true`
- `transcript_used=false`
- `reference_used=false`

Current eligible coverage is five TRAIN talks and one DEV talk. There are eight eligible sources total. Empty pools are valid and intentionally represent a talk with no eligible prepared source; they are not an error or implicit context. No TEST prepared-context pool or TEST result is involved.

The sole context-bearing DEV talk is `ted-sims-witherspoon-ai-climate`. Its eligible source is `deepmind-wind-energy-2019-lead`, an official article with recorded eligibility/provenance metadata. The two other DEV talks, Jeff and Luis, have valid empty pools.

## 7. P3_GLOBAL

P3_GLOBAL is a separate policy experiment, not a redefinition of P0/P1/P2. Its frozen input layout is:

| Input | Dimensions |
|---|---:|
| Current source | 384 |
| Previous committed source | 384 |
| Previous generated target | 384 |
| Prepared global context | 384 |
| Numeric features | 11 |
| **Total** | **1547** |

`prepared-global-v0` encodes each eligible source independently with the pinned MiniLM encoder. Sources are sorted by `source_id`; their embeddings are equally averaged and the result is L2-normalized when more than one source is present. A one-source pool uses that source embedding directly. An empty eligible pool is represented by an exact `float32` zero vector of length 384.

Prepared context affects the policy only. EnViT5 remains frozen and source-only. P3 was trained from scratch as a new MLP because the P2 checkpoint cannot consume the additional 384 dimensions. This makes P3-versus-P2 an architecture-and-training comparison, not a causal prepared-context ablation.

## 8. P3 Training

P3 training used immutable V1 TRAIN supervision:

| Training fact | Value |
|---|---:|
| TRAIN rows | 22,018 |
| LISTEN labels | 19,075 |
| COMMIT labels | 2,943 |
| Positive weight | 6.481481481481482 |

The MLP is:

```text
Linear(input, 256)
GELU
Dropout(0.20)
Linear(256, 64)
GELU
Dropout(0.10)
Linear(64, 1)
```

Checkpoint metadata records AdamW, learning rate `0.001`, weight decay `0.0001`, batch size `256`, 20 epochs, seed `20260809`, and `BCEWithLogitsLoss(pos_weight=TRAIN_LISTEN/TRAIN_COMMIT)`. The persisted checkpoint is SHA-256 validated before loading; checkpoint persistence and restoration were validated in the P3 test suite.

## 9. Full DEV P3 vs P2

The official aggregate P3 DEV metrics are:

| Threshold | BLEU | chrF2 | AL | LAAL |
|---:|---:|---:|---:|---:|
| 0.30 | 25.21 | 61.96 | -1.24 | 2.63 |
| 0.40 | 25.09 | 61.70 | 0.43 | 9.81 |
| 0.50 | 25.65 | 61.92 | 7.54 | 23.54 |
| 0.60 | 26.10 | 61.29 | 0.77 | 48.55 |
| 0.70 | 28.32 | 61.98 | 19.16 | 80.25 |

Same-threshold P3-versus-P2 results were mixed. For example, P3 at 0.50 had slightly lower BLEU, higher chrF2, higher AL/LAAL, and 385 more commits than P2 at 0.50. No P3 point strongly dominated frozen P2 0.50 across BLEU, chrF2, AL, and LAAL.

The critical interpretation is that P3 versus P2 is **not** a clean prepared-context causal ablation. P3 is separately trained from scratch, and it differs from P2 even on Jeff and Luis, where the prepared vector is exactly zero. Differences can result from learned parameters, the added feature, or their interaction.

## 10. Controlled Prepared-Context Ablation

The controlled ablation is the strongest available prepared-context experiment. It compares two inference conditions of the **same P3 checkpoint** with the same input dimensionality, threshold, translator, streaming state, and numeric/semantic features.

| Condition | Prepared-global slice |
|---|---|
| REAL_CONTEXT | Normal prepared-global vector |
| ZERO_CONTEXT | Only dimensions `[1152:1536]` replaced with exact `float32` zeros |

The source stream is the same. Each condition retains its own commits and resulting streaming history, as required by causal rollout semantics; no history is shared. The translator, P3 model, candidate-source construction, and all non-prepared inputs are unchanged.

## 11. Ablation Integrity

Empty-context invariance is an essential control. On Jeff, REAL and ZERO produced equal predictions, equal commits, and identical commit artifacts. On Luis, the same invariance held across the tested threshold grid. Since both conditions use an effective zero vector for valid empty pools, this verifies that zero mode itself does not perturb unrelated P3 behavior.

The Sims comparison can therefore be interpreted as an intervention on the prepared-global slice for the only context-bearing DEV talk, subject to the limited coverage and single-checkpoint caveats below.

## 12. Ablation Result

All deltas below are `REAL_CONTEXT - ZERO_CONTEXT`. Positive BLEU/chrF2 favors REAL quality. Negative AL/LAAL favors REAL on those latency measures. Positive commits means REAL committed more often.

| Threshold | Delta BLEU | Delta chrF2 | Delta AL | Delta LAAL | Delta commits |
|---:|---:|---:|---:|---:|---:|
| 0.30 | -0.03 | -0.02 | -2.29 | -2.29 | 2 |
| 0.40 | -0.01 | -0.07 | 2.23 | 2.22 | 8 |
| 0.50 | -0.22 | 0.04 | 0.51 | -2.07 | 18 |
| 0.60 | 0.21 | 0.14 | 0.38 | -2.60 | 19 |
| 0.70 | -0.44 | -0.08 | -3.88 | -9.82 | 44 |

On Sims specifically:

- At 0.30, quality was slightly worse with REAL context.
- At 0.40, quality was slightly worse with REAL context.
- At 0.50, BLEU was worse while chrF2 was slightly higher with REAL context.
- At 0.60, REAL context was `+0.95` BLEU and `+0.63` chrF2, with 318 REAL commits versus 299 ZERO commits.
- At 0.70, quality was substantially worse with REAL context.

Across the tested Sims thresholds, REAL context consistently caused more commits and therefore shorter source segments than ZERO_CONTEXT.

## 13. Final Scientific Conclusion

Prepared context causally affects the trained P3 policy's commit behavior, but `prepared-global-v0` does not provide a robust, threshold-independent quality/latency improvement.

The current experiment demonstrates policy sensitivity to prepared context. It does **not** demonstrate robust prepared-context superiority. The strongest positive case is Sims at threshold 0.60, but it is descriptive evidence from one context-bearing DEV talk, not a general conclusion or a reason to select a new winner.

## 14. Limitations

- Only one of three DEV talks has prepared context.
- Only eight eligible sources exist in total.
- `prepared-global-v0` is a coarse talk-level representation.
- Global equal averaging can dilute locally useful information.
- P3 training was not designed specifically to force effective use of prepared context.
- Existing rollout artifacts retain only commit-time data, not full LISTEN-step traces.
- No held-out TEST conclusion exists; TEST remains untouched.

## 15. Future Research

Prioritize the next work by expected information value:

1. Retrieval or local prepared context instead of one global average.
2. Context-aware training or regularization designed to make context use measurable and appropriate.
3. More genuinely pre-talk documents with explicit provenance.
4. More context-bearing DEV/evaluation talks.
5. Full timestep policy tracing for direct inspection of candidate evolution and decision divergences.
6. Only later, after a controlled design is justified and frozen, consider TEST.

This does not prescribe a large new experiment now. The immediate engineering need for a research demo is observability: capture real causal trace events without changing rollout behavior.
