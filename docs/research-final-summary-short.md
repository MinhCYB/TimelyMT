# TimelyMT Research Summary: Short Version

## Problem

- English-to-Vietnamese streaming translation for technical talks.
- Frozen EnViT5 produces source-only candidate translations.
- Research question: when should the policy LISTEN, and when should it irreversibly COMMIT?
- At fewer than four source tokens: WAIT, with no translation or policy inference. At four or more: translate, predict `p(COMMIT)`, then LISTEN below threshold or COMMIT at/above it. A 48-token span and talk end force commitment.

## Approach

| Component | Fixed role |
|---|---|
| Source stream | English tokens arriving on a simulated source clock |
| Translator | Frozen source-only EnViT5, deterministic greedy decoding |
| Policy | Causal LISTEN/COMMIT probability from current candidate, history, and numeric stability features |
| Output | Irreversible committed Vietnamese units |

No target reference, alignment, future source, or prepared document is sent to EnViT5.

## V2

| Variant | Inputs |
|---|---|
| P0 | Current source + 11 numeric features |
| P1 | P0 + previous committed source |
| P2 | P1 + previous generated target |

- Frozen MiniLM: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensions, masked mean pooling, L2 normalization.
- P2 input: `3 x 384 + 11 = 1163` dimensions.
- Historical selected configuration: `v2_P2_0.50`.
- V2 is a post-hoc exploratory DEV extension; DEV did not establish V2 quality superiority over V1.

## P3

P3_GLOBAL adds talk-level prepared context to the policy, not the translator.

| P3 input | Dimensions |
|---|---:|
| Current source, previous source, previous target | 1,152 |
| Prepared global context | 384 |
| Numeric features | 11 |
| **Total** | **1,547** |

- Eligible source requirements: `SAFE_PRETALK_CONFIRMED`, pre-talk availability, no transcript, no reference.
- Coverage: 5 TRAIN talks, 1 DEV talk, 8 eligible sources. Empty pools are valid zero vectors.
- P3 is a newly trained MLP, so P3-vs-P2 is not a causal prepared-context test.

## Controlled Ablation

Same P3 checkpoint, threshold, translator, source stream, state, and features:

| Condition | Difference |
|---|---|
| REAL_CONTEXT | Normal 384-dimensional prepared-global vector |
| ZERO_CONTEXT | Only prepared vector slice `[1152:1536]` is exact float32 zeros |

Empty-context integrity control: Jeff and Luis produced identical predictions, commits, and commit artifacts in REAL and ZERO modes.

Aggregate DEV delta, REAL minus ZERO:

| Threshold | Delta BLEU | Delta chrF2 | Delta AL | Delta LAAL | Delta commits |
|---:|---:|---:|---:|---:|---:|
| 0.30 | -0.03 | -0.02 | -2.29 | -2.29 | 2 |
| 0.40 | -0.01 | -0.07 | 2.23 | 2.22 | 8 |
| 0.50 | -0.22 | 0.04 | 0.51 | -2.07 | 18 |
| 0.60 | 0.21 | 0.14 | 0.38 | -2.60 | 19 |
| 0.70 | -0.44 | -0.08 | -3.88 | -9.82 | 44 |

## Main Finding

- Prepared context causally changes the trained P3 policy's commit behavior.
- On the only context-bearing DEV talk, Sims, REAL context consistently produced more commits and shorter segments.
- The most interesting descriptive case is Sims at 0.60: `+0.95` BLEU, `+0.63` chrF2, and 318 REAL versus 299 ZERO commits.
- Results reverse at other thresholds. This is policy sensitivity, not robust prepared-context superiority.

## Limitations

- One context-bearing DEV talk out of three; eight eligible sources total.
- Coarse global averaging can dilute useful local information.
- P3 was not trained to force context use.
- Existing artifacts omit every LISTEN-step probability and candidate translation.
- No TEST result or new winner claim.

## Next Direction

1. Add full causal timestep tracing for a research demo.
2. Investigate retrieval/local prepared context and context-aware training only after trace inspection.
3. Expand genuinely pre-talk context coverage before any later held-out evaluation.
