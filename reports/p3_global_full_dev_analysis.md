# P3_GLOBAL Full DEV Analysis

## Executive Verdict

**D. Results are mixed; the highest-information next experiment is a controlled prepared-context ablation before any TEST work.** P3_GLOBAL and P2 are separately trained policies, so their comparison is not a clean causal test of the prepared-context vector.

## Experiment Integrity

- DEV only. No TEST artifact was read; no rollout, translation generation, training, or metric evaluation was run.
- Expected and found prediction artifacts: P3 15/15; P2 15/15 (30 relevant artifacts total).
- All checked prediction spans are contiguous, cover the complete source token sequence, and have nondecreasing commit times. Both observed artifact versions are `1.0.0`.
- No missing/duplicate relevant grid artifact, orphan metric row, incompatible artifact version, source-span failure, or unexpected commit reason was found. Prepared-context provenance exists only for Sims, as expected.

## DEV Context Coverage

Eligibility is established only by `SAFE_PRETALK_CONFIRMED`, `available_before_talk`, and no transcript/reference use in prepared-context artifacts. Empty manifest source lists are not treated as context.

| talk_id | has_prepared_context | eligible_source_ids | prepared_embedding_norm |
|---:|---:|---:|---:|
| ted-jeff-dean-ai-smart | no | none | 0 |
| ted-luis-von-ahn-crowdsourcing | no | none | 0 |
| ted-sims-witherspoon-ai-climate | yes | deepmind-wind-energy-2019-lead | 1 |

Exactly 1/3 DEV talks has eligible prepared context: Sims (one official pre-talk DeepMind article). Across all five P3 thresholds, prediction provenance matches this eligibility: Jeff and Luis use the exact zero prepared-global vector (norm 0).

## Metric Sources and Schema

**A. Direct official metrics:** existing aggregate metric JSON rows provide BLEU, chrF2, token-level AL/LAAL, commit count, commits/100 source tokens, forced-commit rate, mean first-commit latency/tokens, and mean/median unit duration. Per-talk official rows contain BLEU and chrF2 only.
**B. Derived statistics:** commit spans, reason counts, and inter-commit intervals below are computed by reading existing prediction `commits` arrays. No BLEU/chrF/AL/LAAL was recomputed.
**C. Interpretations:** trade-off, Pareto, and context statements are descriptive inferences, not causal estimates.
**D. Unavailable:** complete LISTEN-step traces, all p(COMMIT) values, candidates, and policy states are absent; only commit-time probabilities/features are stored.

## P3 Threshold Results

| thr | BLEU | chrF2 | AL | LAAL | commits | c/100tok | forced | mean obs pos | mean unit ms | first ms | first tok | mean span | med span |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 25.21 | 61.96 | -1.24 | 2.63 | 2033 | 24.37 | 0.001 | 1511.41 | 1243.77 | 1154.67 | 4 | 4.1 | 4 |
| 0.40 | 25.09 | 61.7 | 0.43 | 9.81 | 1997 | 23.94 | 0.0005 | 1513.32 | 1274.14 | 1154.67 | 4 | 4.18 | 4 |
| 0.50 | 25.65 | 61.92 | 7.54 | 23.54 | 1859 | 22.28 | 0.0011 | 1515.39 | 1389.98 | 1154.67 | 4 | 4.49 | 4 |
| 0.60 | 26.1 | 61.29 | 0.77 | 48.55 | 1542 | 18.48 | 0.0097 | 1524.25 | 1755.43 | 1154.67 | 4 | 5.41 | 4 |
| 0.70 | 28.32 | 61.98 | 19.16 | 80.25 | 1005 | 12.05 | 0.0706 | 1496.75 | 2926.22 | 1154.67 | 4 | 8.3 | 4 |

## P2 Threshold Results

| thr | BLEU | chrF2 | AL | LAAL | commits | c/100tok | forced | mean obs pos | mean unit ms | first ms | first tok | mean span | med span |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 25.37 | 61.4 | -6.14 | 15.26 | 1913 | 22.93 | 0.001 | 1480.94 | 1348.9 | 1154.67 | 4 | 4.36 | 4 |
| 0.40 | 25.16 | 61.17 | -17.89 | 17.41 | 1727 | 20.7 | 0.0017 | 1462.58 | 1535.52 | 4564.67 | 12.67 | 4.83 | 4 |
| 0.50 | 25.99 | 61.46 | -19.42 | 19.59 | 1474 | 17.67 | 0.0027 | 1443.77 | 1868.17 | 4564.67 | 12.67 | 5.66 | 4 |
| 0.60 | 27.75 | 62.06 | 0.41 | 82.3 | 1031 | 12.36 | 0.0068 | 1451.29 | 2846.38 | 4564.67 | 12.67 | 8.09 | 4 |
| 0.70 | 30.04 | 62.74 | -6.43 | 66.23 | 659 | 7.9 | 0.085 | 1434.21 | 4676.36 | 7817 | 20.33 | 12.66 | 4 |

## Same-Threshold P3 vs P2

Deltas are **P3 - P2**. Positive BLEU/chrF2 favors P3 quality; negative AL/LAAL and negative commit count mean lower P3 latency measures or fewer P3 commits, respectively.

| thr | dBLEU | dchrF2 | dAL | dLAAL | d commits |
|---:|---:|---:|---:|---:|---:|
| 0.30 | -0.15 | 0.56 | 4.89 | -12.63 | 120 |
| 0.40 | -0.08 | 0.53 | 18.32 | -7.6 | 270 |
| 0.50 | -0.34 | 0.47 | 26.96 | 3.95 | 385 |
| 0.60 | -1.65 | -0.77 | 0.36 | -33.75 | 511 |
| 0.70 | -1.72 | -0.76 | 25.58 | 14.01 | 346 |

## Commit Behavior

Pooled across three talks; quartiles use linear interpolation. `other forced` means any reason other than `policy` or `talk_end`.

| variant | thr | commits | mean | median | Q1/Q3 | min/max | % 4 | % 4-5 | % >=8 | policy/end/other | mean interval ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 | 0.30 | 2033 | 4.1 | 4 | 4/4 | 2/34 | 97.15% | 98.13% | 0.84% | 2031/2/0 | 1642.23 |
| P3 | 0.40 | 1997 | 4.18 | 4 | 4/4 | 3/26 | 95.49% | 97.15% | 1.75% | 1996/1/0 | 1671.88 |
| P3 | 0.50 | 1859 | 4.49 | 4 | 4/4 | 2/44 | 92.42% | 94.78% | 3.39% | 1857/2/0 | 1796.19 |
| P3 | 0.60 | 1542 | 5.41 | 4 | 4/4 | 3/48 | 86.64% | 89.36% | 7.85% | 1527/3/12 | 2166.17 |
| P3 | 0.70 | 1005 | 8.3 | 4 | 4/4 | 1/48 | 77.51% | 80.80% | 15.72% | 934/3/68 | 3327.08 |
| P2 | 0.30 | 1913 | 4.36 | 4 | 4/4 | 2/39 | 93.73% | 96.08% | 2.35% | 1911/2/0 | 1745.41 |
| P2 | 0.40 | 1727 | 4.83 | 4 | 4/4 | 1/43 | 86.74% | 90.16% | 6.08% | 1724/3/0 | 1927.79 |
| P2 | 0.50 | 1474 | 5.66 | 4 | 4/4 | 1/48 | 79.72% | 84.06% | 12.01% | 1470/3/1 | 2259.35 |
| P2 | 0.60 | 1031 | 8.09 | 4 | 4/6 | 3/48 | 68.19% | 72.26% | 22.50% | 1024/3/4 | 3232.98 |
| P2 | 0.70 | 659 | 12.66 | 4 | 4/15 | 4/48 | 54.17% | 58.57% | 35.81% | 603/3/53 | 5051.45 |

The full DEV Sims 0.50 check is confirmed but does not generalize uniformly: P3 has 1,859 versus P2's 1,474 total commits at 0.50, with pooled mean spans 4.49 versus 5.66 and exact-4 fractions shown above. P3 is more finely segmented at 0.50 overall.

## Context-Bearing vs Empty-Context Talks

Per-talk quality is the official per-talk BLEU/chrF2. Segmentation is derived from each prediction artifact. Sims is context-bearing; Jeff and Luis are empty-context.

| thr | context | talk | BLEU P3/P2 | chrF2 P3/P2 | dBLEU | dchrF2 | commits P3/P2 | mean span P3/P2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | empty | ted-jeff-dean-ai-smart | 25.61/26.51 | 63.93/63.88 | -0.91 | 0.05 | 831/780 | 4.09/4.35 |
| 0.30 | empty | ted-luis-von-ahn-crowdsourcing | 24.7/24.78 | 61.07/59.95 | -0.08 | 1.12 | 790/735 | 4.12/4.43 |
| 0.30 | context | ted-sims-witherspoon-ai-climate | 24.27/24.24 | 59.97/59.49 | 0.03 | 0.47 | 412/398 | 4.1/4.24 |
| 0.40 | empty | ted-jeff-dean-ai-smart | 25.39/27.1 | 63.99/63.94 | -1.71 | 0.05 | 822/696 | 4.13/4.88 |
| 0.40 | empty | ted-luis-von-ahn-crowdsourcing | 24.38/24.57 | 60.49/59.61 | -0.19 | 0.88 | 780/669 | 4.18/4.87 |
| 0.40 | context | ted-sims-witherspoon-ai-climate | 24.27/22.57 | 59.71/58.93 | 1.7 | 0.77 | 395/362 | 4.28/4.67 |
| 0.50 | empty | ted-jeff-dean-ai-smart | 26.72/27.46 | 64.02/63.55 | -0.75 | 0.46 | 787/573 | 4.32/5.93 |
| 0.50 | empty | ted-luis-von-ahn-crowdsourcing | 25.04/26.19 | 60.88/60.89 | -1.15 | -0 | 713/581 | 4.57/5.61 |
| 0.50 | context | ted-sims-witherspoon-ai-climate | 24.65/22.9 | 59.97/58.73 | 1.75 | 1.24 | 359/320 | 4.7/5.28 |
| 0.60 | empty | ted-jeff-dean-ai-smart | 27.27/28.48 | 63.92/63.62 | -1.21 | 0.29 | 646/388 | 5.26/8.75 |
| 0.60 | empty | ted-luis-von-ahn-crowdsourcing | 26.08/27.53 | 60.18/61.41 | -1.45 | -1.23 | 578/408 | 5.64/7.99 |
| 0.60 | context | ted-sims-witherspoon-ai-climate | 23.81/26.7 | 58.5/60.44 | -2.89 | -1.94 | 318/235 | 5.31/7.19 |
| 0.70 | empty | ted-jeff-dean-ai-smart | 29.94/31.47 | 64.43/65.18 | -1.53 | -0.75 | 435/200 | 7.81/16.98 |
| 0.70 | empty | ted-luis-von-ahn-crowdsourcing | 28.09/29.44 | 60.8/61.59 | -1.35 | -0.79 | 344/299 | 9.47/10.9 |
| 0.70 | context | ted-sims-witherspoon-ai-climate | 25.52/28.31 | 59.62/60.38 | -2.8 | -0.75 | 226/160 | 7.47/10.56 |

At 0.50, P3's quality change on Sims is +1.75 BLEU/+1.24 chrF2, while Jeff is -0.75/+0.46 and Luis is -1.15/-0.00. Thus P3's aggregate 0.50 chrF gain coexists with lower aggregate BLEU and higher commit count. Across thresholds, Sims does not show a uniformly larger P3 advantage: at 0.60 it declines sharply relative to P2. Empty-context P3 runs also differ materially from P2, demonstrating that a separately trained P3 policy changes behavior even under a zero context vector.

## Quality / Latency Trade-Off

Within both grids, higher thresholds generally reduce commit frequency and increase mean unit duration, but official AL and LAAL are not monotonic in every adjacent step. P3 BLEU rises overall from 25.21 to 28.32, with a 0.30-to-0.40 dip; chrF2 is nonmonotonic and peaks at 0.70 by a small margin. P3's AL drops anomalously from 7.54 at 0.50 to 0.77 at 0.60 despite longer units, then rises to 19.16 at 0.70. P2 has even stronger nonmonotonicity, including negative AL at 0.50 and high LAAL at 0.60/0.70. These are official metrics, not recomputed here.
A potentially useful descriptive P3 region is 0.50-0.60: 0.50 retains the highest P3 chrF2 before 0.70, while 0.60 reduces commits by 317 with +0.45 BLEU but lower chrF2 and much higher LAAL. That is a trade-off, not an automatic choice.

## Comparison to Frozen V2 P2 0.50

The historical frozen selection remains `v2_P2_0.50`; this report does not alter it. Deltas below are P3 threshold minus that frozen point.

| P3 thr | dBLEU | dchrF2 | dAL | dLAAL | d commits |
|---:|---:|---:|---:|---:|---:|
| 0.30 | -0.78 | 0.5 | 18.17 | -16.96 | 559 |
| 0.40 | -0.91 | 0.25 | 19.84 | -9.78 | 523 |
| 0.50 | -0.34 | 0.47 | 26.96 | 3.95 | 385 |
| 0.60 | 0.1 | -0.17 | 20.19 | 28.96 | 68 |
| 0.70 | 2.33 | 0.52 | 38.57 | 60.66 | -469 |

No P3 point strongly dominates frozen P2 0.50 on the four-objective definition: higher-threshold P3 points improve BLEU but worsen LAAL; lower-latency P3 0.30/0.40 lose BLEU and add commits. P3 0.50 has slightly lower BLEU, higher chrF2, higher AL/LAAL, and 385 more commits, making it an interesting quality-mix trade-off rather than a dominance result.

## Prepared-Context Evidence

1. **Aggregate DEV:** P3 has mixed same-threshold changes against P2; the aggregate results do not isolate context effects.
2. **Context-bearing Sims:** at 0.50 Sims improves descriptively under P3, but the direction changes across thresholds, including a large P3 deficit at 0.60.
3. **Empty-context talks:** P3 differs from P2 on Jeff and Luis despite zero prepared context, so policy retraining itself clearly changes segmentation and quality.
4. **Coverage:** only one context-bearing DEV talk exists. This is not sufficient to establish that prepared context improves quality or latency.
5. **Research value:** the result is descriptive evidence worth following up, not evidence of a prepared-context benefit. Further research is justified only through a controlled ablation.

## Causal Limitations

P3_GLOBAL was trained from scratch as a new MLP, not P2 with one inference feature added. Its 1,547 inputs include four 384-dimensional embeddings (current source, previous committed source, previous generated target, prepared global context) and 11 numeric features. Prepared representation is `prepared-global-v0`; it affects the policy only. The translator is the frozen source-only EnViT5. Therefore P3/P2 differences are policy/segmentation differences under the same translator architecture, but may arise from different learned policy parameters, the prepared-context feature, or their interaction. They cannot be attributed solely to the prepared vector.
A stronger causal test is the same trained P3 architecture evaluated with real prepared context versus its prepared-context input zeroed/removed under a controlled design. This report does not implement that experiment.

## Pareto Observations

Criterion: Higher BLEU and chrF2; lower AL and LAAL; strong dominance requires no worse value on all four and strict improvement on at least one.

Strongly dominated points across the ten P3/P2 grid points: P3 0.40, P3 0.70, P2 0.60.
Not strongly dominated under this strict four-objective criterion: P3 0.30, P3 0.50, P3 0.60, P2 0.30, P2 0.40, P2 0.50, P2 0.70.
The permissive frontier is a consequence of conflicting BLEU, chrF2, AL, and LAAL; it should not be read as equivalence or selection guidance.

## Trace Limitations

Prediction artifacts provide commit-time fields, not complete LISTEN-step traces. Consequently the report can describe realized commit spans, times, reasons, and commit-time p(COMMIT), but cannot characterize threshold crossings at every LISTEN step, candidate translations, or latent policy-state trajectories. Threshold-behavior interpretations are necessarily limited to realized commits and aggregate official metrics.

## Anomalies

No artifact-integrity anomaly was detected in the scoped P3/P2 DEV grid. The nonmonotonic official latency/quality behavior is reported as a research observation, not an artifact corruption finding.

## Recommendation

**D. Results are mixed. The next recommended step is a controlled prepared-context ablation before TEST:** evaluate the same trained P3_GLOBAL policy on the same DEV talks with real prepared context and with the prepared-global input zeroed/removed, preserving all other inputs and conditions. This directly addresses both the from-scratch-policy confounding and the 1/3 context-coverage limitation as far as the current DEV set allows. Do not freeze a new winner from this analysis.
