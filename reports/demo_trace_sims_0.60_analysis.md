# Sims 0.60 REAL vs ZERO Trace Analysis

## Trace Pair Integrity

Synchronized: **yes**. DEV talk `ted-sims-witherspoon-ai-climate`; threshold `0.60`; checkpoint SHA `ccf829fdb7ab521cc12c299583efa7222c965440b1257ddfb35e03ddd7bcadb9`; source tokens `1689`; final source emission `675600` ms; events `1689`.
Exact equality was required for event count and every `event_index`, `source_token_end`, and `observation_ms`; no heuristic alignment was used.

## Summary

Identical decisions: **1200/1689 (71.05%)**. Differing decisions: **489/1689 (28.95%)**.

## Divergence Statistics

REAL COMMIT / ZERO LISTEN: **47**. REAL LISTEN / ZERO COMMIT: **23**.
First divergence: event 131 at 00:53 (53000 ms). Last divergence: event 1687 at 11:15 (675035 ms).
Divergence regions: **159**. Longest: `{'start_event_index': 956, 'end_event_index': 984, 'length': 29}`.

## First Divergence

Event `131` at `53000` ms (00:53), source token end `131`, threshold `0.60`, delta p_commit `+0.055609`.
REAL: span `124..131`: “renewable energy Electrify everything else Deploy solutions that”; translation: “Năng lượng tái tạo Điện khí hoá mọi thứ khác Triển khai các giải pháp đó”; p_commit `0.631034`; `COMMIT` because `policy`.
ZERO: span `124..131`: “renewable energy Electrify everything else Deploy solutions that”; translation: “Năng lượng tái tạo Điện khí hoá mọi thứ khác Triển khai các giải pháp đó”; p_commit `0.575424`; `LISTEN` because `below_threshold`.
This is a useful demo moment because both policies observe the same source clock and candidate span, while their commit probabilities fall on opposite sides of the policy threshold. It isolates a policy timing difference; it does not establish translation improvement.

## Top Demo Moments

1. Event `584` at `03:53`; “spoke to domain experts we found out everything we could about the problem Our team which is a mix of research scientists engineers a product manager a program manager and an”; REAL `0.000001` / `LISTEN`, ZERO `0.929383` / `COMMIT`, delta `-0.929382`. `REAL_LISTEN_ZERO_COMMIT` with a threshold-straddling, readable candidate.
2. Event `588` at `03:54`; “spoke to domain experts we found out everything we could about the problem Our team which is a mix of research scientists engineers a product manager a program manager and an impact analyst decided that”; REAL `0.000001` / `LISTEN`, ZERO `0.901857` / `COMMIT`, delta `-0.901856`. `REAL_LISTEN_ZERO_COMMIT` with a threshold-straddling, readable candidate.
3. Event `1444` at `09:37`; “innovation in AI for”; REAL `0.899753` / `COMMIT`, ZERO `0.003625` / `LISTEN`, delta `+0.896128`. `REAL_COMMIT_ZERO_LISTEN` with a threshold-straddling, readable candidate.
4. Event `865` at `05:45`; “it's not every wind-farm manager that wants to let a bunch of AI researchers test on their multimillion or multibillion-dollar systems But the thing is in order to prove that AI works we have”; REAL `0.004191` / `LISTEN`, ZERO `0.899550` / `COMMIT`, delta `-0.895359`. `REAL_LISTEN_ZERO_COMMIT` with a threshold-straddling, readable candidate.
5. Event `1273` at `08:30`; “we're not going to”; REAL `0.886812` / `COMMIT`, ZERO `0.000006` / `LISTEN`, delta `+0.886806`. `REAL_COMMIT_ZERO_LISTEN` with a threshold-straddling, readable candidate.
6. Event `968` at `06:27`; “component of AI for”; REAL `0.884557` / `COMMIT`, ZERO `0.000003` / `LISTEN`, delta `+0.884554`. `REAL_COMMIT_ZERO_LISTEN` with a threshold-straddling, readable candidate.
7. Event `1011` at `06:45`; “better that AI performance”; REAL `0.881297` / `COMMIT`, ZERO `0.000258` / `LISTEN`, delta `+0.881039`. `REAL_COMMIT_ZERO_LISTEN` with a threshold-straddling, readable candidate.
8. Event `980` at `06:32`; “that can tell you”; REAL `0.874847` / `COMMIT`, ZERO `0.000047` / `LISTEN`, delta `+0.874800`. `REAL_COMMIT_ZERO_LISTEN` with a threshold-straddling, readable candidate.

## Commit Timeline

REAL: `{'total_commits': 318, 'first_commit_ms': 952, 'last_commit_ms': 675600, 'mean_commit_interval_ms': 2128.2271293375393}`.
ZERO: `{'total_commits': 299, 'first_commit_ms': 952, 'last_commit_ms': 675600, 'mean_commit_interval_ms': 2263.9194630872485}`.
First REAL-first point: event 131 at 00:53 (53000 ms). First ZERO-first point: event 410 at 02:44 (164573 ms). First different commit boundary: event 181 at 01:12 (72430 ms). Later same non-WAIT event: event 225 at 01:30 (90165 ms).

## Probability Differences

Paired probability events: `556`. Mean absolute delta: `0.125980`. Median: `0.014456`. Maximum: `0.929382` at event 584 at 03:53 (233025 ms). WAIT events remain null and are excluded.

## Candidate Translation Cascades

A. Same-span probability/decision divergences: `14`; first: event 131 at 00:53 (53000 ms).
B. Different candidate boundaries after earlier commits: `829` events; translations also differ at `626` of them; first translation difference: event 132 at 00:53 (53221 ms).
These later translation differences are downstream effects of altered source spans, not direct prepared-context input to EnViT5.

## Demo Bookmarks

- `first-divergence`: event `131` at `00:53`. First synchronized event with different REAL and ZERO policy decisions.
- `max-probability-gap`: event `584` at `03:53`. Differing decision with the largest absolute REAL-minus-ZERO commit probability gap.
- `real-commit-zero-listen`: event `131` at `00:53`. Clear threshold-straddling policy difference.
- `real-listen-zero-commit`: event `410` at `02:44`. Clear threshold-straddling policy difference.
- `commit-boundary-cascade`: event `181` at `01:12`. Both policies commit but their earlier choices yield different candidate boundaries.
- `late-talk-divergence`: event `1687` at `11:15`. Late non-forced policy difference, useful for showing the effect persists beyond the opening.

## Interpretation Guardrails

- REAL and ZERO use the same P3 checkpoint.
- Prepared context affects the policy only.
- A different commit can alter future candidate spans, causing later candidate translations to differ indirectly.
- Candidate-translation differences after boundary divergence are downstream consequences, not direct context injection into EnViT5.
- This trace is an illustrative controlled DEV case, not general evidence of superiority.
