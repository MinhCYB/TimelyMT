# P3_GLOBAL Sims 0.50 - Single-Talk Sanity Report

## Verdict

**PASS WITH OBSERVATIONS.** The inspected DEV artifact is structurally coherent and covers the complete source stream with contiguous committed spans. Prepared context is non-zero and has concrete source/checksum provenance. Threshold 0.50 is mechanically aggressive: most commits occur at the 4-token policy-inference boundary. This is a descriptive single-talk observation, not a quality or selection result.

## Artifact Identity

| Field | Value | Evidence class |
|---|---|---|
| `artifact_version` | `1.0.0` | Explicitly stored |
| `talk_id` | `ted-sims-witherspoon-ai-climate` | Explicitly stored |
| `split` | `dev` | Explicitly stored |
| `strategy` | `p3_global_0.50` | Explicitly stored |
| `source_token_count` | 1689 | Explicitly stored; equals summed committed spans |
| `source_final_emit_ms` | 675600 | Explicitly stored |
| Commit count | 359 | Derived from stored `commits` |

The artifact top-level schema is exactly `artifact_version`, `commits`, `prediction`, `prepared_context`, `source_final_emit_ms`, `source_token_count`, `split`, `strategy`, and `talk_id`; it has no rollout-wide trace fields. All 359 commit objects have one identical outer schema. Derived span coverage is contiguous from source index 0 through 1688, with span lengths summing to 1689; the final commit observation equals the stored final source emit time (675600 ms).

## Prepared Context

The standalone pool is `prepared-context-v0` for the same DEV talk. It contains 1 source(s), of which 1 satisfy the strict eligibility predicate. The rollout stores a `prepared-global-v0` provenance record with embedding dimension 384, source count 1, `has_eligible_context=True`, and embedding norm 1. A non-zero norm confirms this talk did not receive the zero-vector empty-context representation.

| Eligible source ID | Checksum | Type | Published | Declared leakage metadata |
|---|---|---|---|---|
| `deepmind-wind-energy-2019-lead` | `sha256:8259e773f5f353769d83bdcfe9cad54c0f503bde7e69b71371fb820e4183a4c7` | `official_article` | 2019-02-26T00:00:00Z | `available_before_talk=True`, `transcript_used=False`, `reference_used=False`, `SAFE_PRETALK_CONFIRMED` |

The artifact-level eligible IDs/checksums exactly identify the prepared source used, and the standalone pool supplies its URI and text. Under the validated eligibility rule, transcript/reference leakage is not indicated: eligible sources must be `SAFE_PRETALK_CONFIRMED`, pre-talk available, and have both use flags false. This establishes metadata-level provenance, not an independent historical audit of the external publication claim.

## Commit Schema

| Field | Type observed | Stored/derived | Meaning |
|---|---|---|---|
| `causal_features` | `dict` | Stored | Numeric component of the causal policy state at this committing observation. |
| `commit_probability` | `float` | Stored | P3 policy probability at this committing observation. |
| `observation_emit_ms` | `int` | Stored | Emit time of the observed end token, in milliseconds. |
| `observation_token_index` | `int` | Stored; also derivable from stored fields | Source token index observed when this commit was made. |
| `reason` | `str` | Stored | Streaming termination reason (`policy`, `max_length`, or `talk_end`). |
| `source_clock_duration_ms` | `int` | Stored | End-token emit time minus start-token emit time for the span. |
| `source_end` | `int` | Stored | Inclusive zero-based end index of the committed runtime source span. |
| `source_start` | `int` | Stored | Inclusive zero-based start index of the committed runtime source span. |
| `source_token_count` | `int` | Stored; also derivable from stored fields | Number of source tokens in the committed span. |
| `target_token_count` | `int` | Stored; also derivable from stored fields | Whitespace-token count of `translated_text`. |
| `translated_text` | `str` | Stored | Translator hypothesis for the full committed source span. |

`causal_features` is a stored object with these float-valued fields: `source_buffer_token_count`, `source_buffer_character_count`, `source_clock_elapsed_ms`, `current_target_token_count`, `previous_target_token_count`, `target_token_count_delta`, `previous_current_lcp_ratio`, `previous_current_change_ratio`, `prior_committed_unit_count`, `previous_committed_source_tokens`, and `previous_committed_target_tokens`. Its semantic construction is known from source code; the artifact contains it only at committed observations.

## Commit Statistics

| Statistic | P3_GLOBAL 0.50 |
|---|---:|
| Source tokens / commits | 1689 / 359 |
| Mean / median source tokens per commit | 4.7 / 4 |
| Min / Q1 / Q3 / max | 3 / 4 / 4 / 39 |
| Population SD | 3.56 |
| First / last commit observation | 952 / 675600 ms |
| Mean / median inter-commit interval | 1884.49 / 1610.5 ms |
| Commits per minute of source audio | 31.88 |
| Reasons | {'policy': 358, 'talk_end': 1} |

## Commit-Length Distribution

| Source tokens in committed span | Commit count | Percentage |
|---:|---:|---:|
| 3 | 1 | 0.28% |
| 4 | 320 | 89.14% |
| 5 | 13 | 3.62% |
| 6 | 6 | 1.67% |
| 7 | 4 | 1.11% |
| 8 | 1 | 0.28% |
| 9 | 3 | 0.84% |
| 11 | 1 | 0.28% |
| 12 | 1 | 0.28% |
| 14 | 1 | 0.28% |
| 16 | 1 | 0.28% |
| 18 | 1 | 0.28% |
| 19 | 2 | 0.56% |
| 23 | 1 | 0.28% |
| 35 | 1 | 0.28% |
| 36 | 1 | 0.28% |
| 39 | 1 | 0.28% |

At the exact 4-token minimum inference boundary: **320/359 (89.14%)**. Very short is defined here as 4-5 tokens: **334/359 (93.04%)**. Long is defined here as >=8 tokens: **15/359 (4.18%)**. This directly derives inclusive span lengths from `source_start`/`source_end`, rather than inferring behavior from the aggregate ratio alone. It supports calling 0.50 aggressive for this talk, but does not establish that it is erroneous or generalizes to DEV.

Cumulative commits by source-audio time:

| Elapsed source time | Cumulative commits |
|---:|---:|
| 0 ms | 0 |
| 60000 ms | 33 |
| 120000 ms | 70 |
| 180000 ms | 98 |
| 240000 ms | 128 |
| 300000 ms | 154 |
| 360000 ms | 177 |
| 420000 ms | 212 |
| 480000 ms | 244 |
| 540000 ms | 280 |
| 600000 ms | 314 |
| 660000 ms | 349 |
| 675600 ms | 359 |

## Representative Commits

The artifact does not retain source text for each span, so source spans below are inclusive token-index ranges; no source wording is reconstructed here.

### First 10

| Index | Time (ms) | Source span | Source tokens | Reason | p(COMMIT) | Translated unit |
|---:|---:|---|---:|---|---:|---|
| 1 | 952 | 0..3 | 4 | policy | 0.8440 | Bạn có thể đã có |
| 2 | 2636 | 4..7 | 4 | policy | 0.7450 | trải nghiệm unboxing |
| 3 | 4247 | 8..11 | 4 | policy | 0.9636 | đồ nội thất và đi qua |
| 4 | 6225 | 12..15 | 4 | policy | 0.7136 | chỉ lệnh đi tới cái gì đó |
| 5 | 7616 | 16..19 | 4 | policy | 0.9129 | như thế này Lắp ráp |
| 6 | 9301 | 20..23 | 4 | policy | 0.9647 | Tủ sách theo |
| 7 | 10812 | 24..27 | 4 | policy | 0.9260 | sơ đồ đã cung cấp Có I |
| 8 | 12670 | 28..31 | 4 | policy | 0.8760 | biết một kệ sách |
| 9 | 14827 | 32..35 | 4 | policy | 0.8931 | có vẻ như chắc chắn sẽ không |
| 10 | 16463 | 36..39 | 4 | policy | 0.7742 | đang đọc hợp ngữ |

### Middle (commits 178-182)

| Index | Time (ms) | Source span | Source tokens | Reason | p(COMMIT) | Translated unit |
|---:|---:|---|---:|---|---:|---|
| 178 | 360394 | 898..901 | 4 | policy | 0.7923 | Họ sẽ cho phép chúng tôi |
| 179 | 363046 | 902..907 | 6 | policy | 0.6190 | Thử nghiệm trên hệ thống của họ |
| 180 | 364061 | 908..911 | 4 | policy | 0.9090 | Hãy thử nghiệm trên |
| 181 | 367384 | 912..917 | 6 | policy | 0.5668 | 700 megawatts công suất năng lượng gió của họ |
| 182 | 369138 | 918..921 | 4 | policy | 0.8185 | tương đương với |

### Last 10

| Index | Time (ms) | Source span | Source tokens | Reason | p(COMMIT) | Translated unit |
|---:|---:|---|---:|---|---:|---|
| 350 | 661657 | 1650..1653 | 4 | policy | 0.9220 | sẽ làm sáng tỏ tính khả thi |
| 351 | 663017 | 1654..1657 | 4 | policy | 0.8666 | và giúp chúng tôi lái xe |
| 352 | 664220 | 1658..1661 | 4 | policy | 0.8872 | tác động So trong bạn |
| 353 | 666545 | 1662..1665 | 4 | policy | 0.7015 | những cuộc đối thoại về hành động vì môi trường tiếp theo |
| 354 | 668251 | 1666..1669 | 4 | policy | 0.6709 | Khi ai đó giới thiệu bạn |
| 355 | 669646 | 1670..1673 | 4 | policy | 0.9121 | với một điều thú vị |
| 356 | 671119 | 1674..1677 | 4 | policy | 0.9310 | Xin hãy giúp đỡ để tiến lên |
| 357 | 672669 | 1678..1681 | 4 | policy | 0.7614 | cuộc trò chuyện với |
| 358 | 674400 | 1682..1685 | 4 | policy | 0.9436 | Cảm ơn bạn |
| 359 | 675600 | 1686..1688 | 3 | talk_end | 0.8244 | Chúc mừng và vỗ tay |

## Final Prediction

The stored UTF-8-decoded prediction is 10362 Unicode characters and 2399 whitespace-delimited words (an approximation, not tokenizer tokens). Beginning: `Bạn có thể đã có trải nghiệm unboxing đồ nội thất và đi qua chỉ lệnh đi tới cái gì đó như thế này Lắp ráp Tủ sách theo sơ đồ đã cung cấp Có I biết một kệ sách có vẻ như chắc chắn sẽ không đang đọc hợp ngữ chỉ dẫn nếu tôi không làm Tôi cần t`. Ending: `ính khả thi và giúp chúng tôi lái xe tác động So trong bạn những cuộc đối thoại về hành động vì môi trường tiếp theo Khi ai đó giới thiệu bạn với một điều thú vị Xin hãy giúp đỡ để tiến lên cuộc trò chuyện với Cảm ơn bạn Chúc mừng và vỗ tay`. Vietnamese diacritics decode correctly (for example, `Bạn có thể đã có` in commit 1); the reported PowerShell mojibake is not present in the JSON text decoded as UTF-8.

## P3 vs P2 0.50

This is same-talk descriptive context only; it does not attribute differences to prepared context.

| Statistic | P3_GLOBAL 0.50 | P2 0.50 |
|---|---:|---:|
| Source tokens | 1689 | 1689 |
| Commits | 359 | 320 |
| Mean source tokens / commit | 4.7 | 5.28 |
| First / last commit time (ms) | 952 / 675600 | 952 / 675600 |
| Mean / median interval (ms) | 1884.49 / 1610.5 | 2114.88 / 1671 |
| Final prediction characters / words | 10362 / 2399 | 10114 / 2346 |
| 4-token commits | 320 (89.14%) | 266 (83.12%) |
| Very short commits (4-5) | 334 (93.04%) | 279 (87.19%) |
| Forced `talk_end` commits | 1 | 1 |
| First five spans | 4, 4, 4, 4, 4 | 4, 4, 5, 4, 4 |
| Span distribution | {3: 1, 4: 320, 5: 13, 6: 6, 7: 4, 8: 1, 9: 3, 11: 1, 12: 1, 14: 1, 16: 1, 18: 1, 19: 2, 23: 1, 35: 1, 36: 1, 39: 1} | {4: 266, 5: 13, 6: 6, 7: 4, 8: 4, 9: 3, 10: 4, 11: 3, 12: 2, 13: 1, 14: 3, 17: 1, 18: 1, 19: 1, 20: 2, 24: 1, 27: 1, 33: 2, 36: 1, 38: 1} |

## Optional P2 Threshold Context

| Threshold | Commits | Mean source tokens/commit |
|---:|---:|---:|
| 0.30 | 398 | 4.24 |
| 0.40 | 362 | 4.67 |
| 0.50 | 320 | 5.28 |
| 0.60 | 235 | 7.19 |
| 0.70 | 160 | 10.56 |

These existing Sims/P2 files have the same checked commit schema and are presented only as a descriptive threshold context, not model selection.

## Trace Limitations

The current artifact stores `p(COMMIT)` and the numeric causal feature object only for committed observations. It does **not** store every LISTEN decision, p(COMMIT) at each non-commit timestep, every timestep candidate translation, complete policy state at every timestep, or source text for each commit. Therefore it cannot support a reconstructed timestep-level trace or probability calibration analysis. A future interactive demo would need opt-in per-observation logging of candidate span/index/time, full causal state (or a documented redacted form), candidate translation, p(COMMIT), decision/reason, and a stable run/config identity, written independently from final commit artifacts.

## Findings

1. **Explicitly stored:** identity fields, full committed decision records, final joined prediction, and P3 prepared-context provenance.
2. **Mathematically derived:** 359 contiguous spans fully cover 1689 source tokens; commit-length statistics, observation-time intervals, and cumulative counts. `source_clock_duration_ms` is stored but cannot be recomputed from this artifact alone because start-token emit times are absent.
3. **Known from source semantics:** learned streaming begins policy inference at 4 source tokens, calls the source-only translator on each eligible candidate, commits on `p >= threshold` (or max-length/talk-end), and represents P3 context as eligible-source MiniLM embeddings (equal-average plus normalization only for multiple sources). The final 3-token span is below the normal inference boundary and is emitted by the post-loop `talk_end` flush; it is thus mechanically forced, even though the fallback records a probability.
4. **Not observable:** LISTEN decisions, their probabilities/candidate translations, the full timestep trajectory, original per-span source text, and independent verification of the external source's publication/availability assertion.

## Recommendation Before Full DEV

**Proceed to the full DEV grid.** No structural, provenance-metadata, or mechanical streaming anomaly in this completed single-talk artifact blocks it. Before proceeding, record that 0.50 is boundary-heavy on this talk and preserve the current artifact/report; inspect the full-grid commit-length distributions and any unexpected forced-commit patterns before interpreting results. Do not treat this report as quality evaluation or a winner selection.
