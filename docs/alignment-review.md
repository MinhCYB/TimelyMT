# M0.4 Pilot Alignment Review

## Review Method

The three available TED pilots were aligned with `monotonic_length_position_dp` version `1.0.0`, `max_group_size = 3`, and `skip_penalty = 1.6`. Semantic validation passed for every artifact. Manual review covered the first 10 units, approximately 10 middle units, the last 10 units, and the six highest-cost units of each talk. Because lower cost is better, highest-cost rather than lowest-cost units are the potentially problematic cases.

Structural validity does not establish translation quality. The observations below concern segment correspondence, not the linguistic quality of TED's Vietnamese reference.

## Jeff Dean

- Parsed segments: 160 EN, 154 VI.
- Alignment units: 153.
- Types: 145 `1:1`, 1 `1:2`, 7 `2:1`; all other types zero.
- Unaligned: 0 EN, 0 VI.
- Costs: mean 0.146505, median 0.086785, minimum 0.006927, maximum 1.287263.

Beginning and middle samples were consistently corresponding. For example, the English computer-vision sentence plus its following explanation align as `2:1` to a Vietnamese segment where punctuation did not create the same boundary. The end-of-talk Q&A also remains in order.

Highest-cost examples were still plausible:

- Cost 1.287263, `2:1`: `JD: Thank you.` plus `(Applause)` aligns to `JD: Xin cảm ơn (Tiếng vỗ tay)`.
- Cost 0.909819, `2:1`: two English sentences describing a single-task neural network align to one Vietnamese segment containing both statements.
- Cost 0.770077, `1:2`: one long English Q&A sentence aligns to two Vietnamese sentences.

Observed issue: translated annotation labels and legitimate grouped units are penalized heavily, so absolute cost is not calibrated as an error probability.

## Yejin Choi

- Parsed segments: 133 EN, 118 VI.
- Alignment units: 117.
- Types: 102 `1:1`, 1 `1:2`, 12 `2:1`, 2 `3:1`; all other types zero.
- Unaligned: 0 EN, 0 VI.
- Costs: mean 0.214147, median 0.114888, minimum 0.004767, maximum 1.591813.

Beginning, middle, and ending samples retained the expected discourse sequence. The final `(Applause)` aligns to `(Vỗ tay)`. The larger EN groups reflect Vietnamese punctuation/segmentation differences rather than detected reordering.

Highest-cost examples were plausible but expose calibration limits:

- Cost 1.591813, `3:1`: three English sentences introducing “choose your battles” align to one long Vietnamese segment containing the same transition and question.
- Cost 1.345803, `3:1`: the common-sense challenge and dark-matter analogy are split into three English segments but one Vietnamese segment.
- Cost 0.849199, `2:1`: the GPT-4 “30 hours” answer and `Not good.` align to one Vietnamese segment.

Observed issue: the group-size penalty makes valid `3:1` units dominate the worst-cost list. The raw Vietnamese transcript also has inconsistent terminal punctuation, which creates unavoidable segmentation asymmetry.

## Alona Fyshe

- Parsed segments: 127 EN, 126 VI.
- Alignment units: 126.
- Types: 125 `1:1`, 1 `2:1`; all other types zero.
- Unaligned: 0 EN, 0 VI.
- Costs: mean 0.138704, median 0.108818, minimum 0.014810, maximum 0.812605.

Beginning, middle, and ending samples were consistently corresponding. The sole `2:1` unit combines two English statements about neural networks not existing in the physical world with one Vietnamese segment containing both. The closing laughter, thanks, and applause remain correctly ordered.

Highest-cost examples included the valid `2:1` unit at 0.812605 and short `1:1` questions such as `How can we test that?` at 0.608744. Short groups have unstable length ratios even when their correspondence is clear.

Observed issue: length-ratio cost is noisy for very short segments. No obvious path drift was found in the reviewed samples.

## Decision Gate

Conclusion: **B. Usable but needs tuning.**

The deterministic aligner is structurally complete and produced plausible monotonic correspondence throughout the sampled beginning, middle, end, and worst-cost regions of all three pilots. There is no current evidence that multilingual embeddings are required for these TED transcripts. However, raw cost is not a reliable review ranking across group types: legitimate `2:1`/`3:1` mappings and short sentences are overrepresented among high-cost units, while translated annotation labels cannot be recognized as equivalent.

Before adding semantic models, tune and manually reevaluate group and short-segment penalties on a small hand-checked development subset. If that review reveals semantic path drift hidden by similar lengths and positions, multilingual MiniLM or LaBSE similarity would be a justified later extension. No semantic scorer is added in M0.4.
