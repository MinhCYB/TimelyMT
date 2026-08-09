# Monotonic English-Vietnamese Segment Alignment

## Purpose

M0.4 aligns independently parsed English and Vietnamese segment sequences into provider-neutral bilingual units. Its intermediate contract is [`schemas/aligned-transcript.schema.json`](../schemas/aligned-transcript.schema.json), using JSON Schema Draft 2020-12 and schema version `1.0.0`.

M0.4 alignment is not token timing simulation and does not produce a canonical streaming talk. Parsed inputs remain unchanged.

## Input And Output

The aligner reads:

```text
data/streaming/parsed/<talk_id>/source.en.json
data/streaming/parsed/<talk_id>/target.vi.json
```

It writes:

```text
data/streaming/aligned/<talk_id>/alignment.json
data/streaming/aligned/<talk_id>/review.tsv
```

`alignment.json` records source and target artifacts, method parameters, bilingual alignment units, explicitly unaligned IDs, statistics, and input checksums. Each bilingual unit preserves referenced original text and has a deterministic ID such as `a-000001`. `review.tsv` is a derived UTF-8 debugging view and is not canonical data.

## Monotonic Model

Dynamic-programming state `(i, j)` means that `i` English and `j` Vietnamese segments have been consumed. Transitions never move backward, so output order is monotonic. With default `max_group_size = 3`, bilingual transitions include every bounded combination from 1:1 through 3:3, including the expected 1:2, 2:1, 2:2, 1:3, and 3:1 mappings. The bounded state graph is polynomial rather than combinatorial in transcript length.

Transitions 1:0 and 0:1 skip one source or target segment with an explicit nonzero cost. Skipped segments are not serialized as fake bilingual units; their IDs are recorded under `unaligned_source_segment_ids` or `unaligned_target_segment_ids`.

## Cost Function

The implementation consistently uses lower cost as better. It does not report statistical confidence.

For each bilingual candidate group, scoring uses:

- Length cost: absolute log distance between the candidate target/source normalized-character ratio and the talk-level ratio. The ratio is estimated independently for every talk and does not assume Vietnamese is longer or shorter.
- Position cost: distance between source and target group midpoints measured by cumulative normalized-character progress.
- Timing cost: when both parsed languages contain enough original `start_ms` values, normalized timestamp-position distance is added. Untimed TED transcripts work without this feature and store it as null.
- Group penalty: favors simpler groups, with an additional fixed penalty when both sides contain multiple segments. Its per-extra-segment weight is explicit and persisted.
- Annotation penalty: annotation-only groups such as `(Music)` and `(Applause)` can align cheaply with matching annotations, while annotation-to-speech matches are discouraged.

Scoring-only text normalization uses Unicode NFKC, lowercase text, collapsed whitespace, and removal of punctuation. Annotation-only segments receive a small stable length. Original parsed text is never modified, and alignment `source_text` and `target_text` reconstruct the original referenced segments with single spaces between segments.

The frozen M0 configuration is read from `configs/data/alignment.json`. It records `max_group_size`, `skip_penalty`, and `group_penalty`; CLI values can explicitly override them. Alignment cost remains a path optimization cost and must not be interpreted as correctness or confidence.

## Dynamic Programming

The aligner computes the minimum cumulative path cost over the bounded transition graph and backtracks from the terminal state. Transition iteration and strict replacement rules make equal-cost paths deterministic. Alignment IDs are assigned only to bilingual steps after backtracking, in monotonic order.

Run one pilot:

```console
make align-data ARGS="--talk ted-jeff-dean-ai-smart"
```

Existing output is protected by default. Use `--force` to regenerate it. Optional CLI controls are `--max-group-size`, `--skip-penalty`, `--group-penalty`, and `--config`.

## Validation

Semantic validation checks matching talk IDs, English source and Vietnamese target languages, reference resolution, contiguous groups, monotonic order, deterministic unique alignment IDs, exact text reconstruction, and statistics. Every parsed segment must occur exactly once either in one bilingual alignment unit or in its side's ordered unaligned list. Duplicate consumption, overlap between aligned and unaligned IDs, and silent omission are rejected.

## Human Review

Review `review.tsv` at the beginning, middle, and end of each talk and sort or inspect `score` for the largest costs. A structurally valid path is not evidence of linguistic correctness. Pilot observations and the decision about future semantic scoring are recorded in [`alignment-review.md`](alignment-review.md).

## Limitations

The v1 scorer has no bilingual lexical or semantic knowledge. Similar lengths and positions can still connect unrelated sentences after a local omission, and short segments are especially ambiguous. Segmentation errors can propagate into larger groups. Annotation detection is intentionally shallow and does not translate annotation labels between languages.

If manual review shows systematic semantic mismatches that structural tuning cannot address, a later version may add multilingual MiniLM or LaBSE similarity. No neural model, Hugging Face dependency, translator, timing completion, streaming tokens, baseline, pseudo-label, or policy is part of M0.4.
