# TimelyMT Dataset Quality Report

## M0.8 Status

Dataset v1 is frozen from 17 accepted EN-to-VI TED talks. Gate A passed after the assistant-assisted alignment review approved by the researcher. Gate B corpus scaling passed without a fallback source: 17 accepted talks exceeds the 15-talk desirable minimum.

## Alignment Calibration

The finalized `data/review/alignment-calibration-review.tsv` imported all 75 immutable calibration units without metadata changes. The assistant-assisted alignment review approved by the researcher contains 72 correct, 3 incorrect, and 0 questionable units. The original requested review summary was 72 correct, 2 incorrect, and 1 questionable; neighboring source and target evidence resolved `ted-yejin-choi-ai-smart-stupid/a-000049` as a confirmed 4:1 correction.

`vi-000048` translates `en-000057`; `vi-000049` starts with `en-000058` and continues through `en-000061`. The reviewed preferred boundary is therefore `en-000058..en-000061` to `vi-000049`.

| Configuration | Exact reviewed matches | Correct preserved | Incorrect corrected | Correct broken | Skipped source/target | Grouped units | Long-range drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All 9 max-group-size 3 grid points | 72/74 | 72 | 0/3 | 0 | 0/0 | 24 | 0 |
| Controlled max-group-size 4 comparison | 74/74 | 72 | 3/3 | 0 | 0/0 | 24 | 0 |

All penalty combinations tied on the reviewed measures. The frozen configuration keeps the prior conservative penalties and only changes the evidence-required cap: `max_group_size=4`, `skip_penalty=1.6`, and `group_penalty=0.65`. It is selected on reviewed boundary quality, not alignment cost. The cap remains bounded at four; arbitrary large N:M groups are unsupported.

## Corpus And QA

All 21 curated TED candidates were attempted through the resumable public-page adapter. 17 talks passed acquisition, parsing, frozen-config alignment, timing, canonical construction, and semantic validation. Four candidates remain visible in `data/manifests/acquisition-results.jsonl` and `outputs/dataset/quality-report.json`:

- `ted-janelle-shane-ai-danger`: partial acquisition; Vietnamese reference unavailable.
- `ted-fei-fei-li-human-centered-ai`: TED page returned 404.
- `ted-timnit-gebru-ai-harms`: TED page returned 404.
- `ted-daniela-rus-robots`: TED page returned 404.

No TED video was acquired. All accepted talks use simulated source-clock timing because public TED transcript pages do not expose subtitle timing.

| Measure | Value |
| --- | ---: |
| Accepted talks | 17 |
| Source segments | 2471 |
| Target segments | 2372 |
| Alignment units | 2323 |
| Lexical stream tokens | 41739 |
| Source-clock duration | 16695600 ms |
| Provider | TED: 17 |
| Timing mode | simulated: 17 |
| Structural QA flags | none |

Domain distribution: `ai_ethics` and `ai_ml` have 2 talks each; `ai_climate`, `ai_deep_learning`, `ai_education`, `ai_robotics_design`, `ai_safety`, `autonomous_systems`, `computer_science`, `computer_vision`, `llm_chatgpt`, `llm_reasoning`, `nlp_language`, `nlp_language_models`, and `robotics` have 1 each.

Per-talk speaker, domain, provider, timing mode, segment counts, alignment distribution, unaligned counts, diagnostic costs, checksums, and flags are machine-readable in `outputs/dataset/quality-report.json`. Alignment cost is a structural optimization diagnostic, not confidence.

Lightweight review sampling covers the existing calibration beginning/middle/end units and the highest-cost grouped records from every accepted talk in `outputs/dataset/qa-samples-highest-cost.json`. Representative high-cost grouped records were coherent segment merges. The highest-cost unusual case is `ted-joseph-redmon-computer-vision/a-000030` (4:1, cost 2.256758), where the target omits the final real-time-video sentence; it is retained as a review warning rather than structurally rejected. No accepted talk has skipped source or target segments.

## Manifest, Split, And Freeze

The final manifest is `data/manifests/streaming-dataset.json`, checksum `6730be08eff2ea874aad693e195ff05488a9b2222902f23e6e83c88e3afb2cce`. It contains unique talk IDs and canonical paths, EN-to-VI direction, canonical content checksums, causal streams, no target-in-stream leakage, and complete speaker metadata.

The persisted speaker-aware split is `data/splits/experimental.json`: 12 train (70.6%), 3 dev (17.6%), and 2 test (11.8%) talks. Speaker leakage check: passed. The calibration pilots Alona Fyshe, Jeff Dean, and Yejin Choi are listed in `test_exclusions` and absent from final test; Jeff Dean is in dev and the other two are in train.

## Known Limitations

- TED timing is simulated source-clock timing, not word-level acoustic alignment.
- The deterministic aligner has no bilingual lexical or semantic model.
- Grouped boundaries and short segments can have high diagnostic cost.
- The corpus has one provider and 17 talks; availability can change on future acquisitions.
- Dataset v1 must not be silently changed based on later model results.
