# Dataset Manifests And Talk-Level Splits

## Purpose

M0.7 indexes completed canonical talks in the versioned dataset manifest at `data/manifests/streaming-dataset.json`. A canonical `streaming-talk.json` remains the complete offline artifact for exactly one talk. The dataset manifest is a lightweight external index: it records canonical paths, stable canonical-content checksums, descriptive metadata, timing mode, and small inspection statistics. It never embeds a complete canonical talk.

The manifest and split schemas are JSON Schema Draft 2020-12 contracts:

- `schemas/dataset-manifest.schema.json`
- `schemas/dataset-split.schema.json`

## Building And Inspecting

```console
make manifest-data
python -m timelymt.data.manifest.cli summary
```

The builder scans `data/streaming/processed/*/streaming-talk.json`, loads every artifact through canonical semantic validation, verifies that `talk_id` matches the parent directory, computes the stable canonical-content checksum, and orders entries by `talk_id`. Invalid artifacts or duplicate talk IDs fail rather than being silently indexed.

`content_checksum` excludes only canonical `provenance.processed_at`. The dataset manifest checksum likewise excludes `provenance.generated_at`, so rebuilding unchanged data produces the same identity even though the record time changes.

## External Splits And Leakage Prevention

Split membership is intentionally not a canonical-talk field. Changes to an experiment must not mutate reference data. Split files live under `data/splits/` and bind to a specific `dataset_manifest_checksum`.

All experimental splitting is at **talk level**. Never independently split sentences, segments, alignments, prefixes, stream tokens, hypotheses, pseudo-labels, policy examples, or context/history examples. These records retain `talk_id` and inherit its split:

```text
talk A = train -> every derived record with talk_id A = train
```

A derived-data pipeline must look up the persisted talk assignment and must never recalculate membership by Python `hash()` or filesystem ordering.

## Pilot Corpus

The current Jeff Dean, Yejin Choi, and Alona Fyshe talks are explicitly marked `pilot` in the dataset manifest and in `data/splits/pilot.json`. The current 3-talk pilot corpus is **NOT a final train/dev/test benchmark**. It exists to validate acquisition, parsing, alignment, timing, canonicalization, manifest construction, and future split infrastructure.

## Future Experimental Splits

After M0.8 expands the corpus, persist an approved split explicitly:

```console
python -m timelymt.data.manifest.cli split --seed 42 --train-ratio 0.7 --dev-ratio 0.15 --test-ratio 0.15 --exclude-from-test ted-jeff-dean-ai-smart --exclude-from-test ted-yejin-choi-ai-smart-stupid --exclude-from-test ted-alona-fyshe-ai-understand
```

The generator sorts group identities, uses a seeded PRNG, records its strategy, seed, grouping mode, ratios, and calibration exclusions, and writes a source-of-truth split file. It refuses a dataset below `--minimum-talk-count` (default `10`) unless `--allow-tiny-dataset` is explicitly supplied. M0 finalization must not use that override. Calibration talks are assigned only to train/dev and are rejected if they appear in test.

The default is `--group-by speaker`: all talks by one speaker stay in a single split. Speaker grouping requires speaker metadata for every indexed talk and fails clearly when it is absent. `--group-by talk` is available when speaker grouping is inappropriate. Domain/topic metadata is retained for inspection only; M0.7 does not implement domain stratification.

## Validation

Split validation checks the dataset checksum, recognized split names, valid positive ratios summing to one, known talk IDs, no duplicates, no train/dev/test overlap, full experimental membership coverage, and speaker-group isolation when speaker grouping is selected. Pilot manifests have the simpler `pilot` plus `talk_ids` contract.

## Test Hygiene

Final test talks and all prefixes or other records later derived from them remain test data. They must not be used for alignment tuning, translator selection, pseudo-label thresholds, policy thresholds, or model selection. Dataset membership and the split checksum are frozen together; later content changes require a new dataset version.
