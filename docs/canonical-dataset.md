# Canonical Streaming Dataset Builder

## Purpose

M0.6 assembles completed M0.2-M0.5 artifacts into `data/streaming/processed/<talk_id>/streaming-talk.json`. It does not acquire, parse, align, simulate timing, translate, label, train, evaluate, or create splits. The builder is provider-neutral because upstream stages already normalize provider differences.

```console
make build-data ARGS="--talk ted-jeff-dean-ai-smart"
```

The output is protected unless `--force` is supplied. Assembly and semantic validation complete before writing, so failed validation cannot leave a partial canonical artifact.

## Inputs And Ownership

| Canonical field | Source of truth |
| --- | --- |
| `talk` | M0.2 metadata and acquisition record |
| `source` | M0.3 source wording/identity plus M0.5 finalized source timing |
| `target_reference` | M0.3 Vietnamese wording/identity and original timing only when present |
| `alignments` | M0.4 bilingual units and method metadata |
| `stream` | M0.5 tokens, indices, source references, mode, parameters, emission times |
| `provenance` | Lightweight M0.2-M0.6 lineage |

The builder requires acquisition metadata/record, parsed EN/VI, alignment, and timed EN source artifacts. Missing inputs fail clearly.

## Timing, References, And Alignment

The source reference uses finalized M0.5 timing. `simulated` source clocks are explicitly distinct from real talk duration; recovered caption timing retains `recovered_from_caption_starts` and original timing-source provenance. The builder never invents target timestamps or derives them from English.

Vietnamese is gold reference data for alignment, evaluation, and analysis, never runtime input. Complete source and target arrays retain all segments. Explicit M0.4 unaligned segments have no fake `1:0` or `0:1` unit. M0.4 deterministic `score` is omitted because cost is not confidence; M0.6 neither tunes nor filters alignment units.

## Runtime Boundary

The canonical file is offline research data. A future runtime view may expose permitted metadata, source tokens emitted by the current time, observed source history, committed system target history, and approved context/glossaries. It must not expose gold targets, future source tokens, future alignments, or future timing. Closed stream-token validation rejects target fields. This milestone does not implement a runtime-view adapter.

## Validation And Determinism

Validation checks identities/languages, ordered unique segments, parsed-to-timed source identity/text/order, timing, alignment resolution/reconstruction/non-reuse/monotonicity, unaligned accounting, and stream IDs/indices/tokenization/references/emission times. TimedSource tokenization is authoritative; the builder does not retokenize the stream.

Sorted JSON serialization is deterministic. Rebuilds differ only in `provenance.processed_at`; `canonical_content_checksum` excludes that timestamp for stable comparison.

## Limitations And Future Manifests

Canonical talks contain no train/dev/test membership; future external manifests can reference `talk_id`. Alignment remains usable but needs tuning. Before M0.8 scaling, review skip and group penalties using a manually checked development subset. This is calibration debt, not a reason to reinterpret cost as confidence.
