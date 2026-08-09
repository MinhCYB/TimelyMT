# Canonical Streaming Talk Schema

## Purpose

M0.1 defines the canonical data contract for one bilingual English-Vietnamese talk. The contract is the boundary between acquisition/alignment outputs and later streaming research stages. Its machine-readable definition is [`schemas/streaming-talk.schema.json`](../schemas/streaming-talk.schema.json), using JSON Schema Draft 2020-12 and schema version `1.0.0`.

The canonical object can hold complete research references, but it is not itself a safe runtime-policy input. In particular, the Vietnamese reference and unseen source content are physically separated from `stream`, and runtime code must eventually expose a prefix-limited view rather than the full canonical object. A canonical file represents a fully processed talk; incomplete acquisition, parsing, and alignment artifacts remain under intermediate data paths and must not claim conformance to this schema.

Dataset split membership is deliberately external. Train/dev/test assignments belong under `data/splits/` and/or `data/manifests/`, so changing an experimental split never changes a canonical talk.

## Top-Level Structure

| Field | Role | Visibility |
| --- | --- | --- |
| `schema_version` | Version of the canonical contract | Processing metadata |
| `talk` | Stable identity and descriptive metadata | Conditionally runtime-visible |
| `source` | Complete English reference transcript in original segments | Reference-only as a complete object |
| `target_reference` | Complete gold Vietnamese translation | Reference-only |
| `alignments` | Links between English and Vietnamese segments | Reference-only |
| `stream` | Ordered, timestamped English simulator tokens | Prefix only at runtime |
| `provenance` | Acquisition and processing lineage | Offline processing information |

All six structural sections are required. An array may be empty only when that is the valid final value for a fully processed talk, not as a placeholder for unfinished acquisition or parsing. The schema keeps optional descriptive fields optional rather than inventing unavailable values.

## Talk Metadata

`talk.talk_id` is the stable identifier used by manifests and downstream derived artifacts. `talk.source_language` is fixed to `en`; `talk.target_language` is fixed to `vi`.

Optional fields include `title`, `speaker`, `domain`, `topics`, `provider`, `original_source_id`, `source_url`, and `duration_ms`. `duration_ms`, when known, is a non-negative integer. Provider-specific descriptive values may be retained in `talk.metadata` without weakening the core structure.

Metadata is runtime-visible only when it would realistically be known before or during a talk. For example, a public title or declared topic may be allowed, while metadata computed from the complete transcript must remain offline.

## Source Segments

`source.segments` preserves the provider's English subtitle or transcript units. Every segment contains:

- `segment_id`: identifier unique among source segments in this talk.
- `index`: deterministic, zero-based source segment order.
- `text`: complete reference text for the segment.
- `start_ms` and `end_ms`: non-negative integer timestamps in milliseconds.
- `metadata`: optional provider-specific information.

The complete `source` section is a reference transcript. It is useful for parsing, alignment, regeneration, and analysis, but exposing it at runtime would leak future source content.

## Target Reference Segments

`target_reference.segments` stores gold Vietnamese independently from both `source` and `stream`. Each segment requires `segment_id`, zero-based `index`, and `text`. Original `start_ms` and `end_ms` are optional; if one is present, both must be present.

This section exists only for alignment, evaluation, and offline analysis. Gold Vietnamese text and timing are never inputs to a commit policy.

## Alignment Model

Each item in `alignments` contains an `alignment_id`, non-empty lists of `source_segment_ids` and `target_segment_ids`, and a free-form `method` name. This representation supports 1:1, 1:N, N:1, and N:M mappings without changing source or target segmentation.

`confidence` is optional and, when present, lies in `[0, 1]`. It is omitted when an alignment method does not produce a meaningful confidence. Alignment units and their future mappings are reference-only.

## Streaming Token Model

`stream` is the source-only representation received by the simulator. Each token has:

- `token_id`: identifier unique among stream tokens in this talk.
- `index`: global zero-based stream position.
- `text`: human-readable English token text.
- `source_segment_id`: reference to the source segment that produced it.
- `segment_index`: zero-based token position within that source segment.
- `emit_ms`: simulated or observed emission time in integer milliseconds.

These tokens are human-readable lexical source units, not SentencePiece, BPE, Marian, XLM-R, or any other neural model's subwords. Whitespace is not a token, and subtitle punctuation must not become separate runtime punctuation tokens. Original punctuation remains preserved in `source.segments[].text`. A later translator adapter may tokenize an emitted source prefix for its own model without changing canonical data.

`emit_ms` is the time when the complete source token has become available to the online system. Only the token prefix whose `emit_ms` has been reached is runtime-visible. The complete `stream.tokens` array is an offline representation and includes future tokens.

## Timing Convention

All canonical durations and timestamps use non-negative integer milliseconds, named with the `_ms` suffix. Floating-point seconds are not canonical.

`stream.timing_mode` records the M0.5 timing mode used to obtain token emission times:

- `simulated`
- `recovered_from_caption_starts`

`stream.timing_parameters` retains M0.5 timing settings and makes simulated/recovered provenance explicit. This contract does not define or implement a timing-generation algorithm. Optional target timestamps preserve original provider timing only and do not make target content runtime-visible.

## Runtime Boundary

Runtime-visible information is limited to:

- Talk metadata genuinely known before or during the talk.
- Source tokens already emitted at the current simulation time.
- Source history formed only from already emitted tokens.
- Previously committed translations generated by the system.
- Explicitly permitted context or glossary information in a future contract.

Reference-only information includes:

- Future source tokens and their emission times.
- The complete future English transcript.
- All gold Vietnamese text and target timing.
- Future or complete bilingual alignment information.
- Labels, features, or stability judgments derived using future hypotheses.

The canonical file contains both offline references and the source stream for reproducibility. Downstream runtime code must not deserialize the full object as policy input; a later milestone should define a runtime-view adapter that exposes only the permitted prefix. That adapter is not part of M0.1.

## Provenance

`provenance.processing_version` is required so canonicalization behavior can be identified. Optional fields record the provider and source identifier, acquisition date, processing timestamp, and parser/alignment/timing tool names and versions. Small provider- or tool-specific details may be stored under `metadata` extension objects.

Per-unit alignment `method` and stream `timing_mode` describe the resulting data. Provenance tool records describe which process or implementation produced it. Duplication here is intentional and keeps lineage understandable without coupling the contract to a particular parser or aligner.

## Validation Invariants

JSON Schema enforces structural types, required fields, language constants, identifier syntax, integer/non-negative timestamp fields, timing modes, paired target timing, and confidence bounds. A later semantic validator must additionally enforce cross-record and ordering invariants that JSON Schema cannot express reliably:

- `schema_version` exists and is supported.
- `talk.talk_id` exists.
- Talk and source languages are English (`en`); talk and target-reference languages are Vietnamese (`vi`).
- Source segment IDs are unique within a talk.
- Target segment IDs are unique within a talk.
- Source and target segment `index` values are contiguous from zero and array order follows them.
- Every segment satisfies `end_ms >= start_ms`; source segment timestamps preserve deterministic source order.
- Alignment IDs are unique, and every alignment reference resolves to the corresponding source or target segment.
- Stream token IDs are unique.
- Stream token `index` values are contiguous from zero and array order follows them.
- Every stream token `source_segment_id` resolves to a source segment.
- Token `segment_index` values preserve token order within each source segment.
- The stream token sequence preserves source segment and transcript order.
- `emit_ms` is monotonically non-decreasing.
- Runtime stream data contains source information only.
- Gold target data and future source/alignment/timing information are never runtime-visible.

The schema intentionally does not claim to enforce uniqueness by an object's ID field, reference resolution, `end_ms >= start_ms`, or monotonic sequences. Those require validation across array items and belong in the future data-validation stage.

## Future Derived Artifacts

The following are explicitly not fields of a canonical talk:

- Translation hypotheses.
- LISTEN/COMMIT pseudo-labels or gold labels.
- Policy features and stability features.
- Policy predictions.
- Model-specific tokenization.
- Translator or policy checkpoints.
- Experiment predictions, metrics, logs, or figures.

Source prefixes, frozen-translator hypotheses, stability analyses, and policy pseudo-labels will be derived from canonical talks and stored separately under the appropriate `data/policy/` or output locations. They must retain `talk_id` and other lineage references rather than being embedded back into the canonical talk.

## Minimal Synthetic Example

This object is synthetic and exists only to illustrate the contract. It is not real dataset content.

```json
{
  "schema_version": "1.0.0",
  "talk": {
    "talk_id": "synthetic-001",
    "title": "Synthetic schema example",
    "source_language": "en",
    "target_language": "vi",
    "provider": "synthetic",
    "duration_ms": 1500
  },
  "source": {
    "language": "en",
    "segments": [
      {
        "segment_id": "src-0000",
        "index": 0,
        "text": "We build models.",
        "start_ms": 0,
        "end_ms": 1500
      }
    ]
  },
  "target_reference": {
    "language": "vi",
    "segments": [
      {
        "segment_id": "tgt-0000",
        "index": 0,
        "text": "Chúng tôi xây dựng các mô hình."
      }
    ]
  },
  "alignments": [
    {
      "alignment_id": "align-0000",
      "source_segment_ids": ["src-0000"],
      "target_segment_ids": ["tgt-0000"],
      "method": "manual"
    }
  ],
  "stream": {
    "timing_mode": "simulated",
    "tokens": [
      {
        "token_id": "tok-0000",
        "index": 0,
        "text": "We",
        "source_segment_id": "src-0000",
        "segment_index": 0,
        "emit_ms": 500
      },
      {
        "token_id": "tok-0001",
        "index": 1,
        "text": "build",
        "source_segment_id": "src-0000",
        "segment_index": 1,
        "emit_ms": 1000
      },
      {
        "token_id": "tok-0002",
        "index": 2,
        "text": "models.",
        "source_segment_id": "src-0000",
        "segment_index": 2,
        "emit_ms": 1500
      }
    ]
  },
  "provenance": {
    "provider": "synthetic",
    "alignment": {
      "name": "manual"
    },
    "timing": {
      "name": "subtitle_uniform"
    },
    "processing_version": "m0.1-example"
  }
}
```
