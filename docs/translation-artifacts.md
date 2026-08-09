# Causal Translation Requests And Hypotheses

## Scope

M1.3 connects TimelyMT Streaming Dataset v1 to the frozen English-to-Vietnamese
translator. It creates derived request and hypothesis artifacts only. It does
not select COMMIT boundaries, generate every talk prefix, reset source buffers,
or implement Fixed-N, Fixed-Time, LocalAgreement, oracle segmentation, policy
labels, or pseudo-labels.

Dataset v1 remains the frozen bilingual research dataset under
`data/streaming/processed/`. Requests under `data/policy/prefixes/` and
hypotheses under `data/policy/hypotheses/` are disposable derived data, not a
new dataset version and not additions to Dataset v1.

## Runtime Source View

`RuntimeSourceToken` exposes exactly five fields: `talk_id`, `token_id`, global
`token_index`, lexical `text`, and `emit_ms`. `RuntimeTalk` contains an ordered
tuple of those tokens through an explicit latest-observed token index plus the
talk-level inherited split.

Canonical `source_segment_id` and `segment_index` are deliberately removed
because they reveal subtitle boundaries. The complete canonical source
reference, Vietnamese target reference, bilingual alignments, timing parameters,
and future stream suffix are also absent. This is a narrow immutable type, so
translator-facing code cannot accidentally inspect unsafe canonical fields.

`load_runtime_talk` additionally verifies that the talk exists in the Dataset v1
manifest and that its canonical content checksum matches the frozen manifest.
`runtime_talk_from_canonical` is the lower-level sanitizer used after canonical
validation; neither function mutates the canonical document.

## Inclusive Source Spans

`make_translation_request(runtime_talk, start_token_index, end_token_index,
translator=identity)` accepts a caller-selected inclusive span. Start may be
nonzero. Both indices must exist in the observed runtime view, start must not
exceed end, and end must not exceed the latest observed token. M1.3 never chooses
these indices for the caller.

The observation timestamp is exactly `emit_ms` of the inclusive end token. This
is the time by which every token in a valid monotonic span has arrived. No future
timestamp is consulted or retained.

## Source Reconstruction

There is one reconstruction rule: preserve each runtime lexical token's text
exactly and join consecutive token texts with one ASCII space (`" "`). No
trimming, case normalization, punctuation inference, sentence-final insertion,
source-reference lookup, alignment lookup, or future context is allowed.

For example, canonical reference `Hello, WORLD!` yields runtime lexical source
`Hello WORLD`. Punctuation absent from the runtime tokens is not restored.

## Request Identity

Request IDs have the form `trq-<sha256>`. The digest covers request identity
version, `talk_id`, inclusive start and end token indices, frozen translator
model ID and pinned revision, translator config version and full config
fingerprint, and generation-config fingerprint. Canonical sorted compact JSON
is hashed. Artifact creation time is not an input, so identical snapshots,
spans, and translator configurations produce identical IDs across reruns.

## Hypothesis Rows

`translate_requests` sends only each exact request `source_text` through the
existing frozen `Translator.translate_batch` API. It validates translator
identity, rejects duplicate request IDs before inference, processes engineering
micro-batches in input order, checks that translator output preserved source
text, and creates `TranslationHypothesis` rows.

Each hypothesis records request/talk/split/span/timestamp identity, exact source
and translated text, canonical source span count, translator target token count
when available, frozen model/revision/config fingerprints, and optional
device/dtype/cache-hit metadata. It contains no gold Vietnamese, target
reference, alignment, quality metric, policy label, COMMIT/LISTEN decision,
future stability label, or future hypothesis.

The EnViT5 `en:` source-language control prefix and generated `vi:`
target-language control prefix are internal model controls; neither belongs to
TimelyMT's semantic source or target text. The translator removes only an exact
leading generated `vi:` plus at most one following ASCII space before cache or
artifact consumption. `translated_text` is always this normalized hypothesis,
and downstream stability, policy, and evaluation must use it rather than raw
model output. Raw decoded text is not persisted; the deterministic rule is
identified by translator config version `1.1.0` and its full fingerprint.

`target_token_count`, when available, counts normalized `translated_text` with
the EnViT5 tokenizer and special tokens enabled. It excludes the removed `vi:`
control tag and therefore describes the same public hypothesis consumed by
downstream experiments.

The canonical source span count is distinct from the frozen translator's model
tokenizer count. M1.3 records the number of runtime lexical tokens because that
is the causal span represented by the request.

## Splits And Provenance

Every runtime talk inherits exactly one split by `talk_id` from
`data/splits/experimental.json`. Tokens, spans, requests, and hypotheses are
never independently split. Missing talk assignments fail rather than creating a
new partition.

The companion derived-artifact manifest records:

- artifact schema/type/path, ordered row count, source talk IDs, and SHA-256
  checksum of the JSONL bytes
- `TimelyMT Streaming Dataset v1`, snapshot version/checksum, and snapshot
  manifest path
- experimental split manifest path and stable semantic checksum
- frozen translator model ID, pinned revision, config version, full config
  fingerprint, and generation-config fingerprint
- creation time as explicitly non-semantic manifest metadata

Rows retain translator identity for standalone safety; dataset and split
provenance stays at manifest level to avoid duplicating the full snapshot
identity in every row. JSONL uses UTF-8, sorted keys, compact deterministic JSON,
and one LF-terminated object per row.

## Deferred Decisions

Full-dataset prefix generation is intentionally deferred. M1.3 provides no loop
over all token positions, rolling window, 512-token source policy, reset rule,
sentence boundary rule, punctuation boundary rule, alignment-unit boundary, or
oracle boundary. Later baseline and policy stages must explicitly choose causal
spans and call this infrastructure; those stages must continue to avoid target,
alignment, future-source, and split leakage.
